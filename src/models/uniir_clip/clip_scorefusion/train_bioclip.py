"""
Training Code for BioCLIP-SF with BioCLIP repository patterns
"""

# Standard library
import argparse
import logging
import os
import random
import math
import time

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb
import torch.nn.functional as F

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)
from models.uniir_clip import utils
from bioclip_sf import BioCLIPScoreFusion

# Set up logger
logger = logging.getLogger()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_parameters(model, condition_fn):
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]


def create_optimizer(gain_or_bias_params, rest_params, config):
    return optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=config.trainer_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch_bioclip(model, data, epoch, optimizer, scaler, scheduler, config):
    device = torch.device(f"cuda:{config.dist_config.gpu_id}")
    model.train()

    data.set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data
    num_batches_per_epoch = len(dataloader)
    sample_digits = math.ceil(math.log(len(dataloader.dataset) + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not hasattr(config.trainer_config, 'skip_scheduler') or not config.trainer_config.skip_scheduler:
            scheduler(step)

        # Get data from batch
        txt_batched = batch["txt_batched"].to(device=device, non_blocking=True)
        image_batched = batch["image_batched"].to(device=device, non_blocking=True)
        txt_mask_batched = batch["txt_mask_batched"].to(device=device, non_blocking=True)
        image_mask_batched = batch["image_mask_batched"].to(device=device, non_blocking=True)
        index_mapping = batch["index_mapping"]

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # Forward pass
        model_out = model(batch)
        loss = model_out["loss"]
        accuracy = model_out["accuracy"]

        # Backward pass
        backward(loss, scaler)

        if scaler is not None:
            if hasattr(config, 'horovod') and config.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if hasattr(config, 'grad_clip_norm') and config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if hasattr(config, 'grad_clip_norm') and config.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if hasattr(config, 'grad_clip_norm') and config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).clip_model.logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if utils.is_main_process() and (i % 10 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(txt_batched)
            num_samples = batch_count * batch_size * config.dist_config.world_size
            samples_per_epoch = len(dataloader.dataset)
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in [("loss", loss.item()), ("accuracy", accuracy.item())]:
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val, batch_size)

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = config.dataloader_config.train_batch_size * config.dist_config.world_size / batch_time_m.val
            samples_per_second_per_gpu = config.dataloader_config.train_batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name: val.val for name, val in losses_m.items()})

            if config.wandb_config.enabled:
                wandb.log(log_data)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

    return {name: meter.avg for name, meter in losses_m.items()}


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def evaluate_bioclip(model, val_loader, epoch, config):
    if not utils.is_main_process():
        return {}
    
    device = torch.device(f"cuda:{config.dist_config.gpu_id}")
    model.eval()

    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Get data from batch
            txt_batched = batch["txt_batched"].to(device=device, non_blocking=True)
            image_batched = batch["image_batched"].to(device=device, non_blocking=True)
            txt_mask_batched = batch["txt_mask_batched"].to(device=device, non_blocking=True)
            image_mask_batched = batch["image_mask_batched"].to(device=device, non_blocking=True)
            index_mapping = batch["index_mapping"]

            # Forward pass
            model_out = model(batch)
            loss = model_out["loss"]
            accuracy = model_out["accuracy"]

            batch_size = txt_batched.shape[0]
            cumulative_loss += loss.item() * batch_size
            cumulative_accuracy += accuracy.item() * batch_size
            num_samples += batch_size

            if utils.is_main_process() and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {len(val_loader.dataset)}]\t"
                    f"Loss: {cumulative_loss / num_samples:.6f}\t"
                    f"Accuracy: {cumulative_accuracy / num_samples:.6f}\t"
                )

    loss = cumulative_loss / num_samples
    accuracy = cumulative_accuracy / num_samples
    
    metrics = {
        "val_loss": loss, 
        "val_accuracy": accuracy,
        "epoch": epoch, 
        "num_samples": num_samples
    }

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if config.wandb_config.enabled:
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
):
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = config.dist_config.distributed_mode
    best_accuracy = 0.0
    best_epoch = 0

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch
        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_bioclip(
            model,
            train_loader,
            epoch,
            optimizer,
            scaler,
            scheduler,
            config,
        )

        eval_freq = config.evaluator.eval_freq
        if val_loader is None or epoch % eval_freq != 0:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch)
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            val_stats = evaluate_bioclip(model, val_loader, epoch, config)
            accuracy = val_stats.get("val_accuracy", 0.0)
            
            # Note: still save the model even if the accuracy is not the best
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            
            log_stats = log_results(train_stats, val_stats, None, epoch, best_epoch)

        if utils.is_main_process():
            if config.wandb_config.enabled:
                wandb.log(log_stats)

        dist.barrier()  # Wait for the master process to finish writing the log file
        torch.cuda.empty_cache()


def main(config):
    is_distributed_mode = config.dist_config.distributed_mode

    # Set up seed for reproducibility
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    cudnn.benchmark = True

    # Initialize and load model
    print("Creating BioCLIP-SF model...")
    model_config = config.model
    pretrained_clip_model_dir = os.path.join(config.uniir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading BioCLIP model to {pretrained_clip_model_dir}...")
    model = BioCLIPScoreFusion(
        model_name=model_config.clip_vision_model_name,
        device="cuda",
        config=config,
    )
    model.float()  # Convert to fp32 for training

    # Set up optimizer, and scaler
    # Apply different optimization strategies to different parameters
    # This is adapted from the UniVL-DR codebase
    exclude_condition = lambda n, p: p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    include_condition = lambda n, p: not exclude_condition(n, p)
    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    optimizer = create_optimizer(gain_or_bias_params, rest_params, config)
    scaler = GradScaler()  # Initialize the GradScaler

    # If resume training, load the checkpoint
    ckpt_config = model_config.ckpt_config
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"loading BioCLIPScoreFusion checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # Move model to GPUs
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_dataset, train_collector = build_mbeir_dataset_from_config(
        config=config,
        tokenizer=tokenizer,
        img_preprocess_fn=img_preprocess_fn,
        dataset_type=DatasetType.MAIN_TRAIN,
    )
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,  # Note: since we use sampler, shuffle should be False
        collate_fn=train_collector,
        drop_last=True,
    )

    enable_eval = config.evaluator.enable_eval
    valid_loader = None
    if enable_eval:
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )
        in_batch_val_sampler = DistributedSampler(
            dataset=in_batch_val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,  # Note: since we use sampler, shuffle should be False
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Initializing the scheduler
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)

    epoch = 0
    if ckpt_config.resume_training:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1

    # Training loop
    dist.barrier()
    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--uniir_dir",
        type=str,
        default="/data/UniIR",
        help="Path to UniIR directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    # Set up wandb
    if config.wandb_config.enabled and utils.is_main_process():
        load_dotenv()  # Load .env and get WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY
        wandb_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")

        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found. Ensure it's set in the .env file.")

        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=config.wandb_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Set up logger
    if utils.is_main_process():
        logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
        logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
        if not os.path.exists(logger_out_dir):
            os.makedirs(logger_out_dir, exist_ok=True)
        handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info(config)

    main(config)

    # Close wandb
    if config.wandb_config.enabled and utils.is_main_process():
        wandb.finish()

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group() 