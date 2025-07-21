"""
Utility functions for retrieval experiments on MBEIR.
"""

# Standard Library imports
import os
import random

# Third-party imports
import numpy as np
import torch


def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if (
                int(relevance_score) > 0
            ):  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Loaded {len(qrel)} queries from {filename}")
    print(
        f"Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid


# TODO: Write a hashed id to unhased id converter.
def load_runfile(filename, load_task_id=False):
    run_results = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            if qid not in run_results:
                run_results[qid] = []
            if load_task_id:
                qid, _, did, rank, score, run_id, task_id = parts
                run_results[qid].append(
                    {
                        "did": did,
                        "rank": int(rank),
                        "score": float(score),
                        "task_id": task_id,
                    }
                )
            else:
                parts = parts[:6]
                qid, _, did, rank, score, run_id = parts
                run_results[qid].append(
                    {
                        "did": did,
                        "rank": int(rank),
                        "score": float(score),
                    }
                )
    print(f"Loaded results for {len(run_results)} queries from {filename}")
    return run_results


def build_model_from_config(config):
    model_name = config.model.name

    if model_name == "BioCLIPScoreFusion":  # This does not work
        from models.uniir_clip.clip_scorefusion.bioclip_sf import BioCLIPScoreFusion

        # initialize BioCLIPScoreFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        print(f"Creating BioCLIPScoreFusion model...")
        model = BioCLIPScoreFusion(
            model_name=model_config.clip_vision_model_name,
            device="cuda",
            config=config,
        )
        model.float()
        # Convert to fp32 for training

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(
            config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name
        )
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BioCLIPScoreFusion checkpoint from {checkpoint_path}")
        import torch.serialization
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata
        from typing import Any
        import collections
        from omegaconf.nodes import AnyNode
        from omegaconf.base import Metadata

        torch.serialization.add_safe_globals(
            [
                DictConfig,
                ContainerMetadata,
                Any,
                dict,
                collections.defaultdict,
                AnyNode,
                Metadata,
            ]
        )
        ckpt = torch.load(checkpoint_path)
        print(
            "Checkpoint keys:", list(ckpt["model"].keys())[:20]
        )  # print first 20 keys
        print("Model keys:", list(model.state_dict().keys())[:20])
        model.load_state_dict(ckpt["model"], strict=False)

    elif model_name == "CLIPScoreFusion":
        from models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion

        # initialize CLIPScoreFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        download_root = os.path.join(uniir_dir, model_config.pretrained_clip_model_dir)
        print(f"Downloading CLIP model to {download_root}...")
        model = CLIPScoreFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=download_root,
        )
        model.float()
        # The origial CLIP was in fp16 so we need to convert it to fp32

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(
            config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name
        )
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        import torch.serialization
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata
        from typing import Any
        import collections
        from omegaconf.nodes import AnyNode
        from omegaconf.base import Metadata

        torch.serialization.add_safe_globals(
            [
                DictConfig,
                ContainerMetadata,
                Any,
                dict,
                collections.defaultdict,
                AnyNode,
                Metadata,
            ]
        )
        ckpt = torch.load(checkpoint_path)
        print(
            "Checkpoint keys:", list(ckpt["model"].keys())[:20]
        )  # print first 20 keys
        print("Model keys:", list(model.state_dict().keys())[:20])
        model.load_state_dict(ckpt["model"], strict=False)

    elif model_name == "CLIPFeatureFusion":
        from models.uniir_clip.clip_featurefusion.clip_ff import CLIPFeatureFusion

        # initialize CLIPFeatureFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        download_root = os.path.join(uniir_dir, model_config.pretrained_clip_model_dir)
        print(f"Downloading CLIP model to {download_root}...")
        model = CLIPFeatureFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=download_root,
        )
        model.float()
        # The origial CLIP was in fp16 so we need to convert it to fp32

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(
            config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name
        )
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPFeatureFusion checkpoint from {checkpoint_path}")
        import torch.serialization
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata
        from typing import Any
        import collections
        from omegaconf.nodes import AnyNode
        from omegaconf.base import Metadata

        torch.serialization.add_safe_globals(
            [
                DictConfig,
                ContainerMetadata,
                Any,
                dict,
                collections.defaultdict,
                AnyNode,
                Metadata,
            ]
        )
        ckpt = torch.load(checkpoint_path)
        print(
            "Checkpoint keys:", list(ckpt["model"].keys())[:20]
        )  # print first 20 keys
        print("Model keys:", list(model.state_dict().keys())[:20])
        model.load_state_dict(ckpt["model"], strict=False)

    elif model_name == "BLIPScoreFusion":
        from models.uniir_blip.blip_scorefusion.blip_sf import BLIPScoreFusion

        model_config = config.model
        model = BLIPScoreFusion(
            med_config=os.path.join(
                "../models/uniir_blip", "backbone/configs/med_config.json"
            ),
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(
            config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name
        )
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BLIPScoreFusion checkpoint from {checkpoint_path}")
        import torch.serialization

        ckpt = torch.load(checkpoint_path)
        print(
            "Checkpoint keys:", list(ckpt["model"].keys())[:20]
        )  # print first 20 keys
        print("Model keys:", list(model.state_dict().keys())[:20])
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata
        from typing import Any

        torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, Any, dict])
        model.load_state_dict(torch.load(checkpoint_path)["model"])

    elif model_name == "BLIPFeatureFusion":
        from models.uniir_blip.blip_featurefusion.blip_ff import BLIPFeatureFusion

        model_config = config.model
        model = BLIPFeatureFusion(
            med_config=os.path.join(
                "../models/uniir_blip", "backbone/configs/med_config.json"
            ),
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(
            config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name
        )
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BLIPFeatureFusion checkpoint from {checkpoint_path}")
        import torch.serialization

        ckpt = torch.load(checkpoint_path)
        print(
            "Checkpoint keys:", list(ckpt["model"].keys())[:20]
        )  # print first 20 keys
        print("Model keys:", list(model.state_dict().keys())[:20])
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata
        from typing import Any

        torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, Any, dict])
        model.load_state_dict(torch.load(checkpoint_path)["model"])
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
        # Notes: Add other models here
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
