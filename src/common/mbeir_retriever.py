"""
This module contains the code for indexing and retrieval using FAISS based on the MBEIR embeddings.
"""

import os
import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from datetime import datetime
import json

import numpy as np
import csv
import gc

import faiss
import pickle
import torch

from data.preprocessing.utils import (
    load_jsonl_as_list,
    save_list_as_jsonl,
    count_entries_in_file,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_dataset_stats,
    unhash_did,
    unhash_qid,
    get_mbeir_task_name,
    DATASET_CAN_NUM_UPPER_BOUND,  # <-- import upper bound
)
import dist_utils
from interactive_retriever import InteractiveRetriever


def create_index(config):
    """This script builds the faiss index for the embeddings generated"""
    uniir_dir = config.uniir_dir
    index_config = config.index_config
    embed_dir_name = index_config.embed_dir_name
    index_dir_name = index_config.index_dir_name
    expt_dir_name = config.experiment.path_suffix

    idx_cand_pools_config = index_config.cand_pools_config
    assert idx_cand_pools_config.enable_idx, "Indexing is not enabled for candidate pool"
    split_name = "cand_pool"
    cand_pool_name_list = idx_cand_pools_config.cand_pools_name_to_idx

    # Pretty Print dataset to index
    print("-" * 30)
    print(f"Split: {split_name}, Candidate pool to index: {cand_pool_name_list}")
    print("-" * 30)

    for cand_pool_name in cand_pool_name_list:
        cand_pool_name = cand_pool_name.lower()

        embed_data_file = f"mbeir_{cand_pool_name}_{split_name}_embed.npy"
        embed_data_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, embed_data_file)
        embed_data_hashed_id_file = f"mbeir_{cand_pool_name}_{split_name}_ids.npy"
        embed_data_hashed_id_path = os.path.join(
            uniir_dir,
            embed_dir_name,
            expt_dir_name,
            split_name,
            embed_data_hashed_id_file,
        )

        print(f"Building index for {embed_data_path} and {embed_data_hashed_id_path}")

        # Load the embeddings and IDs from the .npy files
        embedding_list = np.load(embed_data_path).astype("float32")
        hashed_id_list = np.load(embed_data_hashed_id_path)
        print(f"Candidate pool embeddings shape: {embedding_list.shape}")
        if embedding_list.sum() == 0:
            print("WARNING: All candidate pool embeddings are zero!")
        else:
            print(f"Candidate pool embeddings - mean: {embedding_list.mean():.6f}, std: {embedding_list.std():.6f}")
            
            # Check if all candidate embeddings are identical
            if embedding_list.shape[0] > 1:
                first_cand_embedding = embedding_list[0]
                identical_count = 0
                for i in range(1, min(10, embedding_list.shape[0])):  # Check first 10
                    if np.allclose(embedding_list[i], first_cand_embedding, atol=1e-6):
                        identical_count += 1
                print(f"Identical candidate embeddings: {identical_count}/{min(9, embedding_list.shape[0]-1)} are identical to the first one")
                
                # Show some sample candidate embeddings
                print(f"First candidate embedding (first 10 values): {embedding_list[0][:10]}")
                print(f"Second candidate embedding (first 10 values): {embedding_list[1][:10]}")
                print(f"Last candidate embedding (first 10 values): {embedding_list[-1][:10]}")
                
                # DEBUG: Check L2 norms before normalization
                l2_norms_before = np.linalg.norm(embedding_list, axis=1)
                print(f"DEBUG: Candidate L2 norms before normalization - range: [{l2_norms_before.min():.6f}, {l2_norms_before.max():.6f}]")
                print(f"DEBUG: Candidate L2 norms mean: {l2_norms_before.mean():.6f}")
                
                # Check for zero vectors
                zero_vectors = np.sum(l2_norms_before == 0.0)
                print(f"DEBUG: Zero L2 norm candidate vectors: {zero_vectors}/{embedding_list.shape[0]}")
        # Convert string IDs like '1:7476031' to unique ints for FAISS
        if hashed_id_list.dtype.type is np.str_ or hashed_id_list.dtype == object:
            def strid_to_intid(s):
                dataset_id, data_within_id = map(int, s.split(":"))
                return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id
            hashed_id_list = np.array([strid_to_intid(s) for s in hashed_id_list], dtype=np.int64)
        else:
            hashed_id_list = hashed_id_list.astype(np.int64)

        # Check unique ids
        assert len(hashed_id_list) == len(set(hashed_id_list)), "IDs should be unique"

        # Normalize the embeddings
        faiss.normalize_L2(embedding_list)
        
        # DEBUG: Check L2 norms after normalization
        l2_norms_after = np.linalg.norm(embedding_list, axis=1)
        print(f"DEBUG: Candidate L2 norms after normalization - range: [{l2_norms_after.min():.6f}, {l2_norms_after.max():.6f}]")
        print(f"DEBUG: Candidate L2 norms mean: {l2_norms_after.mean():.6f}")
        
        # Check if normalization worked properly
        if not np.allclose(l2_norms_after, 1.0, atol=1e-6):
            print("WARNING: L2 normalization may not have worked properly!")
        else:
            print("DEBUG: L2 normalization successful - all vectors have unit norm")

        # Dimension of the embeddings
        d = embedding_list.shape[1]

        # Create the FAISS index on the CPU
        faiss_config = index_config.faiss_config
        assert faiss_config.dim == d, "The dimension of the index does not match the dimension of the embeddings!"
        metric = getattr(faiss, faiss_config.metric)
        cpu_index = faiss.index_factory(
            faiss_config.dim,
            f"IDMap,{faiss_config.idx_type}",
            metric,
        )
        print("Creating FAISS index with the following parameters:")
        print(f"Index type: {faiss_config.idx_type}")
        print(f"Metric: {faiss_config.metric}")
        print(f"Dimension: {faiss_config.dim}")

        # Distribute the index across multiple GPUs
        ngpus = faiss.get_num_gpus()
        print(f"Number of GPUs used for indexing: {ngpus}")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)

        # Add data to the GPU index
        index_gpu.add_with_ids(embedding_list, hashed_id_list)

        # Transfer the GPU index back to the CPU for saving
        index_cpu = faiss.index_gpu_to_cpu(index_gpu)

        # Save the CPU index to disk
        index_path = os.path.join(
            uniir_dir,
            index_dir_name,
            expt_dir_name,
            split_name,
            f"mbeir_{cand_pool_name}_{split_name}.index",
        )
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index_cpu, index_path)
        print(f"Successfully indexed {index_cpu.ntotal} documents")
        print(f"Index saved to: {index_path}")

        # 1. Delete large objects
        del embedding_list
        del hashed_id_list
        del cpu_index
        del index_gpu
        del index_cpu

        # 2. Force garbage collection
        gc.collect()


# def compute_recall_at_k(relevant_docs, retrieved_indices, k):
#     if not relevant_docs:
#         return 0.0  # Return 0 if there are no relevant documents

#     top_k_retrieved_indices_set = set(retrieved_indices[:k])
#     relevant_docs_set = set(relevant_docs)

#     assert len(relevant_docs_set) == len(relevant_docs), "Relevant docs should not contain duplicates"
#     assert len(top_k_retrieved_indices_set) == len(
#         retrieved_indices[:k]
#     ), "Retrieved docs should not contain duplicates"

#     relevant_retrieved = relevant_docs_set.intersection(top_k_retrieved_indices_set)
#     recall_at_k = len(relevant_retrieved) / len(relevant_docs)
#     return recall_at_k



def search_index(query_embed_path, cand_index_path, batch_size=10, num_cand_to_retrieve=10):
    # Load the full query embeddings
    query_embeddings = np.load(query_embed_path).astype("float32")
    print(f"Faiss: loaded query embeddings from {query_embed_path} with shape: {query_embeddings.shape}")

    # DEBUG: Check query embeddings before normalization
    print(f"DEBUG: Query embeddings before normalization - range: [{query_embeddings.min():.6f}, {query_embeddings.max():.6f}]")
    print(f"DEBUG: Query embeddings before normalization - mean: {query_embeddings.mean():.6f}, std: {query_embeddings.std():.6f}")
    
    # Check if query embeddings are all zero
    zero_query_count = np.sum(query_embeddings == 0.0)
    total_query_elements = query_embeddings.size
    print(f"DEBUG: Zero query embeddings: {zero_query_count}/{total_query_elements} ({100*zero_query_count/total_query_elements:.2f}%)")

    # Normalize the full query embeddings
    faiss.normalize_L2(query_embeddings)
    
    # DEBUG: Check query embeddings after normalization
    print(f"DEBUG: Query embeddings after normalization - range: [{query_embeddings.min():.6f}, {query_embeddings.max():.6f}]")
    print(f"DEBUG: Query embeddings after normalization - mean: {query_embeddings.mean():.6f}, std: {query_embeddings.std():.6f}")
    
    # Check L2 norms after normalization (should be 1.0 for all vectors)
    l2_norms = np.linalg.norm(query_embeddings, axis=1)
    print(f"DEBUG: L2 norms after normalization - range: [{l2_norms.min():.6f}, {l2_norms.max():.6f}]")
    print(f"DEBUG: L2 norms mean: {l2_norms.mean():.6f}")

    # Load the saved CPU index from disk
    index_cpu = faiss.read_index(cand_index_path)
    print(f"Faiss: loaded index from {cand_index_path}")
    print(f"Faiss: Number of documents in the index: {index_cpu.ntotal}")
    
    # Debug: Check if index is properly built
    if hasattr(index_cpu, 'get_xb'):
        try:
            # Try to get some vectors from the index
            sample_vectors = index_cpu.get_xb()[:5]  # Get first 5 vectors
            print(f"Sample vectors from index (first 5, first 10 values each):")
            for i, vec in enumerate(sample_vectors):
                print(f"  Vector {i}: {vec[:10]}")
                
            # Check L2 norms of index vectors
            index_l2_norms = np.linalg.norm(sample_vectors, axis=1)
            print(f"DEBUG: Index vectors L2 norms: {index_l2_norms}")
        except Exception as e:
            print(f"Could not get vectors from index: {e}")
    else:
        print("Index does not have get_xb method")

    # Use CPU-based search for reliability
    print(f"Faiss: Using CPU-based search for reliability")
    
    all_distances = []
    all_indices = []

    # Process in batches
    for i in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[i : i + batch_size]
        distances, indices = search_index_with_batch(batch, index_cpu, num_cand_to_retrieve)
        all_distances.append(distances)
        all_indices.append(indices)

    # Stack results for distances and indices
    final_distances = np.vstack(all_distances)
    final_indices = np.vstack(all_indices)
    
    # DEBUG: Final check of results
    print(f"DEBUG: Final distances shape: {final_distances.shape}")
    print(f"DEBUG: Final distances range: [{final_distances.min():.6f}, {final_distances.max():.6f}]")
    print(f"DEBUG: Final distances mean: {final_distances.mean():.6f}")

    return final_distances, final_indices


def search_index_with_batch(query_embeddings_batch, index_cpu, num_cand_to_retrieve=10):
    # Ensure query_embeddings_batch is numpy array with dtype float32
    assert isinstance(query_embeddings_batch, np.ndarray) and query_embeddings_batch.dtype == np.float32
    print(f"Faiss: query_embeddings_batch.shape: {query_embeddings_batch.shape}")

    # Query the CPU index
    distances, indices = index_cpu.search(query_embeddings_batch, num_cand_to_retrieve)  # (number_of_queries, k)
    
    # DEBUG: Verify that we're getting different results for different queries
    if query_embeddings_batch.shape[0] > 1:
        first_indices = indices[0]
        second_indices = indices[1]
        indices_different = not np.array_equal(first_indices, second_indices)
        print(f"DEBUG: First and second query indices are different: {indices_different}")
        if indices_different:
            print(f"DEBUG: First query indices: {first_indices[:5]}")
            print(f"DEBUG: Second query indices: {second_indices[:5]}")
        else:
            print("WARNING: All queries are returning the same indices!")
    
    # DEBUG: Print detailed information about the search results
    print(f"DEBUG: distances.shape: {distances.shape}, indices.shape: {indices.shape}")
    print(f"DEBUG: distances dtype: {distances.dtype}, indices dtype: {indices.dtype}")
    print(f"DEBUG: distances range: [{distances.min():.6f}, {distances.max():.6f}]")
    print(f"DEBUG: distances mean: {distances.mean():.6f}, std: {distances.std():.6f}")
    print(f"DEBUG: First query distances: {distances[0]}")
    print(f"DEBUG: First query indices: {indices[0]}")
    
    # DEBUG: Check for zero distances
    zero_distances = np.sum(distances == 0.0)
    total_distances = distances.size
    print(f"DEBUG: Zero distances: {zero_distances}/{total_distances} ({100.0 * zero_distances / total_distances:.2f}%)")
    
    # DEBUG: Check for NaN or Inf values
    nan_distances = np.sum(np.isnan(distances))
    inf_distances = np.sum(np.isinf(distances))
    print(f"DEBUG: NaN distances: {nan_distances}, Inf distances: {inf_distances}")
    
    # DEBUG: Check a few sample query embeddings and their dot products
    if query_embeddings_batch.shape[0] > 0:
        sample_query = query_embeddings_batch[0]
        print(f"DEBUG: Sample query embedding (first 10 values): {sample_query[:10]}")
        print(f"DEBUG: Sample query embedding norm: {np.linalg.norm(sample_query):.6f}")
        
        # Try to get a few candidate embeddings from the index to check dot products
        try:
            # This might not work with all FAISS index types, but worth trying
            if hasattr(index_cpu, 'get_xb'):
                cand_embeddings = index_cpu.get_xb()
                print(f"DEBUG: Candidate embeddings shape: {cand_embeddings.shape}")
                if cand_embeddings.shape[0] > 0:
                    sample_cand = cand_embeddings[0]
                    print(f"DEBUG: Sample candidate embedding (first 10 values): {sample_cand[:10]}")
                    print(f"DEBUG: Sample candidate embedding norm: {np.linalg.norm(sample_cand):.6f}")
                    
                    # Calculate dot product manually
                    dot_product = np.dot(sample_query, sample_cand)
                    print(f"DEBUG: Manual dot product between sample query and candidate: {dot_product:.6f}")
        except Exception as e:
            print(f"DEBUG: Could not access candidate embeddings from index: {e}")
    
    print(f"DEBUG: Final distances shape: {distances.shape}")
    print(f"DEBUG: Final distances range: [{distances.min():.6f}, {distances.max():.6f}]")
    print(f"DEBUG: Final distances mean: {distances.mean():.6f}")
    
    return distances, indices


def get_raw_retrieved_candidates(
    queries_path, candidates_path, retrieved_indices, hashed_query_ids, complement_retriever
):
    # Load raw queries
    qid_to_queries = {}
    with open(queries_path, "r") as f:
        for l in f:
            q = json.loads(l.strip())
            assert q["qid"] not in qid_to_queries, "qids must be unique"
            qid_to_queries[q["qid"]] = q

    # Load raw candidates
    did_to_candidates = {}
    with open(candidates_path, "r") as f:
        for l in f:
            c = json.loads(l.strip())
            assert c["did"] not in did_to_candidates, "dids must be unique"
            did_to_candidates[c["did"]] = c

    retrieved_dict = {}
    complement_queries_list = []  # Used to map complement queries to original qids.
    for idx, indices in enumerate(retrieved_indices):
        retrieved_cands = []
        qid = unhash_qid(hashed_query_ids[idx])
        query = qid_to_queries[qid]
        for hashed_doc_id in indices:
            doc_id = unhash_did(hashed_doc_id)
            retrieved_cands.append(did_to_candidates[doc_id])
        retrieved_dict[qid] = {"query": query, "candidates": retrieved_cands}
        # For each candidate with image/text modality create a complement query to retrieve the candidate's complement candidate with text/image modality.
        if complement_retriever:
            complement_modalities = {"text": "image", "image": "text"}
            complement_queries = [
                (
                    cand.get("modality"),
                    cand.get("txt"),
                    cand.get("img_path"),
                    complement_modalities[cand.get("modality")],
                )
                for cand in retrieved_cands
                if cand["modality"] in complement_modalities.keys()
            ]
            complement_queries_list.append((qid, complement_queries))
            complement_retriever.add_queries(complement_queries)

    # Retrieve complement candidates for all queries at once.
    if complement_retriever:
        retrieved_complements = complement_retriever.retrieve(k=10)
        complement_queries_start_index = 0
        for qid, complement_queries in complement_queries_list:
            complement_candidates = []
            complement_queries_end_index = complement_queries_start_index + len(complement_queries)
            retrieved_comp_cands = retrieved_complements[complement_queries_start_index:complement_queries_end_index]
            complement_queries_start_index = complement_queries_end_index
            for idx, complement_query in enumerate(complement_queries):
                complement_cand = None
                q_modality = complement_query[0]
                for cand in retrieved_comp_cands[idx]:
                    if cand["modality"] == complement_modalities[q_modality]:
                        # The retrieved complement candidate should not be the same as the original query.
                        if (
                            cand.get("img_path")
                            and cand.get("img_path") != retrieved_dict[qid]["query"]["query_img_path"]
                        ):
                            complement_cand = cand
                            break
                        if cand.get("txt") and cand.get("txt") != retrieved_dict[qid]["query"]["query_txt"]:
                            complement_cand = cand
                            break
                if not complement_cand:
                    print(f"retrieved_dict[qid]: {retrieved_dict[qid].__repr__()}")
                    print(f"retrieved_comp_cands: {retrieved_comp_cands[idx]}")
                complement_candidates.append(complement_cand)
            retrieved_dict[qid]["complement_candidates"] = complement_candidates
    return retrieved_dict


def run_retrieval(config, query_embedder_config=None):
    """This script runs retrieval on the faiss index"""
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    embed_dir_name = retrieval_config.embed_dir_name
    index_dir_name = retrieval_config.index_dir_name
    query_dir_name = retrieval_config.query_dir_name
    candidate_dir_name = retrieval_config.candidate_dir_name
    expt_dir_name = config.experiment.path_suffix

    # Create results directory if it doesn't exist
    results_dir_name = retrieval_config.results_dir_name
    exp_results_dir = os.path.join(uniir_dir, results_dir_name, expt_dir_name)
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_run_file_dir = os.path.join(exp_results_dir, "run_files")
    os.makedirs(exp_run_file_dir, exist_ok=True)
    exp_retrieved_cands_dir = os.path.join(exp_results_dir, "retrieved_candidates")
    os.makedirs(exp_retrieved_cands_dir, exist_ok=True)
    exp_tsv_results_dir = os.path.join(exp_results_dir, "final_tsv")
    os.makedirs(exp_tsv_results_dir, exist_ok=True)

    splits = []
    # Load the dataset splits to embed
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        retrieval_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if retrieval_dataset_config and retrieval_dataset_config.enable_retrieve:
            dataset_name_list = getattr(retrieval_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(retrieval_dataset_config, "correspond_cand_pools_name", None)
            dataset_embed_dir = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name)
            splits.append(
                (
                    split_name,
                    dataset_embed_dir,
                    dataset_name_list,
                    cand_pool_name_list,
                )
            )

    # Pretty Print dataset to index
    print("-" * 30)
    for (
        split_name,
        dataset_embed_dir,
        dataset_name_list,
        cand_pool_name_list,
    ) in splits:
        print(
            f"Split: {split_name}, Retrieval Datasets: {dataset_name_list}, Candidate Pools: {cand_pool_name_list}"
        )
        print("-" * 30)

    cand_index_dir = os.path.join(uniir_dir, index_dir_name, expt_dir_name, "cand_pool")
    for (
        split,
        dataset_embed_dir,
        dataset_name_list,
        cand_pool_name_list,
    ) in splits:
        for dataset_name, cand_pool_name in zip(
            dataset_name_list, cand_pool_name_list
        ):
            print("\n" + "-" * 30)
            print(f"Retriever: Retrieving for query:{dataset_name} | split:{split} | from cand_pool:{cand_pool_name}")

            dataset_name = dataset_name.lower()
            cand_pool_name = cand_pool_name.lower()

            # Load query Hashed IDs
            embed_query_id_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{split}_ids.npy")
            hashed_query_ids = np.load(embed_query_id_path)

            # Load query embeddings
            embed_query_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{split}_embed.npy")

            # Load the candidate pool index
            cand_index_path = os.path.join(cand_index_dir, f"mbeir_{cand_pool_name}_cand_pool.index")

            # Search the index
            k = 10  # Default k value
            print(f"Retriever: Searching with k={k}")
            retrieved_cand_dist, retrieved_indices = search_index(
                embed_query_path,
                cand_index_path,
                # batch_size=hashed_query_ids.shape[0], #TODO
                batch_size=64,  # or 16, 64, etc. -- tune for your GPU
                num_cand_to_retrieve=10,
            )

            # Open a file to write the run results
            if cand_pool_name == "union":
                run_id = f"mbeir_{dataset_name}_union_pool_{split}_k{k}"
            else:
                run_id = f"mbeir_{dataset_name}_single_pool_{split}_k{k}"
            run_file_name = f"{run_id}_run.txt"
            run_file_path = os.path.join(exp_run_file_dir, run_file_name)
            
            # DEBUG: Print information about the retrieved results before writing
            print(f"DEBUG: retrieved_cand_dist.shape: {retrieved_cand_dist.shape}")
            print(f"DEBUG: retrieved_indices.shape: {retrieved_indices.shape}")
            print(f"DEBUG: retrieved_cand_dist range: [{retrieved_cand_dist.min():.6f}, {retrieved_cand_dist.max():.6f}]")
            print(f"DEBUG: retrieved_cand_dist mean: {retrieved_cand_dist.mean():.6f}")
            
            # Check if all scores are zero
            zero_score_count = np.sum(retrieved_cand_dist == 0.0)
            total_scores = retrieved_cand_dist.size
            print(f"DEBUG: Zero scores in retrieved_cand_dist: {zero_score_count}/{total_scores} ({100*zero_score_count/total_scores:.2f}%)")
            
            # Show first few scores
            if retrieved_cand_dist.shape[0] > 0:
                print(f"DEBUG: First query scores: {retrieved_cand_dist[0]}")
                print(f"DEBUG: First query indices: {retrieved_indices[0]}")
            
            with open(run_file_path, "w") as run_file:
                for idx, (distances, indices) in enumerate(zip(retrieved_cand_dist, retrieved_indices)):
                    qid = unhash_qid(hashed_query_ids[idx])
                    for rank, (hashed_doc_id, score) in enumerate(zip(indices, distances), start=1):
                        # Format: query-id Q0 document-id rank score run-id
                        # Note: since we are using the cosine similarity, we don't need to invert the scores.
                        doc_id = unhash_did(hashed_doc_id)
                        
                        # DEBUG: Print first few lines being written
                        if idx < 3 and rank <= 3:
                            print(f"DEBUG: Writing line - qid: {qid}, doc_id: {doc_id}, rank: {rank}, score: {score}")
                        
                        run_file_line = f"{qid} Q0 {doc_id} {rank} {score} {run_id}\n"
                        run_file.write(run_file_line)
            print(f"Retriever: Run file saved to {run_file_path}")

            # Store raw retrieved candidates for downstream applications like UniRAG
            if retrieval_config.raw_retrieval:
                queries_path = os.path.join(
                    mbeir_data_dir,
                    query_dir_name,
                    f"{split}/mbeir_{dataset_name}_{split}.jsonl",
                )
                candidates_path = os.path.join(
                    mbeir_data_dir, candidate_dir_name, f"mbeir_{cand_pool_name}_{split}_cand_pool.jsonl"
                )
                # When retrieving complement candidates, we want to find the closest image to a candidate caption or vica versa to have image-text pairs.
                # "MSCOCO" dataset supports both image->text and text->image queries.
                dataset_name = "MSCOCO"
                complement_retriever = (
                    InteractiveRetriever(cand_index_path, candidates_path, dataset_name, query_embedder_config)
                    if retrieval_config.retrieve_image_text_pairs
                    else None
                )
                retrieved_dict = get_raw_retrieved_candidates(
                    queries_path, candidates_path, retrieved_indices, hashed_query_ids, complement_retriever
                )
                retrieved_file_name = f"{run_id}_retrieved.jsonl"
                retrieved_file_path = os.path.join(exp_retrieved_cands_dir, retrieved_file_name)
                with open(retrieved_file_path, "w") as retrieved_file:
                    for _, v in retrieved_dict.items():
                        json.dump(v, retrieved_file)
                        retrieved_file.write("\n")
                print(f"Retriever: Retrieved file saved to {retrieved_file_path}")




def run_hard_negative_mining(config):
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    expt_dir_name = config.experiment.path_suffix
    embed_dir_name = retrieval_config.embed_dir_name
    index_dir_name = retrieval_config.index_dir_name
    hard_negs_dir_name = retrieval_config.hard_negs_dir_name

    # Query data file name
    retrieval_train_dataset_config = retrieval_config.train_datasets_config
    assert retrieval_train_dataset_config.enable_retrieve, "Hard negative mining is not enabled for training data"
    dataset_name = retrieval_train_dataset_config.datasets_name[
        0
    ].lower()  # Only extract hard negatives for the first dataset
    dataset_split_name = "train"
    # Load query data
    query_data_path = os.path.join(mbeir_data_dir, "train", f"mbeir_{dataset_name}_{dataset_split_name}.jsonl")
    query_data_list = load_jsonl_as_list(query_data_path)

    # Load query IDs
    dataset_embed_dir = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, dataset_split_name)
    embed_data_id_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{dataset_split_name}_ids.npy")
    query_ids = np.load(embed_data_id_path)

    # Load query embeddings
    embed_data_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{dataset_split_name}_embed.npy")

    # Load the candidate pool index
    cand_pool_name = retrieval_train_dataset_config.correspond_cand_pools_name[
        0
    ].lower()  # Only extract the first candidate pool
    cand_pool_split_name = "cand_pool"
    cand_index_dir = os.path.join(uniir_dir, index_dir_name, expt_dir_name, cand_pool_split_name)
    cand_index_path = os.path.join(cand_index_dir, f"mbeir_{cand_pool_name}_{cand_pool_split_name}.index")

    # Pretty Print dataset to perform hard negative mining
    print("-" * 30)
    print(f"Hard Negative mining, Datasets: {dataset_name}, Candidate Pools: {cand_pool_name}")
    print("-" * 30)

    # Search the index
    num_hard_negs = retrieval_config.num_hard_negs
    k = retrieval_config.k
    _, retrieved_indices = search_index(
        embed_data_path,
        cand_index_path,
        batch_size=query_ids.shape[0],
        num_cand_to_retrieve=k,
    )  # nested list of (number_of_queries, k)
    assert len(query_ids) == len(retrieved_indices)

    # Add hard negatives to the query data
    for i, query_id in enumerate(query_ids):
        query_data = query_data_list[i]
        query_id = unhash_qid(query_id)
        assert query_id == query_data["qid"]
        retrieved_indices_for_qid = retrieved_indices[i]
        retrieved_indices_for_qid = [unhash_did(idx) for idx in retrieved_indices_for_qid]

        pos_cand_list = query_data["pos_cand_list"]
        neg_cand_list = query_data["neg_cand_list"]

        # Add hard negatives to the query data
        hard_negatives = [
            doc_id
            for doc_id in retrieved_indices_for_qid
            if doc_id not in pos_cand_list and doc_id not in neg_cand_list
        ]

        # Ensure that hard_negatives has a minimum length of num_hard_negs
        if 0 < len(hard_negatives) < num_hard_negs:
            multiplier = num_hard_negs // len(hard_negatives)
            remainder = num_hard_negs % len(hard_negatives)
            hard_negatives = hard_negatives * multiplier + hard_negatives[:remainder]
        elif len(hard_negatives) == 0:
            print("Warning: hard_negatives list is empty.")

        # Truncate hard_negatives to a maximum length
        hard_negatives = hard_negatives[:num_hard_negs]
        query_data["neg_cand_list"].extend(hard_negatives)

    # Save the query data with hard negatives
    query_data_with_hard_negs_path = os.path.join(
        mbeir_data_dir,
        "train",
        hard_negs_dir_name,
        f"mbeir_{dataset_name}_hard_negs_{dataset_split_name}.jsonl",
    )  # mbeir_data_dir/train/hard_negs_dir_name/mbeir_agg_hard_negs_train.jsonl
    os.makedirs(os.path.dirname(query_data_with_hard_negs_path), exist_ok=True)
    save_list_as_jsonl(query_data_list, query_data_with_hard_negs_path)

    # Print statistics
    total_entries, _data = count_entries_in_file(query_data_with_hard_negs_path)
    print(f"MBEIR Train Data with Hard Negatives saved to {query_data_with_hard_negs_path}")
    print(f"Total number of entries in {query_data_with_hard_negs_path}: {total_entries}")
    cand_pool_path = os.path.join(
        mbeir_data_dir,
        cand_pool_split_name,
        f"mbeir_{cand_pool_name}_{cand_pool_split_name}.jsonl",
    )
    cand_pool = load_mbeir_format_pool_file_as_dict(cand_pool_path, doc_key_to_content=True, key_type="did")
    print_mbeir_format_dataset_stats(_data, cand_pool)


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Pipeline")
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--query_embedder_config_path",
        default="",
        help="Path to the query embedder config file. Used when retrieving candidates with complement modalities in raw_retrieval mode.",
    )
    parser.add_argument("--enable_create_index", action="store_true", help="Enable create index")
    parser.add_argument(
        "--enable_hard_negative_mining",
        action="store_true",
        help="Enable hard negative mining",
    )
    parser.add_argument("--enable_retrieval", action="store_true", help="Enable retrieval")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    print(OmegaConf.to_yaml(config, sort_keys=False))

    query_embedder_config = None  # Fix: Always define this variable
    interactive_retrieval = True if args.query_embedder_config_path else False
    if interactive_retrieval:
        query_embedder_config = OmegaConf.load(args.query_embedder_config_path)
        query_embedder_config.uniir_dir = args.uniir_dir
        query_embedder_config.mbeir_data_dir = args.mbeir_data_dir
        # Initialize distributed mode
        args.dist_url = query_embedder_config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
        dist_utils.init_distributed_mode(args)
        query_embedder_config.dist_config.gpu_id = args.gpu
        query_embedder_config.dist_config.distributed_mode = args.distributed

    if args.enable_hard_negative_mining:
        run_hard_negative_mining(config)

    if args.enable_create_index:
        create_index(config)

    if args.enable_retrieval:
        run_retrieval(config, query_embedder_config)

    # Destroy the process group
    if interactive_retrieval and query_embedder_config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
