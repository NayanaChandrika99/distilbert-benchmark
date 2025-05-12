#!/usr/bin/env python
"""
Profile batch size performance for DistilBERT benchmarks.

This script runs detailed profiling of different batch sizes to identify bottlenecks
and understand anomalies in the batch-size scaling behavior.
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List

# Add parent directory to sys.path to allow importing from src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_model
from src.runner import prepare_dataset
from src.metrics.profiler import run_detailed_profiling
import torch
from torch.utils.data import DataLoader


def profile_batch_sizes(
    batch_sizes: List[int],
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cpu",
    dataset_name: str = "glue",
    dataset_subset: str = "sst2",
    dataset_split: str = "validation",
    max_length: int = 128,
    cache_dir: str = "data/cached",
    iterations: int = 10,
    warmup_runs: int = 2,
    output_file: str = "results/analysis/batch_profile.jsonl",
):
    """
    Profile performance characteristics for different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to profile
        model_name: HuggingFace model name
        device: Device to run on ("cpu" or "cuda:n")
        dataset_name: Dataset name
        dataset_subset: Dataset subset
        dataset_split: Dataset split
        max_length: Maximum sequence length
        cache_dir: Directory to cache datasets and models
        iterations: Number of profiling iterations
        warmup_runs: Number of warmup runs before profiling
        output_file: Path to save profiling results
    """
    # Load model
    print(f"Loading model: {model_name} on {device}")
    model = load_model(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )
    model.eval()
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}/{dataset_subset}/{dataset_split}")
    dataset = prepare_dataset(
        dataset_name=dataset_name,
        subset=dataset_subset,
        split=dataset_split,
        model_name=model_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Clear the output file
    with open(output_file, "w") as f:
        pass
    
    # Create DataLoaders and profile each batch size
    results = []
    for batch_size in tqdm(batch_sizes, desc="Profiling batch sizes"):
        print(f"\nProfiling batch size: {batch_size}")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,  # Disable multiprocessing for more accurate measurements
        )
        
        # Run detailed profiling
        profile_metrics = run_detailed_profiling(
            model=model,
            dataloader=dataloader,
            device=device,
            iterations=iterations,
            warmup_runs=warmup_runs,
        )
        
        # Add batch size
        profile_metrics["batch_size"] = batch_size
        
        # Print summary
        print(f"Batch size: {batch_size}")
        print(f"  Forward pass time: {profile_metrics['forward_pass_time_ms_mean']:.2f} ms ({profile_metrics.get('forward_pass_percent', 0):.1f}%)")
        print(f"  Data movement time: {profile_metrics['data_movement_time_ms_mean']:.2f} ms ({profile_metrics.get('data_movement_percent', 0):.1f}%)")
        print(f"  Preprocessing time: {profile_metrics['preprocessing_time_ms_mean']:.2f} ms ({profile_metrics.get('preprocessing_percent', 0):.1f}%)")
        print(f"  Total time: {profile_metrics['total_time_ms']:.2f} ms")
        
        # Save to file
        with open(output_file, "a") as f:
            f.write(json.dumps(profile_metrics) + "\n")
        
        results.append(profile_metrics)
    
    print(f"Profiling results saved to {output_file}")
    return results


def main():
    """Parse arguments and run profiling."""
    parser = argparse.ArgumentParser(description="Profile batch size performance")
    parser.add_argument(
        "--batch-sizes", type=str, default="1,2,4,8",
        help="Comma-separated list of batch sizes to profile"
    )
    parser.add_argument(
        "--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Model to profile"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (cpu, cuda:0, etc.)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of profiling iterations"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=2,
        help="Number of warmup runs before profiling"
    )
    parser.add_argument(
        "--output", type=str, default="results/analysis/batch_profile.jsonl",
        help="Path to save profiling results"
    )
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    # Run profiling
    profile_batch_sizes(
        batch_sizes=batch_sizes,
        model_name=args.model,
        device=args.device,
        iterations=args.iterations,
        warmup_runs=args.warmup_runs,
        output_file=args.output,
    )


if __name__ == "__main__":
    main() 