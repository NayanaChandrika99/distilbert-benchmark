#!/usr/bin/env python
"""
Analyze detailed batch profiling results.

This script analyzes the detailed profiling data to understand 
the batch-4 anomaly in performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSONL file."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metrics by batch size."""
    # Create a copy to avoid modifying the original
    norm_df = df.copy()
    
    # Normalize metrics by batch size
    if 1 in df["batch_size"].values:
        base_metrics = df[df["batch_size"] == 1]
        
        # Time per sample for batch size 1
        base_forward_time = base_metrics["forward_pass_time_ms_mean"].values[0]
        
        # Calculate expected linear scaling
        norm_df["expected_forward_time_ms"] = base_forward_time * df["batch_size"]
        
        # Calculate efficiency (ratio of expected to actual)
        norm_df["forward_pass_efficiency"] = norm_df["expected_forward_time_ms"] / norm_df["forward_pass_time_ms_mean"]
        
        # Calculate per-sample metrics
        norm_df["forward_time_per_sample_ms"] = norm_df["forward_pass_time_ms_mean"] / norm_df["batch_size"]
        
        # Normalize to batch size 1 (how many times slower is each sample compared to batch size 1)
        norm_df["forward_time_normalized"] = norm_df["forward_time_per_sample_ms"] / base_forward_time
    
    return norm_df


def plot_profiling_metrics(df: pd.DataFrame, output_dir: str):
    """Plot detailed profiling metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Forward pass time vs batch size
    plt.figure(figsize=(10, 6))
    plt.plot(df["batch_size"], df["forward_pass_time_ms_mean"], "o-", label="Actual forward pass time")
    if "expected_forward_time_ms" in df.columns:
        plt.plot(df["batch_size"], df["expected_forward_time_ms"], "--", label="Expected linear scaling")
    plt.xlabel("Batch Size")
    plt.ylabel("Forward Pass Time (ms)")
    plt.title("Forward Pass Time vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/forward_pass_time.png")
    
    # Plot 2: Forward pass time per sample
    if "forward_time_per_sample_ms" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["forward_time_per_sample_ms"], "o-")
        plt.xlabel("Batch Size")
        plt.ylabel("Forward Pass Time per Sample (ms)")
        plt.title("Per-Sample Forward Pass Time")
        plt.grid(True)
        plt.savefig(f"{output_dir}/forward_time_per_sample.png")
    
    # Plot 3: Efficiency
    if "forward_pass_efficiency" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["forward_pass_efficiency"], "o-")
        plt.axhline(y=1.0, color="r", linestyle="--", label="Perfect efficiency")
        plt.xlabel("Batch Size")
        plt.ylabel("Efficiency (expected/actual)")
        plt.title("Forward Pass Efficiency")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/forward_pass_efficiency.png")
    
    # Plot 4: Relative slowdown per sample
    if "forward_time_normalized" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["forward_time_normalized"], "o-")
        plt.axhline(y=1.0, color="r", linestyle="--", label="Batch size 1 reference")
        plt.xlabel("Batch Size")
        plt.ylabel("Relative Time per Sample")
        plt.title("Time per Sample (Normalized to Batch Size 1)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/relative_slowdown.png")
    
    # Plot 5: Forward pass time boxplot
    plt.figure(figsize=(10, 6))
    boxplot_data = []
    labels = []
    for _, row in df.sort_values("batch_size").iterrows():
        batch_size = int(row["batch_size"])
        forward_min = row["forward_pass_time_ms_min"]
        forward_max = row["forward_pass_time_ms_max"]
        forward_mean = row["forward_pass_time_ms_mean"]
        forward_median = row["forward_pass_time_ms_median"]
        forward_std = row["forward_pass_time_ms_std"]
        
        # Create box plot data [min, q1, median, q3, max]
        # Estimate q1 and q3 since we don't have them
        q1 = forward_mean - forward_std/2
        q3 = forward_mean + forward_std/2
        boxplot_data.append([forward_min, q1, forward_median, q3, forward_max])
        labels.append(f"Batch {batch_size}")
    
    plt.boxplot(boxplot_data, labels=labels, vert=True)
    plt.ylabel("Forward Pass Time (ms)")
    plt.title("Forward Pass Time Distribution by Batch Size")
    plt.grid(True, axis="y")
    plt.savefig(f"{output_dir}/forward_pass_boxplot.png")


def analyze_bottlenecks(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze batch processing bottlenecks."""
    analysis = {}
    
    # Find anomalous batch size
    if "forward_time_normalized" in df.columns:
        # The batch size with the highest per-sample time (normalized)
        anomalous_idx = df["forward_time_normalized"].idxmax()
        anomalous_batch = df.loc[anomalous_idx]
        
        # The reference batch size 1
        if 1 in df["batch_size"].values:
            reference = df[df["batch_size"] == 1].iloc[0]
            
            analysis["most_anomalous_batch"] = int(anomalous_batch["batch_size"])
            analysis["slowdown_factor"] = anomalous_batch["forward_time_normalized"]
            analysis["expected_time"] = anomalous_batch["expected_forward_time_ms"]
            analysis["actual_time"] = anomalous_batch["forward_pass_time_ms_mean"]
            
            # Compute efficiency drop
            analysis["efficiency"] = anomalous_batch["forward_pass_efficiency"]
            analysis["efficiency_drop_percent"] = (1 - analysis["efficiency"]) * 100
            
            # Time per sample comparison
            analysis["time_per_sample_reference"] = reference["forward_pass_time_ms_mean"]
            analysis["time_per_sample_anomalous"] = anomalous_batch["forward_time_per_sample_ms"]
    
    # Check if there's a non-linear jump in forward pass time
    forward_times = df.sort_values("batch_size")["forward_pass_time_ms_mean"].values
    batch_sizes = df.sort_values("batch_size")["batch_size"].values
    
    if len(forward_times) >= 3:
        # Calculate the increase ratio between consecutive batch sizes
        increase_ratios = []
        for i in range(1, len(forward_times)):
            # Ratio of increase in time divided by ratio of increase in batch size
            time_increase = forward_times[i] / forward_times[i-1]
            batch_increase = batch_sizes[i] / batch_sizes[i-1]
            increase_ratios.append(time_increase / batch_increase)
        
        # Find the largest non-linear jump
        max_ratio_idx = np.argmax(increase_ratios)
        analysis["largest_jump_batch_from"] = int(batch_sizes[max_ratio_idx])
        analysis["largest_jump_batch_to"] = int(batch_sizes[max_ratio_idx + 1])
        analysis["largest_jump_ratio"] = increase_ratios[max_ratio_idx]
    
    return analysis


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze batch profile results")
    parser.add_argument(
        "--input", type=str, default="results/analysis/batch_profile_detailed.jsonl",
        help="Path to batch profile results JSONL file"
    )
    parser.add_argument(
        "--output", type=str, default="results/analysis/batch_profile_charts",
        help="Directory to save analysis outputs"
    )
    args = parser.parse_args()
    
    # Load profiling results
    results = load_jsonl(args.input)
    if not results:
        print(f"No results found in {args.input}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print basic info
    print("\nBatch Profiling Results:")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Forward Time (ms)':<20} {'Per Sample (ms)':<20} {'Forward Min (ms)':<16} {'Forward Max (ms)':<16}")
    print("-" * 80)
    for _, row in df.sort_values("batch_size").iterrows():
        per_sample = row["forward_pass_time_ms_mean"] / row["batch_size"]
        print(f"{int(row['batch_size']):<12} {row['forward_pass_time_ms_mean']:<20.2f} {per_sample:<20.2f} {row['forward_pass_time_ms_min']:<16.2f} {row['forward_pass_time_ms_max']:<16.2f}")
    
    # Normalize metrics
    norm_df = normalize_metrics(df)
    
    # Analyze bottlenecks
    analysis = analyze_bottlenecks(norm_df)
    
    print("\nPerformance Analysis:")
    print("=" * 80)
    
    if "most_anomalous_batch" in analysis:
        print(f"Most anomalous batch size: {analysis['most_anomalous_batch']}")
        print(f"Slowdown factor: {analysis['slowdown_factor']:.2f}x per sample")
        print(f"Efficiency: {analysis['efficiency']:.2f} ({analysis['efficiency_drop_percent']:.1f}% efficiency drop)")
        print(f"Expected time: {analysis['expected_time']:.2f} ms")
        print(f"Actual time: {analysis['actual_time']:.2f} ms")
    
    if "largest_jump_batch_from" in analysis:
        print(f"\nLargest non-linear jump: batch {analysis['largest_jump_batch_from']} → {analysis['largest_jump_batch_to']}")
        print(f"Jump ratio: {analysis['largest_jump_ratio']:.2f}x (1.0 = perfect linear scaling)")
    
    # Plot metrics
    plot_profiling_metrics(norm_df, args.output)
    print(f"\nPlots saved to {args.output}")


if __name__ == "__main__":
    main() 