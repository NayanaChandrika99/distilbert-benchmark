#!/usr/bin/env python
"""
Batch anomaly detector for DistilBERT benchmarks.

This script analyzes benchmark results to identify anomalies 
in processing different batch sizes.
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


def extract_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract metrics from benchmark results into a DataFrame."""
    data = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "batch_size": result["batch_size"],
            "device": result["device"],
            "latency_ms_mean": metrics.get("latency_ms_mean", 0),
            "latency_ms_min": metrics.get("latency_ms_min", 0),
            "latency_ms_max": metrics.get("latency_ms_max", 0),
            "latency_ms_std": metrics.get("latency_ms_std", 0),
            "throughput_mean": metrics.get("throughput_mean", 0),
            "data_prep_time_s": metrics.get("data_prep_time_s", 0),
            "inference_time_s": metrics.get("inference_time_s", 0),
            "cpu_memory_mb_max": metrics.get("cpu_memory_mb_max", 0),
            "total_time_s": metrics.get("data_prep_time_s", 0) + metrics.get("inference_time_s", 0),
            "per_item_latency_ms": metrics.get("latency_ms_mean", 0) / result["batch_size"],
            "efficiency": metrics.get("throughput_mean", 0) / result["batch_size"],
        }
        data.append(row)
    return pd.DataFrame(data)


def analyze_batch_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze batch size anomalies in benchmark results."""
    # Add normalized metrics (relative to batch size 1)
    if 1 in df["batch_size"].values:
        base_latency = df[df["batch_size"] == 1]["latency_ms_mean"].values[0]
        base_throughput = df[df["batch_size"] == 1]["throughput_mean"].values[0]
        
        df["latency_normalized"] = df["latency_ms_mean"] / base_latency
        df["throughput_normalized"] = df["throughput_mean"] / base_throughput
        df["expected_latency"] = base_latency * df["batch_size"]
        df["expected_throughput"] = base_throughput * df["batch_size"]
        
        # Calculate efficiency metrics
        df["latency_efficiency"] = df["expected_latency"] / df["latency_ms_mean"]
        df["throughput_efficiency"] = df["throughput_mean"] / df["expected_throughput"]
    
    # Find the most anomalous batch size
    if len(df) > 1:
        # For latency, lower is better, so we're looking for unexpected increases
        df["latency_anomaly_score"] = np.abs(
            df["latency_ms_mean"] / df["batch_size"] - df[df["batch_size"] == 1]["latency_ms_mean"].values[0]
        )
        
        # For throughput, higher is better, so we're looking for unexpected decreases
        df["throughput_anomaly_score"] = np.abs(
            1 - (df["throughput_mean"] / df["batch_size"]) / (df[df["batch_size"] == 1]["throughput_mean"].values[0])
        )
        
        # Combined anomaly score
        df["anomaly_score"] = df["latency_anomaly_score"] + df["throughput_anomaly_score"]
        
        most_anomalous = df.loc[df["anomaly_score"].idxmax()]
        
        return {
            "most_anomalous_batch": int(most_anomalous["batch_size"]),
            "anomaly_score": most_anomalous["anomaly_score"],
            "latency_vs_expected": most_anomalous["latency_ms_mean"] / most_anomalous["expected_latency"],
            "throughput_vs_expected": most_anomalous["throughput_mean"] / most_anomalous["expected_throughput"],
            "data_prep_time": most_anomalous["data_prep_time_s"],
            "inference_time": most_anomalous["inference_time_s"],
        }
    
    return {"most_anomalous_batch": None}


def plot_batch_metrics(df: pd.DataFrame, output_dir: str = "results/analysis"):
    """Plot batch size metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Latency vs Batch Size
    plt.figure(figsize=(10, 6))
    plt.plot(df["batch_size"], df["latency_ms_mean"], "o-", label="Actual Latency")
    if "expected_latency" in df.columns:
        plt.plot(df["batch_size"], df["expected_latency"], "--", label="Expected Latency (Linear)")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/latency_vs_batch.png")
    
    # Plot 2: Throughput vs Batch Size
    plt.figure(figsize=(10, 6))
    plt.plot(df["batch_size"], df["throughput_mean"], "o-", label="Actual Throughput")
    if "expected_throughput" in df.columns:
        plt.plot(df["batch_size"], df["expected_throughput"], "--", label="Expected Throughput (Linear)")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/throughput_vs_batch.png")
    
    # Plot 3: Per-item Latency
    plt.figure(figsize=(10, 6))
    plt.plot(df["batch_size"], df["per_item_latency_ms"], "o-")
    plt.xlabel("Batch Size")
    plt.ylabel("Per-item Latency (ms)")
    plt.title("Per-item Latency vs Batch Size")
    plt.grid(True)
    plt.savefig(f"{output_dir}/per_item_latency.png")
    
    # Plot 4: Timing Breakdown
    plt.figure(figsize=(10, 6))
    width = 0.35
    batch_sizes = df["batch_size"].values
    x = np.arange(len(batch_sizes))
    plt.bar(x, df["data_prep_time_s"], width, label="Data Preparation")
    plt.bar(x, df["inference_time_s"], width, bottom=df["data_prep_time_s"], label="Inference")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")
    plt.title("Time Breakdown by Batch Size")
    plt.xticks(x, batch_sizes)
    plt.legend()
    plt.savefig(f"{output_dir}/time_breakdown.png")
    
    # Plot 5: Efficiency metrics if available
    if "latency_efficiency" in df.columns and "throughput_efficiency" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["batch_size"], df["latency_efficiency"], "o-", label="Latency Efficiency")
        plt.plot(df["batch_size"], df["throughput_efficiency"], "o-", label="Throughput Efficiency")
        plt.axhline(y=1.0, color="r", linestyle="--")
        plt.xlabel("Batch Size")
        plt.ylabel("Efficiency (1.0 = ideal)")
        plt.title("Processing Efficiency by Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/batch_efficiency.png")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze batch size anomalies in benchmark results")
    parser.add_argument(
        "--input", type=str, default="results/analysis/batch_analysis.jsonl",
        help="Path to benchmark results JSONL file"
    )
    parser.add_argument(
        "--output", type=str, default="results/analysis",
        help="Directory to save analysis outputs"
    )
    args = parser.parse_args()
    
    # Load and process results
    results = load_jsonl(args.input)
    df = extract_metrics(results)
    
    # Sort by batch size
    df = df.sort_values("batch_size")
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    print(f"{'Batch Size':<10} {'Latency (ms)':<15} {'Throughput (samples/s)':<25} {'Data Prep (s)':<15} {'Inference (s)':<15}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{int(row['batch_size']):<10} {row['latency_ms_mean']:<15.2f} {row['throughput_mean']:<25.2f} {row['data_prep_time_s']:<15.4f} {row['inference_time_s']:<15.4f}")
    
    # Analyze anomalies
    anomalies = analyze_batch_anomalies(df)
    
    print("\nAnomaly Analysis:")
    print("=" * 80)
    if anomalies["most_anomalous_batch"]:
        print(f"Most anomalous batch size: {anomalies['most_anomalous_batch']}")
        print(f"Anomaly score: {anomalies['anomaly_score']:.4f}")
        print(f"Actual vs expected latency: {anomalies['latency_vs_expected']:.2f}x")
        print(f"Actual vs expected throughput: {anomalies['throughput_vs_expected']:.2f}x")
        print(f"Data preparation time: {anomalies['data_prep_time']:.4f}s")
        print(f"Inference time: {anomalies['inference_time']:.4f}s")
    else:
        print("No anomalies detected (insufficient data)")
    
    # Plot metrics
    plot_batch_metrics(df, args.output)
    print(f"\nPlots saved to {args.output}")


if __name__ == "__main__":
    main() 