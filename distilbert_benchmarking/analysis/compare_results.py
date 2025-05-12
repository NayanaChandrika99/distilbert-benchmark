"""
Compare CPU and GPU benchmark results.

This script compares benchmark results from CPU and GPU runs and generates
a markdown report with the comparison.
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Any


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare CPU and GPU benchmark results"
    )
    parser.add_argument(
        "--cpu",
        type=str,
        required=True,
        help="Path to CPU benchmark results (JSONL file)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="Path to GPU benchmark results (JSONL file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison.md",
        help="Path to output markdown file",
    )
    return parser.parse_args()


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of benchmark results
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")

    return results


def create_comparison_dataframe(
    cpu_results: List[Dict[str, Any]], gpu_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a DataFrame comparing CPU and GPU results.

    Args:
        cpu_results: List of CPU benchmark results
        gpu_results: List of GPU benchmark results

    Returns:
        DataFrame with comparison results
    """
    # Extract batch sizes and metrics from both results
    comparison_data = []

    # Get common batch sizes
    cpu_batch_sizes = [result["batch_size"] for result in cpu_results]
    gpu_batch_sizes = [result["batch_size"] for result in gpu_results]
    common_batch_sizes = sorted(set(cpu_batch_sizes).intersection(set(gpu_batch_sizes)))

    # Get key metrics to compare
    metrics_to_compare = ["latency_ms_mean", "throughput_mean", "cpu_memory_mb_max"]

    # Add GPU-specific metrics if available
    if any("gpu_memory_mb_max" in result["metrics"] for result in gpu_results):
        metrics_to_compare.append("gpu_memory_mb_max")
    if any("gpu_avg_power_w" in result["metrics"] for result in gpu_results):
        metrics_to_compare.append("gpu_avg_power_w")

    # Create data for each batch size
    for batch_size in common_batch_sizes:
        cpu_result = next(
            (r for r in cpu_results if r["batch_size"] == batch_size), None
        )
        gpu_result = next(
            (r for r in gpu_results if r["batch_size"] == batch_size), None
        )

        if cpu_result and gpu_result:
            row = {"batch_size": batch_size}

            for metric in metrics_to_compare:
                # CPU metric
                if metric in cpu_result["metrics"]:
                    row[f"cpu_{metric}"] = cpu_result["metrics"][metric]

                # GPU metric
                if metric in gpu_result["metrics"]:
                    row[f"gpu_{metric}"] = gpu_result["metrics"][metric]

                    # Calculate speedup for comparable metrics
                    if (
                        metric in ["latency_ms_mean", "throughput_mean"]
                        and metric in cpu_result["metrics"]
                    ):
                        if metric == "latency_ms_mean":
                            # Lower is better for latency
                            speedup = (
                                cpu_result["metrics"][metric]
                                / gpu_result["metrics"][metric]
                            )
                        else:
                            # Higher is better for throughput
                            speedup = (
                                gpu_result["metrics"][metric]
                                / cpu_result["metrics"][metric]
                            )
                        row[f"speedup_{metric}"] = speedup

            comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    return df


def generate_markdown_report(
    df: pd.DataFrame,
    cpu_results: List[Dict[str, Any]],
    gpu_results: List[Dict[str, Any]],
    output_path: str,
):
    """
    Generate a markdown report comparing CPU and GPU results.

    Args:
        df: DataFrame with comparison results
        cpu_results: List of CPU benchmark results
        gpu_results: List of GPU benchmark results
        output_path: Path to output markdown file
    """
    # Extract system information
    cpu_info = cpu_results[0]["system_info"] if cpu_results else {}
    gpu_info = gpu_results[0]["system_info"] if gpu_results else {}

    # Extract model information
    model_info = cpu_results[0]["model_info"] if cpu_results else {}
    if not model_info and gpu_results:
        model_info = gpu_results[0]["model_info"]

    # Create the report
    with open(output_path, "w") as f:
        # Title
        f.write("# DistilBERT Benchmark: CPU vs. GPU Comparison\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(
            "This report compares the performance of DistilBERT on CPU and GPU devices.\n\n"
        )

        # System information
        f.write("## System Information\n\n")
        f.write("### CPU\n\n")
        if cpu_info:
            f.write(f"- Platform: {cpu_info.get('platform', 'Unknown')}\n")
            f.write(f"- Processor: {cpu_info.get('processor', 'Unknown')}\n")
            f.write(f"- CPU Count: {cpu_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"- Python: {cpu_info.get('python_version', 'Unknown')}\n")
            f.write(f"- PyTorch: {cpu_info.get('torch_version', 'Unknown')}\n\n")
        else:
            f.write("No CPU information available.\n\n")

        f.write("### GPU\n\n")
        if gpu_info and gpu_info.get("torch_cuda_available", False):
            f.write(
                f"- CUDA Available: {gpu_info.get('torch_cuda_available', False)}\n"
            )
            f.write(f"- CUDA Version: {gpu_info.get('cuda_version', 'Unknown')}\n")
            f.write(f"- GPU Count: {gpu_info.get('cuda_device_count', 'Unknown')}\n")
            if "gpu_info" in gpu_info and gpu_info["gpu_info"]:
                for i, gpu in enumerate(gpu_info["gpu_info"]):
                    f.write(f"- GPU {i}: {gpu.get('name', 'Unknown')}\n")
                    if "total_memory_mb" in gpu:
                        f.write(f"  - Memory: {gpu['total_memory_mb'] / 1024:.2f} GB\n")
            f.write("\n")
        else:
            f.write("No GPU information available or CUDA not available.\n\n")

        # Model information
        f.write("## Model Information\n\n")
        if model_info:
            f.write(f"- Model: {model_info.get('name', 'Unknown')}\n")
            f.write(f"- Hidden Size: {model_info.get('hidden_size', 'Unknown')}\n")
            f.write(
                f"- Hidden Layers: {model_info.get('num_hidden_layers', 'Unknown')}\n"
            )
            f.write(
                f"- Attention Heads: {model_info.get('num_attention_heads', 'Unknown')}\n"
            )
            if "num_parameters" in model_info:
                f.write(f"- Parameters: {model_info['num_parameters']:,}\n\n")
            else:
                f.write("\n")
        else:
            f.write("No model information available.\n\n")

        # Comparison table
        f.write("## Performance Comparison\n\n")
        if not df.empty:
            # Format the DataFrame for markdown
            markdown_table = df.to_markdown(index=False, floatfmt=".2f")
            f.write(markdown_table + "\n\n")

            # Add interpretation
            f.write("### Interpretation\n\n")

            # Latency comparison
            if "speedup_latency_ms_mean" in df.columns:
                avg_latency_speedup = df["speedup_latency_ms_mean"].mean()
                max_latency_speedup = df["speedup_latency_ms_mean"].max()
                max_batch_size = df.loc[
                    df["speedup_latency_ms_mean"].idxmax(), "batch_size"
                ]

                f.write(
                    f"- **Latency**: GPU is on average {avg_latency_speedup:.2f}x faster than CPU.\n"
                )
                f.write(
                    f"  - The maximum speedup of {max_latency_speedup:.2f}x is achieved at batch size {max_batch_size}.\n\n"
                )

            # Throughput comparison
            if "speedup_throughput_mean" in df.columns:
                avg_throughput_speedup = df["speedup_throughput_mean"].mean()
                max_throughput_speedup = df["speedup_throughput_mean"].max()
                max_batch_size = df.loc[
                    df["speedup_throughput_mean"].idxmax(), "batch_size"
                ]

                f.write(
                    f"- **Throughput**: GPU achieves on average {avg_throughput_speedup:.2f}x higher throughput than CPU.\n"
                )
                f.write(
                    f"  - The maximum throughput improvement of {max_throughput_speedup:.2f}x is achieved at batch size {max_batch_size}.\n\n"
                )

            # Memory usage comparison
            if (
                "gpu_gpu_memory_mb_max" in df.columns
                and "cpu_cpu_memory_mb_max" in df.columns
            ):
                f.write("- **Memory Usage**:\n")
                for _, row in df.iterrows():
                    batch_size = row["batch_size"]
                    cpu_mem = row.get("cpu_cpu_memory_mb_max", 0)
                    gpu_mem = row.get("gpu_gpu_memory_mb_max", 0)

                    if cpu_mem > 0 and gpu_mem > 0:
                        f.write(
                            f"  - Batch size {batch_size}: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB\n"
                        )
                f.write("\n")

            # Power consumption if available
            if "gpu_gpu_avg_power_w" in df.columns:
                f.write("- **Power Consumption**:\n")
                for _, row in df.iterrows():
                    batch_size = row["batch_size"]
                    gpu_power = row.get("gpu_gpu_avg_power_w", 0)

                    if gpu_power > 0:
                        f.write(f"  - Batch size {batch_size}: GPU {gpu_power:.2f} W\n")
                f.write("\n")

            # Optimal batch size recommendation
            if "speedup_throughput_mean" in df.columns:
                optimal_batch_size = df.loc[
                    df["speedup_throughput_mean"].idxmax(), "batch_size"
                ]
                f.write(
                    f"- **Recommendation**: For optimal performance, use batch size {optimal_batch_size} on GPU.\n\n"
                )
        else:
            f.write(
                "No comparison data available. Make sure both CPU and GPU benchmark results have common batch sizes.\n\n"
            )

        # Conclusion
        f.write("## Conclusion\n\n")
        if not df.empty and "speedup_throughput_mean" in df.columns:
            avg_speedup = df["speedup_throughput_mean"].mean()
            if avg_speedup > 1.5:
                f.write(
                    f"The GPU provides significant performance benefits with an average of {avg_speedup:.2f}x higher throughput "
                )
                f.write(
                    "compared to CPU execution. For production deployments, GPU is recommended for maximum performance.\n\n"
                )
            elif avg_speedup > 1:
                f.write(
                    f"The GPU provides moderate performance benefits with an average of {avg_speedup:.2f}x higher throughput "
                )
                f.write(
                    "compared to CPU execution. Consider GPU for latency-sensitive applications.\n\n"
                )
            else:
                f.write(
                    f"The GPU provides minimal performance benefits with an average of {avg_speedup:.2f}x higher throughput "
                )
                f.write(
                    "compared to CPU execution. CPU may be more cost-effective for this model and batch sizes.\n\n"
                )
        else:
            f.write(
                "Insufficient data to draw conclusions about CPU vs GPU performance.\n\n"
            )

    print(f"Comparison report saved to {output_path}")


def main():
    """Main function."""
    args = parse_args()

    # Load results
    cpu_results = load_results(args.cpu)
    gpu_results = load_results(args.gpu)

    print(f"Loaded {len(cpu_results)} CPU results and {len(gpu_results)} GPU results")

    # Create comparison DataFrame
    df = create_comparison_dataframe(cpu_results, gpu_results)

    # Generate markdown report
    generate_markdown_report(df, cpu_results, gpu_results, args.output)


if __name__ == "__main__":
    main()
