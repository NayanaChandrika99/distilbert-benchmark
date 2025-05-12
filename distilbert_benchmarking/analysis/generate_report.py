"""
Generate insights report from benchmark results.

This script analyzes benchmark data and creates a markdown report with
performance insights and recommendations.
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate insights report from benchmark results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark results (JSONL file)",
    )
    parser.add_argument(
        "--output", type=str, default="insights.md", help="Path to output markdown file"
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Optional path to another benchmark result file to compare with",
    )
    parser.add_argument(
        "--figures",
        type=str,
        help="Optional path to directory with figures to include in the report",
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
    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")

    return results


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert results list to a DataFrame.

    Args:
        results: List of benchmark results

    Returns:
        DataFrame with benchmark results
    """
    # Flatten metrics and extract key information
    data = []
    for result in results:
        row = {
            "batch_size": result["batch_size"],
            "device": result["device"],
            "model": result["model"],
            "timestamp": result["timestamp"],
            "sequence_length": result["metrics"].get("sequence_length", 0),
            "warmup_runs": result["metrics"].get("warmup_runs", 0),
            "iterations": result["metrics"].get("iterations", 0),
        }

        # Add all metrics
        for metric, value in result["metrics"].items():
            row[metric] = value

        data.append(row)

    return pd.DataFrame(data)


def find_optimal_batch_size(
    df: pd.DataFrame, metric: str = "throughput_mean", higher_is_better: bool = True
) -> int:
    """
    Find the optimal batch size based on a metric.

    Args:
        df: DataFrame with benchmark results
        metric: Metric to optimize
        higher_is_better: Whether higher metric values are better

    Returns:
        Optimal batch size
    """
    if metric not in df.columns:
        print(f"Warning: Metric {metric} not found in results")
        return df["batch_size"].iloc[0]

    if higher_is_better:
        return df.loc[df[metric].idxmax(), "batch_size"]
    else:
        return df.loc[df[metric].idxmin(), "batch_size"]


def find_bottlenecks(df: pd.DataFrame) -> List[str]:
    """
    Identify potential performance bottlenecks.

    Args:
        df: DataFrame with benchmark results

    Returns:
        List of potential bottlenecks
    """
    bottlenecks = []

    # Check if throughput plateaus for large batch sizes
    if "throughput_mean" in df.columns and len(df) > 2:
        # Sort by batch size
        df_sorted = df.sort_values(by="batch_size")

        # Check if throughput is not increasing significantly for the largest batch sizes
        throughput_last = df_sorted["throughput_mean"].iloc[-1]
        throughput_second_last = df_sorted["throughput_mean"].iloc[-2]

        if throughput_last <= throughput_second_last * 1.1:  # Less than 10% improvement
            bottlenecks.append(
                "Throughput plateaus for large batch sizes, possibly due to memory bandwidth limitations"
            )

    # Check if memory usage is high relative to available memory
    if "cpu_memory_mb_max" in df.columns:
        if df["cpu_memory_mb_max"].max() > 16000:  # More than 16GB
            bottlenecks.append(
                "High CPU memory usage may cause swapping and performance degradation"
            )

    if "gpu_memory_mb_max" in df.columns:
        gpu_memory_usage = df["gpu_memory_mb_max"].max()
        # Check if system_info contains GPU memory information
        if "system_info" in df.iloc[0]:
            system_info = df.iloc[0]["system_info"]
            if (
                "cuda_device_count" in system_info
                and system_info["cuda_device_count"] > 0
            ):
                if "gpu_info" in system_info and len(system_info["gpu_info"]) > 0:
                    gpu_info = system_info["gpu_info"][0]
                    if "total_memory_mb" in gpu_info:
                        total_memory = gpu_info["total_memory_mb"]
                        if (
                            gpu_memory_usage > 0.9 * total_memory
                        ):  # More than 90% of GPU memory
                            bottlenecks.append(
                                "GPU memory usage is close to capacity, which may limit batch size scaling"
                            )

    # Check for high latency variance
    if "latency_ms_mean" in df.columns and "latency_ms_std" in df.columns:
        # Calculate coefficient of variation
        cv = df["latency_ms_std"] / df["latency_ms_mean"]
        if cv.max() > 0.2:  # More than 20% variation
            bottlenecks.append(
                "High latency variance indicates potential system interference or thermal throttling"
            )

    return bottlenecks


def generate_insights(
    df: pd.DataFrame, model_info: Dict[str, Any], system_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate insights from benchmark results.

    Args:
        df: DataFrame with benchmark results
        model_info: Model information
        system_info: System information

    Returns:
        Dictionary with insights
    """
    insights = {}

    # Sort by batch size for analysis
    df = df.sort_values(by="batch_size")

    # Device type
    device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
    insights["device"] = device_type

    # Optimal batch size for throughput
    if "throughput_mean" in df.columns:
        optimal_batch_size = find_optimal_batch_size(df, "throughput_mean", True)
        insights["optimal_batch_size"] = optimal_batch_size
        insights["max_throughput"] = df.loc[
            df["batch_size"] == optimal_batch_size, "throughput_mean"
        ].iloc[0]

    # Optimal batch size for latency
    if "latency_ms_mean" in df.columns:
        min_latency_batch_size = find_optimal_batch_size(df, "latency_ms_mean", False)
        insights["min_latency_batch_size"] = min_latency_batch_size
        insights["min_latency"] = df.loc[
            df["batch_size"] == min_latency_batch_size, "latency_ms_mean"
        ].iloc[0]

    # Check for memory efficiency
    if "cpu_memory_mb_max" in df.columns:
        memory_per_sample = df["cpu_memory_mb_max"] / df["batch_size"]
        most_efficient_idx = memory_per_sample.idxmin()
        most_efficient_batch_size = df.loc[most_efficient_idx, "batch_size"]
        insights["most_memory_efficient_batch_size"] = most_efficient_batch_size
        insights["memory_efficiency"] = (
            df.loc[most_efficient_idx, "cpu_memory_mb_max"] / most_efficient_batch_size
        )

    # Check for scaling efficiency
    if "throughput_mean" in df.columns and len(df) > 1:
        # Calculate throughput vs batch size ratio
        df["throughput_per_sample"] = df["throughput_mean"] / df["batch_size"]

        # Find where scaling starts to decline
        throughput_per_sample_max = df["throughput_per_sample"].max()
        throughput_per_sample_last = df["throughput_per_sample"].iloc[-1]

        # If throughput per sample at largest batch size is less than 70% of maximum,
        # we have significant scaling inefficiency
        if throughput_per_sample_last < 0.7 * throughput_per_sample_max:
            insights["scaling_efficiency"] = (
                throughput_per_sample_last / throughput_per_sample_max
            )
            insights["scaling_bottleneck_batch_size"] = df.loc[
                df["throughput_per_sample"].idxmax(), "batch_size"
            ]

    # Find potential bottlenecks
    bottlenecks = find_bottlenecks(df)
    if bottlenecks:
        insights["bottlenecks"] = bottlenecks

    # Rough performance estimate for production
    if "throughput_mean" in df.columns and "optimal_batch_size" in insights:
        max_throughput = insights["max_throughput"]
        samples_per_day = max_throughput * 3600 * 24  # samples per day
        insights["samples_per_day"] = samples_per_day

    return insights


def generate_markdown_report(
    df: pd.DataFrame,
    insights: Dict[str, Any],
    model_info: Dict[str, Any],
    system_info: Dict[str, Any],
    figures_dir: str = None,
    output_path: str = "insights.md",
) -> None:
    """
    Generate a markdown report with insights.

    Args:
        df: DataFrame with benchmark results
        insights: Dictionary with insights
        model_info: Model information
        system_info: System information
        figures_dir: Directory with figures to include
        output_path: Path to output markdown file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w") as f:
        # Title and timestamp
        f.write("# DistilBERT Benchmark Insights Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- **Model**: {model_info.get('name', 'Unknown')}\n")
        f.write(f"- **Hidden Size**: {model_info.get('hidden_size', 'Unknown')}\n")
        f.write(
            f"- **Hidden Layers**: {model_info.get('num_hidden_layers', 'Unknown')}\n"
        )
        f.write(
            f"- **Attention Heads**: {model_info.get('num_attention_heads', 'Unknown')}\n"
        )
        if "num_parameters" in model_info:
            f.write(f"- **Parameters**: {model_info['num_parameters']:,}\n")
        f.write("\n")

        # System information
        f.write("## System Information\n\n")
        f.write(f"- **Platform**: {system_info.get('platform', 'Unknown')}\n")
        f.write(f"- **Processor**: {system_info.get('processor', 'Unknown')}\n")
        if "cpu_count" in system_info:
            f.write(f"- **CPU Count**: {system_info['cpu_count']}\n")
        f.write(
            f"- **Python Version**: {system_info.get('python_version', 'Unknown')}\n"
        )
        f.write(
            f"- **PyTorch Version**: {system_info.get('torch_version', 'Unknown')}\n"
        )

        if system_info.get("torch_cuda_available", False):
            f.write(
                f"- **CUDA Available**: {system_info.get('torch_cuda_available', False)}\n"
            )
            if "cuda_version" in system_info:
                f.write(f"- **CUDA Version**: {system_info['cuda_version']}\n")
            if "cuda_device_count" in system_info:
                f.write(f"- **GPU Count**: {system_info['cuda_device_count']}\n")

            if "gpu_info" in system_info and system_info["gpu_info"]:
                for i, gpu in enumerate(system_info["gpu_info"]):
                    f.write(f"- **GPU {i}**: {gpu.get('name', 'Unknown')}\n")
                    if "total_memory_mb" in gpu:
                        f.write(f"  - Memory: {gpu['total_memory_mb'] / 1024:.2f} GB\n")
        f.write("\n")

        # Benchmark summary
        f.write("## Benchmark Summary\n\n")
        f.write(f"- **Device**: {insights.get('device', 'Unknown').upper()}\n")
        f.write(
            f"- **Batch Sizes Tested**: {', '.join(map(str, sorted(df['batch_size'].unique())))}\n"
        )
        f.write(f"- **Sequence Length**: {df['sequence_length'].iloc[0]}\n")
        f.write(f"- **Iterations**: {df['iterations'].iloc[0]}\n")
        f.write(f"- **Warmup Runs**: {df['warmup_runs'].iloc[0]}\n")
        f.write("\n")

        # Key Performance Metrics
        f.write("## Key Performance Metrics\n\n")

        # Latency and throughput tables
        metrics_table = pd.DataFrame(
            {
                "Batch Size": df["batch_size"],
                "Latency (ms)": df["latency_ms_mean"],
                "Throughput (samples/sec)": df["throughput_mean"],
            }
        )

        # Add memory metrics if available
        if "cpu_memory_mb_max" in df.columns:
            metrics_table["CPU Memory (MB)"] = df["cpu_memory_mb_max"]

        if "gpu_memory_mb_max" in df.columns:
            metrics_table["GPU Memory (MB)"] = df["gpu_memory_mb_max"]

        # Sort by batch size and format the table
        metrics_table = metrics_table.sort_values(by="Batch Size")
        f.write(metrics_table.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n")

        # Include figures if available
        if figures_dir and os.path.exists(figures_dir):
            f.write("## Performance Visualizations\n\n")

            # Find and include figures
            figure_exts = [".png", ".jpg", ".svg", ".pdf"]
            figure_files = []
            for ext in figure_exts:
                figure_files.extend(list(Path(figures_dir).glob(f"*{ext}")))

            for figure_file in sorted(figure_files):
                # Get absolute path for the figure
                abs_figure_path = os.path.abspath(figure_file)
                figure_name = (
                    os.path.splitext(os.path.basename(figure_file))[0]
                    .replace("_", " ")
                    .title()
                )
                f.write(f"### {figure_name}\n\n")
                f.write(f"![{figure_name}]({abs_figure_path})\n\n")

        # Performance Insights
        f.write("## Performance Insights\n\n")

        # Optimal batch size
        if "optimal_batch_size" in insights and "max_throughput" in insights:
            f.write(
                f"- **Optimal Batch Size for Throughput**: {insights['optimal_batch_size']}\n"
            )
            f.write(f"  - Achieves {insights['max_throughput']:.2f} samples/second\n")

        # Latency-optimized batch size
        if "min_latency_batch_size" in insights and "min_latency" in insights:
            f.write(
                f"- **Optimal Batch Size for Latency**: {insights['min_latency_batch_size']}\n"
            )
            f.write(f"  - Achieves {insights['min_latency']:.2f} ms latency\n")

        # Memory efficiency
        if (
            "most_memory_efficient_batch_size" in insights
            and "memory_efficiency" in insights
        ):
            f.write(
                f"- **Most Memory-Efficient Batch Size**: {insights['most_memory_efficient_batch_size']}\n"
            )
            f.write(f"  - Uses {insights['memory_efficiency']:.2f} MB per sample\n")

        # Scaling efficiency
        if (
            "scaling_efficiency" in insights
            and "scaling_bottleneck_batch_size" in insights
        ):
            f.write(f"- **Scaling Efficiency**: {insights['scaling_efficiency']:.2f}\n")
            f.write(
                f"  - Scaling starts to decline after batch size {insights['scaling_bottleneck_batch_size']}\n"
            )

        # Production throughput estimate
        if "samples_per_day" in insights:
            samples_per_day = insights["samples_per_day"]
            f.write(
                f"- **Estimated Daily Throughput**: {samples_per_day:,.0f} samples/day\n"
            )
            f.write(
                f"  - Equivalent to {samples_per_day/1000000:.2f} million samples/day\n"
            )

        f.write("\n")

        # Bottlenecks
        if "bottlenecks" in insights and insights["bottlenecks"]:
            f.write("## Potential Bottlenecks\n\n")
            for bottleneck in insights["bottlenecks"]:
                f.write(f"- {bottleneck}\n")
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if "optimal_batch_size" in insights:
            f.write(
                f"- For maximum throughput, use batch size **{insights['optimal_batch_size']}**\n"
            )

        if "min_latency_batch_size" in insights:
            f.write(
                f"- For minimum latency, use batch size **{insights['min_latency_batch_size']}**\n"
            )

        if "bottlenecks" in insights and insights["bottlenecks"]:
            f.write("- Address the identified bottlenecks to improve performance:\n")
            for i, bottleneck in enumerate(insights["bottlenecks"]):
                recommendation = ""
                if "memory" in bottleneck.lower():
                    recommendation = "Consider using a device with more memory or optimizing memory usage"
                elif "throughput plateaus" in bottleneck.lower():
                    recommendation = "Consider using mixed precision or model optimization techniques"
                elif "latency variance" in bottleneck.lower():
                    recommendation = "Ensure consistent thermal conditions and minimize system background tasks"

                if recommendation:
                    f.write(f"  - {recommendation}\n")

        f.write("\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        device = insights.get("device", "the device").upper()

        if "max_throughput" in insights and "min_latency" in insights:
            max_throughput = insights["max_throughput"]
            min_latency = insights["min_latency"]

            f.write(
                f"The DistilBERT model demonstrates {max_throughput:.2f} samples/second maximum throughput "
            )
            f.write(f"and {min_latency:.2f} ms minimum latency on {device}. ")

            if "scaling_efficiency" in insights:
                scaling_efficiency = insights["scaling_efficiency"]
                if scaling_efficiency < 0.5:
                    f.write(
                        f"The model shows poor scaling efficiency ({scaling_efficiency:.2f}) with larger batch sizes, "
                    )
                    f.write("suggesting resource limitations. ")
                elif scaling_efficiency < 0.8:
                    f.write(
                        f"The model shows moderate scaling efficiency ({scaling_efficiency:.2f}) with larger batch sizes. "
                    )
                else:
                    f.write(
                        f"The model shows good scaling efficiency ({scaling_efficiency:.2f}) with larger batch sizes. "
                    )

            if "samples_per_day" in insights:
                samples_per_day = insights["samples_per_day"]
                f.write(
                    f"At optimal settings, this configuration can process approximately {samples_per_day/1000000:.2f} million samples per day."
                )

        f.write("\n")


def main():
    """Main function."""
    args = parse_args()

    # Load results
    results = load_results(args.input)
    if not results:
        print(f"Error: No valid results found in {args.input}")
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Extract model and system info from first result
    model_info = results[0].get("model_info", {})
    system_info = results[0].get("system_info", {})

    # Generate insights
    insights = generate_insights(df, model_info, system_info)

    # Generate markdown report
    generate_markdown_report(
        df, insights, model_info, system_info, args.figures, args.output
    )

    print(f"Report generated and saved to {args.output}")


if __name__ == "__main__":
    main()
