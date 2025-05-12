"""
Generate visualizations from benchmark results.

This script reads benchmark result data and generates various plots for analysis.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from benchmark results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark results (JSONL file)",
    )
    parser.add_argument(
        "--output", type=str, default="figures", help="Directory to save output figures"
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Optional path to another benchmark result file to compare with",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "jpg"],
        help="Output figure format",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
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


def setup_plot_style():
    """Set up matplotlib plot style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 18


def plot_latency_vs_batch_size(
    df: pd.DataFrame,
    output_dir: str,
    compare_df: Optional[pd.DataFrame] = None,
    format: str = "png",
    dpi: int = 300,
):
    """
    Plot latency vs batch size.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save output figures
        compare_df: Optional DataFrame with comparison results
        format: Output figure format
        dpi: Figure DPI
    """
    plt.figure()

    # Plot main data
    device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
    plt.plot(
        df["batch_size"],
        df["latency_ms_mean"],
        "o-",
        label=f"{device_type.upper()} Mean",
        linewidth=2,
        markersize=8,
    )

    # Add error bars
    if "latency_ms_std" in df.columns:
        plt.fill_between(
            df["batch_size"],
            df["latency_ms_mean"] - df["latency_ms_std"],
            df["latency_ms_mean"] + df["latency_ms_std"],
            alpha=0.2,
        )

    # Plot p90 latency
    if "latency_ms_p90" in df.columns:
        plt.plot(
            df["batch_size"],
            df["latency_ms_p90"],
            "s--",
            label=f"{device_type.upper()} p90",
            linewidth=1.5,
            markersize=6,
        )

    # Plot comparison data if provided
    if compare_df is not None:
        compare_device = (
            compare_df["device"].iloc[0]
            if compare_df["device"].nunique() == 1
            else "mixed"
        )
        plt.plot(
            compare_df["batch_size"],
            compare_df["latency_ms_mean"],
            "o-",
            label=f"{compare_device.upper()} Mean",
            linewidth=2,
            markersize=8,
        )

        # Add error bars for comparison
        if "latency_ms_std" in compare_df.columns:
            plt.fill_between(
                compare_df["batch_size"],
                compare_df["latency_ms_mean"] - compare_df["latency_ms_std"],
                compare_df["latency_ms_mean"] + compare_df["latency_ms_std"],
                alpha=0.2,
            )

        # Plot p90 latency for comparison
        if "latency_ms_p90" in compare_df.columns:
            plt.plot(
                compare_df["batch_size"],
                compare_df["latency_ms_p90"],
                "s--",
                label=f"{compare_device.upper()} p90",
                linewidth=1.5,
                markersize=6,
            )

    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Batch Size")
    plt.grid(True)
    plt.legend()

    # Set x-axis to show all batch sizes
    plt.xticks(df["batch_size"])

    # Make sure y-axis starts from 0
    plt.ylim(bottom=0)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"latency_vs_batch_size.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved latency plot to {output_path}")


def plot_throughput_vs_batch_size(
    df: pd.DataFrame,
    output_dir: str,
    compare_df: Optional[pd.DataFrame] = None,
    format: str = "png",
    dpi: int = 300,
):
    """
    Plot throughput vs batch size.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save output figures
        compare_df: Optional DataFrame with comparison results
        format: Output figure format
        dpi: Figure DPI
    """
    plt.figure()

    # Plot main data
    device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
    plt.plot(
        df["batch_size"],
        df["throughput_mean"],
        "o-",
        label=f"{device_type.upper()} Mean",
        linewidth=2,
        markersize=8,
    )

    # Plot comparison data if provided
    if compare_df is not None:
        compare_device = (
            compare_df["device"].iloc[0]
            if compare_df["device"].nunique() == 1
            else "mixed"
        )
        plt.plot(
            compare_df["batch_size"],
            compare_df["throughput_mean"],
            "o-",
            label=f"{compare_device.upper()} Mean",
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Throughput vs Batch Size")
    plt.grid(True)
    plt.legend()

    # Set x-axis to show all batch sizes
    plt.xticks(df["batch_size"])

    # Make sure y-axis starts from 0
    plt.ylim(bottom=0)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"throughput_vs_batch_size.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved throughput plot to {output_path}")


def plot_memory_usage_vs_batch_size(
    df: pd.DataFrame,
    output_dir: str,
    compare_df: Optional[pd.DataFrame] = None,
    format: str = "png",
    dpi: int = 300,
):
    """
    Plot memory usage vs batch size.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save output figures
        compare_df: Optional DataFrame with comparison results
        format: Output figure format
        dpi: Figure DPI
    """
    plt.figure()

    # Check if we have CPU and/or GPU memory
    has_cpu_memory = "cpu_memory_mb_max" in df.columns
    has_gpu_memory = "gpu_memory_mb_max" in df.columns

    # Plot CPU memory if available
    if has_cpu_memory:
        device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
        plt.plot(
            df["batch_size"],
            df["cpu_memory_mb_max"],
            "o-",
            label=f"{device_type.upper()} CPU Memory",
            linewidth=2,
            markersize=8,
        )

    # Plot GPU memory if available
    if has_gpu_memory:
        device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
        plt.plot(
            df["batch_size"],
            df["gpu_memory_mb_max"],
            "s-",
            label=f"{device_type.upper()} GPU Memory",
            linewidth=2,
            markersize=8,
        )

    # Plot comparison data if provided
    if compare_df is not None:
        compare_device = (
            compare_df["device"].iloc[0]
            if compare_df["device"].nunique() == 1
            else "mixed"
        )

        # Plot comparison CPU memory if available
        if "cpu_memory_mb_max" in compare_df.columns:
            plt.plot(
                compare_df["batch_size"],
                compare_df["cpu_memory_mb_max"],
                "o--",
                label=f"{compare_device.upper()} CPU Memory",
                linewidth=1.5,
                markersize=6,
            )

        # Plot comparison GPU memory if available
        if "gpu_memory_mb_max" in compare_df.columns:
            plt.plot(
                compare_df["batch_size"],
                compare_df["gpu_memory_mb_max"],
                "s--",
                label=f"{compare_device.upper()} GPU Memory",
                linewidth=1.5,
                markersize=6,
            )

    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs Batch Size")
    plt.grid(True)
    plt.legend()

    # Set x-axis to show all batch sizes
    plt.xticks(df["batch_size"])

    # Make sure y-axis starts from 0
    plt.ylim(bottom=0)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"memory_vs_batch_size.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved memory usage plot to {output_path}")


def plot_energy_consumption(
    df: pd.DataFrame,
    output_dir: str,
    compare_df: Optional[pd.DataFrame] = None,
    format: str = "png",
    dpi: int = 300,
):
    """
    Plot energy consumption vs batch size.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save output figures
        compare_df: Optional DataFrame with comparison results
        format: Output figure format
        dpi: Figure DPI
    """
    # Check if we have energy data
    has_cpu_energy = "cpu_energy_j" in df.columns
    has_gpu_energy = "gpu_energy_j" in df.columns or "gpu_avg_power_w" in df.columns

    if not has_cpu_energy and not has_gpu_energy:
        print("No energy consumption data available, skipping energy plot.")
        return

    plt.figure()

    # Plot CPU energy if available
    if has_cpu_energy:
        device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
        plt.plot(
            df["batch_size"],
            df["cpu_energy_j"],
            "o-",
            label=f"{device_type.upper()} CPU Energy",
            linewidth=2,
            markersize=8,
        )

    # For GPU, use energy if available, otherwise use power (which is energy/time)
    if "gpu_energy_j" in df.columns:
        device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
        plt.plot(
            df["batch_size"],
            df["gpu_energy_j"],
            "s-",
            label=f"{device_type.upper()} GPU Energy",
            linewidth=2,
            markersize=8,
        )
    elif "gpu_avg_power_w" in df.columns:
        device_type = df["device"].iloc[0] if df["device"].nunique() == 1 else "mixed"
        # Convert power (watts) to energy (joules) by multiplying by time
        # Assuming the benchmark time is related to latency * iterations
        if "latency_ms_mean" in df.columns and "iterations" in df.columns:
            energy = df["gpu_avg_power_w"] * (
                df["latency_ms_mean"] * df["iterations"] / 1000
            )
            plt.plot(
                df["batch_size"],
                energy,
                "s-",
                label=f"{device_type.upper()} GPU Energy (estimated)",
                linewidth=2,
                markersize=8,
            )
        else:
            # Just plot power if we can't estimate energy
            plt.plot(
                df["batch_size"],
                df["gpu_avg_power_w"],
                "s-",
                label=f"{device_type.upper()} GPU Power",
                linewidth=2,
                markersize=8,
            )
            plt.ylabel("Power (W)")
            plt.title("Power Consumption vs Batch Size")

    # Plot comparison data if provided with similar logic
    if compare_df is not None:
        compare_device = (
            compare_df["device"].iloc[0]
            if compare_df["device"].nunique() == 1
            else "mixed"
        )

        if "cpu_energy_j" in compare_df.columns:
            plt.plot(
                compare_df["batch_size"],
                compare_df["cpu_energy_j"],
                "o--",
                label=f"{compare_device.upper()} CPU Energy",
                linewidth=1.5,
                markersize=6,
            )

        if "gpu_energy_j" in compare_df.columns:
            plt.plot(
                compare_df["batch_size"],
                compare_df["gpu_energy_j"],
                "s--",
                label=f"{compare_device.upper()} GPU Energy",
                linewidth=1.5,
                markersize=6,
            )
        elif "gpu_avg_power_w" in compare_df.columns:
            if (
                "latency_ms_mean" in compare_df.columns
                and "iterations" in compare_df.columns
            ):
                energy = compare_df["gpu_avg_power_w"] * (
                    compare_df["latency_ms_mean"] * compare_df["iterations"] / 1000
                )
                plt.plot(
                    compare_df["batch_size"],
                    energy,
                    "s--",
                    label=f"{compare_device.upper()} GPU Energy (estimated)",
                    linewidth=1.5,
                    markersize=6,
                )
            else:
                plt.plot(
                    compare_df["batch_size"],
                    compare_df["gpu_avg_power_w"],
                    "s--",
                    label=f"{compare_device.upper()} GPU Power",
                    linewidth=1.5,
                    markersize=6,
                )
                plt.ylabel("Power (W)")
                plt.title("Power Consumption vs Batch Size")

    plt.xlabel("Batch Size")
    if "gpu_avg_power_w" in df.columns and "gpu_energy_j" not in df.columns:
        plt.ylabel("Power (W)")
        plt.title("Power Consumption vs Batch Size")
    else:
        plt.ylabel("Energy (J)")
        plt.title("Energy Consumption vs Batch Size")

    plt.grid(True)
    plt.legend()

    # Set x-axis to show all batch sizes
    plt.xticks(df["batch_size"])

    # Make sure y-axis starts from 0
    plt.ylim(bottom=0)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"energy_vs_batch_size.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved energy consumption plot to {output_path}")


def main():
    """Main function."""
    args = parse_args()

    # Set up matplotlib style
    setup_plot_style()

    # Load results
    results = load_results(args.input)
    df = results_to_dataframe(results)

    # Sort by batch size
    df = df.sort_values(by="batch_size")

    # Load comparison results if provided
    compare_df = None
    if args.compare:
        compare_results = load_results(args.compare)
        compare_df = results_to_dataframe(compare_results)
        compare_df = compare_df.sort_values(by="batch_size")

    # Generate plots
    plot_latency_vs_batch_size(df, args.output, compare_df, args.format, args.dpi)
    plot_throughput_vs_batch_size(df, args.output, compare_df, args.format, args.dpi)
    plot_memory_usage_vs_batch_size(df, args.output, compare_df, args.format, args.dpi)
    plot_energy_consumption(df, args.output, compare_df, args.format, args.dpi)

    print(f"All plots saved to {args.output}")


if __name__ == "__main__":
    main()
