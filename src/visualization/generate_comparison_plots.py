#!/usr/bin/env python3
"""
Generate comparative benchmark plots for DistilBERT performance data.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

# Set plot style
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Default directory paths
DEFAULT_INPUT_DIR = "distilbert_outputs"
DEFAULT_OUTPUT_DIR = "distilbert_outputs/comparison_plots"

def load_jsonl(filepath):
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def plot_batch_comparisons(input_dir, output_dir):
    """Generate plots comparing performance across batch sizes."""
    # Load batch comparison data
    batch_data_path = os.path.join(input_dir, "batch_comparison.jsonl")
    if not os.path.exists(batch_data_path):
        print(f"Error: {batch_data_path} not found")
        return
    
    batch_data = load_jsonl(batch_data_path)
    
    # Extract key metrics
    df = pd.DataFrame([
        {
            'batch_size': data['batch_size'],
            'latency_ms': data['metrics']['latency_ms_mean'],
            'throughput': data['metrics']['throughput_mean'],
            'gpu_memory_mb': data['metrics']['gpu_memory_mb_max'],
            'cpu_memory_mb': data['metrics']['cpu_memory_mb_max'],
            'gpu_power_w': data['metrics'].get('gpu_avg_power_w', np.nan)
        }
        for data in batch_data
    ]).sort_values('batch_size')
    
    # Calculate memory efficiency
    df['memory_per_sample'] = df['gpu_memory_mb'] / df['batch_size']
    df['samples_per_watt'] = df['throughput'] / df['gpu_power_w']
    
    # Create batch_size output directory if not exists
    batch_size_dir = os.path.join(output_dir, "batch_size")
    os.makedirs(batch_size_dir, exist_ok=True)
    
    # 1. Throughput vs Batch Size
    plt.figure(figsize=(12, 8))
    plt.plot(df['batch_size'], df['throughput'], 'o-', linewidth=3, markersize=10)
    plt.title('DistilBERT Throughput vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/second)')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['batch_size'])
    
    # Add value annotations above each point
    for i, row in df.iterrows():
        plt.annotate(f"{row['throughput']:.1f}", 
                    (row['batch_size'], row['throughput']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(batch_size_dir, "throughput_vs_batchsize.png"), dpi=300)
    plt.close()
    
    # 2. Latency vs Batch Size
    plt.figure(figsize=(12, 8))
    plt.plot(df['batch_size'], df['latency_ms'], 'o-', linewidth=3, markersize=10, color='#ff7f0e')
    plt.title('DistilBERT Latency vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['batch_size'])
    
    # Add value annotations above each point
    for i, row in df.iterrows():
        plt.annotate(f"{row['latency_ms']:.1f} ms", 
                    (row['batch_size'], row['latency_ms']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(batch_size_dir, "latency_vs_batchsize.png"), dpi=300)
    plt.close()
    
    # 3. GPU Memory vs Batch Size
    plt.figure(figsize=(12, 8))
    plt.plot(df['batch_size'], df['gpu_memory_mb'], 'o-', linewidth=3, markersize=10, color='#2ca02c')
    plt.title('DistilBERT GPU Memory Usage vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('GPU Memory (MB)')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['batch_size'])
    
    # Add value annotations above each point
    for i, row in df.iterrows():
        plt.annotate(f"{row['gpu_memory_mb']:.1f} MB", 
                    (row['batch_size'], row['gpu_memory_mb']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(batch_size_dir, "memory_vs_batchsize.png"), dpi=300)
    plt.close()
    
    # 4. Memory per Sample vs Batch Size
    plt.figure(figsize=(12, 8))
    plt.plot(df['batch_size'], df['memory_per_sample'], 'o-', linewidth=3, markersize=10, color='#d62728')
    plt.title('DistilBERT Memory Efficiency (MB per Sample)')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory per Sample (MB)')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['batch_size'])
    
    # Add value annotations above each point
    for i, row in df.iterrows():
        plt.annotate(f"{row['memory_per_sample']:.1f} MB", 
                    (row['batch_size'], row['memory_per_sample']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(batch_size_dir, "efficiency_vs_batchsize.png"), dpi=300)
    plt.close()
    
    # 5. Combined metrics (multi-axis)
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    color1 = '#1f77b4'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (samples/second)', color=color1)
    line1 = ax1.plot(df['batch_size'], df['throughput'], 'o-', linewidth=3, markersize=10, color=color1, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(df['batch_size'])
    
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Latency (ms)', color=color2)
    line2 = ax2.plot(df['batch_size'], df['latency_ms'], 's--', linewidth=3, markersize=10, color=color2, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = '#2ca02c'
    ax3.set_ylabel('GPU Memory (MB)', color=color3)
    line3 = ax3.plot(df['batch_size'], df['gpu_memory_mb'], '^-.', linewidth=3, markersize=10, color=color3, label='GPU Memory')
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Add legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    plt.title('DistilBERT Performance Metrics vs Batch Size')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(batch_size_dir, "combined_metrics.png"), dpi=300)
    plt.close()
    
    # 6. Throughput/Latency Efficiency
    plt.figure(figsize=(12, 8))
    plt.plot(df['batch_size'], df['throughput']/df['batch_size'], 'o-', linewidth=3, markersize=10, color='#9467bd')
    plt.title('DistilBERT Throughput Scaling Efficiency')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput per Batch Item')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['batch_size'])
    
    # Add value annotations above each point
    for i, row in df.iterrows():
        efficiency = row['throughput']/row['batch_size']
        plt.annotate(f"{efficiency:.2f}", 
                    (row['batch_size'], efficiency),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(batch_size_dir, "scaling_efficiency.png"), dpi=300)
    plt.close()
    
    print("Batch size comparison plots generated successfully.")

def plot_mixed_precision_comparison(input_dir, output_dir):
    """Generate plots comparing FP32 vs mixed precision performance."""
    # Load mixed precision data
    mixed_precision_path = os.path.join(input_dir, "mixed_precision_results.jsonl")
    batch_data_path = os.path.join(input_dir, "batch_comparison.jsonl")
    
    if not (os.path.exists(mixed_precision_path) and os.path.exists(batch_data_path)):
        print(f"Error: Required files for mixed precision comparison not found")
        return
    
    mixed_data = load_jsonl(mixed_precision_path)
    batch_data = load_jsonl(batch_data_path)
    
    # Extract and combine metrics
    compare_data = []
    
    # First process the mixed precision data
    mp_batch_sizes = set()
    for data in mixed_data:
        batch_size = data['batch_size']
        mp_batch_sizes.add(batch_size)
        compare_data.append({
            'batch_size': batch_size,
            'precision': 'Mixed Precision',
            'latency_ms': data['metrics']['latency_ms_mean'],
            'throughput': data['metrics']['throughput_mean'],
            'gpu_memory_mb': data['metrics'].get('gpu_memory_mb_max', 0),
            'gpu_power_w': data['metrics'].get('gpu_avg_power_w', np.nan)
        })
    
    # Then find matching batch sizes in the FP32 data
    for data in batch_data:
        batch_size = data['batch_size']
        if batch_size in mp_batch_sizes:
            compare_data.append({
                'batch_size': batch_size,
                'precision': 'FP32',
                'latency_ms': data['metrics']['latency_ms_mean'],
                'throughput': data['metrics']['throughput_mean'],
                'gpu_memory_mb': data['metrics'].get('gpu_memory_mb_max', 0),
                'gpu_power_w': data['metrics'].get('gpu_avg_power_w', np.nan)
            })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(compare_data)
    
    # Check if we have data to plot
    if len(df) == 0 or 'FP32' not in df['precision'].values:
        print("Error: No matching FP32 data found for mixed precision comparison")
        return
    
    # Get unique batch sizes for x-axis
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create precision output directory if not exists
    precision_dir = os.path.join(output_dir, "precision")
    os.makedirs(precision_dir, exist_ok=True)
    
    # 1. Throughput comparison
    plt.figure(figsize=(12, 8))
    
    bar_width = 0.35
    index = np.arange(len(batch_sizes))
    
    # Initialize arrays to store values
    fp32_throughput = []
    amp_throughput = []
    
    # Gather data for each batch size
    for batch_size in batch_sizes:
        batch_df = df[df['batch_size'] == batch_size]
        
        fp32_row = batch_df[batch_df['precision'] == 'FP32']
        amp_row = batch_df[batch_df['precision'] == 'Mixed Precision']
        
        fp32_val = fp32_row['throughput'].values[0] if len(fp32_row) > 0 else 0
        amp_val = amp_row['throughput'].values[0] if len(amp_row) > 0 else 0
        
        fp32_throughput.append(fp32_val)
        amp_throughput.append(amp_val)
    
    # Create the bar charts
    bars1 = plt.bar(index, fp32_throughput, bar_width, label='FP32', color='#1f77b4')
    bars2 = plt.bar(index + bar_width, amp_throughput, bar_width, label='Mixed Precision', color='#ff7f0e')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/second)')
    plt.title('DistilBERT Throughput: FP32 vs Mixed Precision')
    plt.xticks(index + bar_width/2, batch_sizes)
    plt.legend()
    
    # Add value annotations above each bar
    for i, v in enumerate(fp32_throughput):
        if v > 0:
            plt.text(i, v * 1.01, f"{v:.1f}", ha='center', va='bottom')
    
    for i, v in enumerate(amp_throughput):
        if v > 0:
            plt.text(i + bar_width, v * 1.01, f"{v:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(precision_dir, "throughput_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Latency comparison
    plt.figure(figsize=(12, 8))
    
    # Initialize arrays to store values
    fp32_latency = []
    amp_latency = []
    
    # Gather data for each batch size
    for batch_size in batch_sizes:
        batch_df = df[df['batch_size'] == batch_size]
        
        fp32_row = batch_df[batch_df['precision'] == 'FP32']
        amp_row = batch_df[batch_df['precision'] == 'Mixed Precision']
        
        fp32_val = fp32_row['latency_ms'].values[0] if len(fp32_row) > 0 else 0
        amp_val = amp_row['latency_ms'].values[0] if len(amp_row) > 0 else 0
        
        fp32_latency.append(fp32_val)
        amp_latency.append(amp_val)
    
    # Create the bar charts
    bars1 = plt.bar(index, fp32_latency, bar_width, label='FP32', color='#1f77b4')
    bars2 = plt.bar(index + bar_width, amp_latency, bar_width, label='Mixed Precision', color='#ff7f0e')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title('DistilBERT Latency: FP32 vs Mixed Precision')
    plt.xticks(index + bar_width/2, batch_sizes)
    plt.legend()
    
    # Add value annotations above each bar
    for i, v in enumerate(fp32_latency):
        if v > 0:
            plt.text(i, v * 1.01, f"{v:.1f}", ha='center', va='bottom')
    
    for i, v in enumerate(amp_latency):
        if v > 0:
            plt.text(i + bar_width, v * 1.01, f"{v:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(precision_dir, "latency_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Speedup percentage (only for batch sizes that have both FP32 and mixed precision data)
    plt.figure(figsize=(12, 8))
    
    # Calculate improvement percentages
    valid_indices = []
    throughput_improvement = []
    latency_improvement = []
    
    for i, (batch_size, fp32_t, amp_t, fp32_l, amp_l) in enumerate(zip(batch_sizes, fp32_throughput, amp_throughput, fp32_latency, amp_latency)):
        if fp32_t > 0 and amp_t > 0 and fp32_l > 0 and amp_l > 0:
            valid_indices.append(i)
            throughput_improvement.append((amp_t - fp32_t) / fp32_t * 100)
            latency_improvement.append((fp32_l - amp_l) / fp32_l * 100)
    
    if not valid_indices:
        print("Warning: No valid data for speedup percentage calculation")
        return
    
    valid_index = np.array(valid_indices)
    valid_batch_sizes = [batch_sizes[i] for i in valid_indices]
    
    plt.bar(valid_index - bar_width/2, throughput_improvement, bar_width, label='Throughput Improvement', color='#2ca02c')
    plt.bar(valid_index + bar_width/2, latency_improvement, bar_width, label='Latency Improvement', color='#d62728')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Improvement (%)')
    plt.title('DistilBERT Mixed Precision Improvement')
    plt.xticks(valid_index, valid_batch_sizes)
    plt.legend()
    
    # Add value annotations above each bar
    for i, v, idx in zip(range(len(throughput_improvement)), throughput_improvement, valid_indices):
        plt.text(idx - bar_width/2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom')
    
    for i, v, idx in zip(range(len(latency_improvement)), latency_improvement, valid_indices):
        plt.text(idx + bar_width/2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(precision_dir, "speedup_percentage.png"), dpi=300)
    plt.close()
    
    print("Mixed precision comparison plots generated successfully.")

def plot_cpu_gpu_comparison(input_dir, output_dir):
    """Generate plots comparing CPU vs GPU performance."""
    # Load CPU and GPU data
    cpu_results_path = os.path.join(input_dir, "cpu_results.jsonl")
    batch_data_path = os.path.join(input_dir, "batch_comparison.jsonl")
    
    if not (os.path.exists(cpu_results_path) and os.path.exists(batch_data_path)):
        print(f"Error: Required files not found")
        return
    
    cpu_data = load_jsonl(cpu_results_path)
    gpu_data = load_jsonl(batch_data_path)
    
    # Create CPU/GPU output directory if not exists
    cpu_gpu_dir = os.path.join(output_dir, "cpu_gpu")
    os.makedirs(cpu_gpu_dir, exist_ok=True)
    
    # Extract and combine metrics
    compare_data = []
    
    # Process CPU data
    for data in cpu_data:
        batch_size = data['batch_size']
        compare_data.append({
            'batch_size': batch_size,
            'device': 'CPU',
            'latency_ms': data['metrics']['latency_ms_mean'],
            'throughput': data['metrics']['throughput_mean'],
            'memory_mb': data['metrics']['cpu_memory_mb_max']
        })
    
    # Process GPU data - keep only matching batch sizes
    cpu_batch_sizes = [data['batch_size'] for data in cpu_data]
    for data in gpu_data:
        batch_size = data['batch_size']
        if batch_size in cpu_batch_sizes:
            compare_data.append({
                'batch_size': batch_size,
                'device': 'GPU',
                'latency_ms': data['metrics']['latency_ms_mean'],
                'throughput': data['metrics']['throughput_mean'],
                'memory_mb': data['metrics']['cpu_memory_mb_max']
            })
    
    df = pd.DataFrame(compare_data)
    
    # Get unique batch sizes for x-axis
    batch_sizes = sorted(df['batch_size'].unique())
    
    # 1. Throughput comparison (log scale)
    plt.figure(figsize=(12, 8))
    
    bar_width = 0.35
    index = np.arange(len(batch_sizes))
    
    cpu_throughput = df[df['device'] == 'CPU'].sort_values('batch_size')['throughput'].values
    gpu_throughput = df[df['device'] == 'GPU'].sort_values('batch_size')['throughput'].values
    
    bars1 = plt.bar(index, cpu_throughput, bar_width, label='CPU', color='#1f77b4')
    bars2 = plt.bar(index + bar_width, gpu_throughput, bar_width, label='GPU', color='#ff7f0e')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/second)')
    plt.title('DistilBERT Throughput: CPU vs GPU')
    plt.xticks(index + bar_width/2, batch_sizes)
    plt.legend()
    plt.yscale('log')
    
    # Add value annotations above each bar
    for i, v in enumerate(cpu_throughput):
        plt.text(i, v * 1.1, f"{v:.1f}", ha='center', va='bottom')
    
    for i, v in enumerate(gpu_throughput):
        plt.text(i + bar_width, v * 1.1, f"{v:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cpu_gpu_dir, "throughput_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Latency comparison (log scale)
    plt.figure(figsize=(12, 8))
    
    cpu_latency = df[df['device'] == 'CPU'].sort_values('batch_size')['latency_ms'].values
    gpu_latency = df[df['device'] == 'GPU'].sort_values('batch_size')['latency_ms'].values
    
    bars1 = plt.bar(index, cpu_latency, bar_width, label='CPU', color='#1f77b4')
    bars2 = plt.bar(index + bar_width, gpu_latency, bar_width, label='GPU', color='#ff7f0e')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title('DistilBERT Latency: CPU vs GPU')
    plt.xticks(index + bar_width/2, batch_sizes)
    plt.legend()
    plt.yscale('log')
    
    # Add value annotations above each bar
    for i, v in enumerate(cpu_latency):
        plt.text(i, v * 1.1, f"{v:.1f}", ha='center', va='bottom')
    
    for i, v in enumerate(gpu_latency):
        plt.text(i + bar_width, v * 1.1, f"{v:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cpu_gpu_dir, "latency_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Speedup factor
    plt.figure(figsize=(12, 8))
    
    # Calculate speedup factors
    throughput_speedup = [gpu / cpu for cpu, gpu in zip(cpu_throughput, gpu_throughput)]
    latency_speedup = [cpu / gpu for cpu, gpu in zip(cpu_latency, gpu_latency)]
    
    plt.bar(index - bar_width/2, throughput_speedup, bar_width, label='Throughput Speedup', color='#2ca02c')
    plt.bar(index + bar_width/2, latency_speedup, bar_width, label='Latency Speedup', color='#d62728')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor (GPU/CPU)')
    plt.title('DistilBERT GPU Speedup Factors')
    plt.xticks(index, batch_sizes)
    plt.legend()
    
    # Add value annotations above each bar
    for i, v in enumerate(throughput_speedup):
        plt.text(i - bar_width/2, v + 0.5, f"{v:.1f}x", ha='center', va='bottom')
    
    for i, v in enumerate(latency_speedup):
        plt.text(i + bar_width/2, v + 0.5, f"{v:.1f}x", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cpu_gpu_dir, "speedup_factors.png"), dpi=300)
    plt.close()
    
    print("CPU vs GPU comparison plots generated successfully.")

def create_mixed_precision_batch_sweep(output_dir):
    """Generate a plan for mixed precision batch size sweep."""
    sweep_dir = os.path.join(output_dir, "mixed_precision_batch_sweep_plan.md")
    
    content = f"""# Mixed Precision Batch Size Sweep Plan

## Objective
Extend the existing benchmark with mixed precision (AMP) tests across all batch sizes to identify optimal throughput and latency configurations.

## Current Findings
- Mixed precision has been tested with batch sizes 8, 16, and 32
- Initial results show ~4% throughput improvement at batch size 8
- Larger batch sizes should be tested to determine if scaling improves with mixed precision

## Proposed Test Matrix

| Batch Size | FP32 (done) | Mixed Precision |
|------------|-------------|-----------------|
| 8          | ✅          | ✅              |
| 16         | ✅          | ✅              |
| 32         | ✅          | ✅              |
| 64         | ✅          | 🔲              |
| 128        | ✅          | 🔲              |
| 256        | 🔲          | 🔲              |

## Implementation Steps

1. Extend the existing benchmark script to support larger batch sizes with mixed precision
2. Run the following command for each missing configuration:

```bash
python benchmark_distilbert.py \\
  --model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english" \\
  --batch_size=64 \\
  --device="cuda" \\
  --mixed_precision \\
  --output_dir="distilbert_outputs"
```

3. Run the same for batch size 128 and 256 (if GPU memory allows)
4. Rerun the comparison analysis with the new complete dataset

## Expected Benefits

- Confirm if mixed precision provides greater scaling efficiency at larger batch sizes
- Identify the optimal batch size for mixed precision inference
- Quantify power efficiency improvements with mixed precision

## Data Analysis

Update the plotting scripts to include all new data points and generate:
1. Combined throughput plots showing FP32 vs AMP across all batch sizes
2. Scaling efficiency analysis comparing FP32 and AMP
3. Energy efficiency metrics (samples/joule) for each configuration

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(sweep_dir, 'w') as f:
        f.write(content)
    
    print(f"Mixed precision batch sweep plan created at {sweep_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate DistilBERT benchmark comparison plots")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help=f"Input directory containing benchmark results (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory for plots (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()

def main():
    """Main function to run all plot generation."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print("Generating DistilBERT benchmark plots...")
    plot_batch_comparisons(args.input, args.output)
    plot_mixed_precision_comparison(args.input, args.output)
    plot_cpu_gpu_comparison(args.input, args.output)
    create_mixed_precision_batch_sweep(args.output)
    print("All benchmark plots generated successfully.")

if __name__ == "__main__":
    main() 