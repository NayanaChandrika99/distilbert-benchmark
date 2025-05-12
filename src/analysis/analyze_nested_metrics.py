import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# Path to your results directory
results_dir = "distilbert_outputs"
output_dir = os.path.join(results_dir, "analysis_csv")
os.makedirs(output_dir, exist_ok=True)
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

print(f"Analyzing benchmark results in {results_dir}")
print(f"Output will be saved to {output_dir}")

# Function to load and flatten JSONL files with nested metrics
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            # Extract nested metrics and flatten them
            if 'metrics' in record:
                metrics = record.pop('metrics')
                record.update(metrics)  # Add metrics fields to main record
            data.append(record)
    return pd.DataFrame(data)

# Process batch comparison data
batch_file = os.path.join(results_dir, "batch_comparison.jsonl")
if os.path.exists(batch_file):
    print(f"Loading batch comparison data from {batch_file}")
    df = load_jsonl(batch_file)
    
    print(f"Available columns: {list(df.columns)}")
    
    # Calculate per-sample metrics
    df['per_sample_latency_ms'] = df['latency_ms_mean'] / df['batch_size']
    if 'gpu_memory_mb_max' in df.columns:
        df['memory_per_sample_MB'] = df['gpu_memory_mb_max'] / df['batch_size']
    
    # Calculate scaling efficiency (relative to ideal linear scaling)
    smallest_batch = df['batch_size'].min()
    baseline = df[df['batch_size'] == smallest_batch].iloc[0]
    df['scaling_efficiency'] = (df['throughput_mean'] / baseline['throughput_mean']) / (df['batch_size'] / smallest_batch)
    
    # Save to CSV
    batch_csv = os.path.join(output_dir, "batch_analysis.csv")
    
    # Select relevant columns for CSV export
    core_columns = ['batch_size', 'latency_ms_mean', 'per_sample_latency_ms', 'throughput_mean', 'scaling_efficiency']
    if 'gpu_memory_mb_max' in df.columns:
        core_columns.extend(['gpu_memory_mb_max', 'memory_per_sample_MB'])
    
    df[core_columns].sort_values('batch_size').to_csv(batch_csv, index=False)
    print(f"Saved batch analysis to {batch_csv}")
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='batch_size', y='throughput_mean', data=df, marker='o')
    plt.title('Throughput vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/second)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'throughput.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='batch_size', y='scaling_efficiency', data=df, marker='o')
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal scaling')
    plt.title('Scaling Efficiency vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Scaling Efficiency (closer to 1.0 is better)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'scaling_efficiency.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='batch_size', y='per_sample_latency_ms', data=df, marker='o')
    plt.title('Per-Sample Latency vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Latency per Sample (ms)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'per_sample_latency.png'))
    plt.close()
    
    if 'memory_per_sample_MB' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='batch_size', y='memory_per_sample_MB', data=df, marker='o')
        plt.title('Memory Usage per Sample vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Memory per Sample (MB)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'memory_per_sample.png'))
        plt.close()
    
    # Generate a report explaining the observed patterns
    report_lines = []
    report_lines.append("# DistilBERT Batch Size Performance Analysis\n")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get key metrics
    max_throughput_idx = df['throughput_mean'].idxmax()
    max_throughput_batch = df.loc[max_throughput_idx]
    small_batch = df[df['batch_size'] == smallest_batch].iloc[0]
    large_batch = df[df['batch_size'] == df['batch_size'].max()].iloc[0]
    
    # Add throughput analysis
    report_lines.append("## Throughput Analysis\n")
    report_lines.append(f"- Small batch (size {small_batch['batch_size']}) throughput: {small_batch['throughput_mean']:.2f} samples/second")
    report_lines.append(f"- Optimal batch (size {max_throughput_batch['batch_size']}) throughput: {max_throughput_batch['throughput_mean']:.2f} samples/second")
    report_lines.append(f"- Large batch (size {large_batch['batch_size']}) throughput: {large_batch['throughput_mean']:.2f} samples/second")
    report_lines.append(f"- Throughput improvement from small to optimal: {(max_throughput_batch['throughput_mean']/small_batch['throughput_mean'] - 1)*100:.1f}%")
    
    if large_batch['throughput_mean'] < max_throughput_batch['throughput_mean']:
        report_lines.append(f"- **Throughput decline from optimal to large batch: {(1 - large_batch['throughput_mean']/max_throughput_batch['throughput_mean'])*100:.1f}%**")
    
    # Root causes analysis
    report_lines.append("\n## Root Causes of Observed Performance Patterns\n")
    
    # Small batch analysis
    report_lines.append("### 1. Low Throughput at Small Batch Sizes\n")
    report_lines.append("**Root causes:**")
    report_lines.append("- **Fixed overhead dominance**: Each batch incurs constant costs regardless of size")
    report_lines.append("  - CUDA kernel launch overhead (microseconds per launch)")
    report_lines.append("  - Memory allocation/deallocation costs")
    report_lines.append("  - Execution queue management")
    
    report_lines.append("- **GPU underutilization**: Small batches don't provide enough parallelism")
    report_lines.append("  - NVIDIA GPUs have thousands of CUDA cores that remain partly idle")
    report_lines.append("  - The GPU's SIMD architecture is most efficient with high parallelism")
    
    # Optimal batch analysis
    report_lines.append("\n### 2. Optimal Throughput at Medium Batch Sizes\n")
    report_lines.append("**Root causes:**")
    report_lines.append("- **Parallelism saturation**: The GPU reaches high occupancy")
    report_lines.append("  - Most CUDA cores are active and processing useful work")
    report_lines.append("  - Memory access patterns become more coalesced")
    
    report_lines.append("- **Balanced resource usage**: Balance between:")
    report_lines.append("  - Compute resources (CUDA cores)")
    report_lines.append("  - Memory bandwidth")
    report_lines.append("  - Cache utilization")
    
    # Large batch analysis (if there's a performance decline)
    if large_batch['throughput_mean'] < max_throughput_batch['throughput_mean']:
        report_lines.append("\n### 3. Declining Throughput at Large Batch Sizes\n")
        report_lines.append("**Root causes:**")
        
        report_lines.append("- **Memory constraints**: ")
        if 'gpu_memory_mb_max' in df.columns:
            report_lines.append(f"  - Memory usage increases from {max_throughput_batch['gpu_memory_mb_max']:.1f}MB at optimal batch to {large_batch['gpu_memory_mb_max']:.1f}MB at largest batch")
        report_lines.append("  - Increased cache pressure and thrashing")
        report_lines.append("  - Internal memory fragmentation increases")
        
        report_lines.append("- **Execution serialization**:")
        report_lines.append("  - More serialization of operations occurs as batch size grows")
        report_lines.append("  - Memory transfer operations start blocking compute operations")
    
    # Recommendations
    report_lines.append("\n## Recommendations\n")
    report_lines.append(f"- **For maximum throughput**: Use batch size {max_throughput_batch['batch_size']}")
    
    # Write the report
    report_path = os.path.join(output_dir, "batch_size_analysis.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Generated analysis report: {report_path}")

# Process CPU vs GPU comparison if both files exist
cpu_file = os.path.join(results_dir, "cpu_results.jsonl")
gpu_file = os.path.join(results_dir, "absolute_results.jsonl")

if os.path.exists(cpu_file) and os.path.exists(gpu_file):
    print(f"Processing CPU vs GPU comparison")
    cpu_df = load_jsonl(cpu_file)
    gpu_df = load_jsonl(gpu_file)
    
    # Ensure we have common batch sizes for comparison
    common_batch_sizes = set(cpu_df['batch_size']).intersection(set(gpu_df['batch_size']))
    
    if common_batch_sizes:
        # Create a comparison dataframe
        comparison = []
        for bs in sorted(common_batch_sizes):
            cpu_row = cpu_df[cpu_df['batch_size'] == bs].iloc[0]
            gpu_row = gpu_df[gpu_df['batch_size'] == bs].iloc[0]
            
            comparison.append({
                'batch_size': bs,
                'cpu_latency_ms': cpu_row['latency_ms_mean'],
                'gpu_latency_ms': gpu_row['latency_ms_mean'],
                'cpu_throughput_sps': cpu_row['throughput_mean'],
                'gpu_throughput_sps': gpu_row['throughput_mean'],
                'speedup_factor': gpu_row['throughput_mean'] / cpu_row['throughput_mean'],
                'latency_reduction': 1 - (gpu_row['latency_ms_mean'] / cpu_row['latency_ms_mean'])
            })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        # Save to CSV
        comparison_csv = os.path.join(output_dir, "cpu_gpu_comparison.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Saved CPU vs GPU comparison to {comparison_csv}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='batch_size', y='speedup_factor', data=comparison_df, marker='o')
        plt.title('GPU Speedup Factor vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup Factor (GPU/CPU)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'gpu_speedup_factor.png'))
        plt.close()

# Process mixed precision comparison if both files exist
mp_file = os.path.join(results_dir, "mixed_precision_results.jsonl")
if os.path.exists(mp_file) and os.path.exists(batch_file):
    print(f"Processing mixed precision comparison")
    mp_df = load_jsonl(mp_file)
    reg_df = load_jsonl(batch_file)  # Regular precision results
    
    # Find common batch sizes
    common_batch_sizes = set(mp_df['batch_size']).intersection(set(reg_df['batch_size']))
    
    if common_batch_sizes:
        # Create comparison dataframe
        precision_comparison = []
        for bs in sorted(common_batch_sizes):
            mp_row = mp_df[mp_df['batch_size'] == bs].iloc[0]
            reg_row = reg_df[reg_df['batch_size'] == bs].iloc[0]
            
            precision_comparison.append({
                'batch_size': bs,
                'mixed_precision_latency_ms': mp_row['latency_ms_mean'],
                'regular_precision_latency_ms': reg_row['latency_ms_mean'],
                'mixed_precision_throughput': mp_row['throughput_mean'],
                'regular_precision_throughput': reg_row['throughput_mean'],
                'throughput_improvement': mp_row['throughput_mean'] / reg_row['throughput_mean'],
                'latency_reduction': 1 - (mp_row['latency_ms_mean'] / reg_row['latency_ms_mean'])
            })
        
        # Convert to DataFrame
        precision_df = pd.DataFrame(precision_comparison)
        
        # Save to CSV
        precision_csv = os.path.join(output_dir, "precision_comparison.csv")
        precision_df.to_csv(precision_csv, index=False)
        print(f"Saved precision comparison to {precision_csv}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='batch_size', y='throughput_improvement', data=precision_df, marker='o')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No improvement')
        plt.title('Mixed Precision Throughput Improvement vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput Improvement Factor')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'mixed_precision_improvement.png'))
        plt.close()

print(f"\nAll analyses completed. Results saved to {output_dir}") 