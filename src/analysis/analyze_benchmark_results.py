import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime

# Path to your results directory
results_dir = "distilbert_outputs"
output_dir = os.path.join(results_dir, "analysis_csv")
os.makedirs(output_dir, exist_ok=True)

print(f"Analyzing benchmark results in {results_dir}")
print(f"Output will be saved to {output_dir}")

# Function to load JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Process each JSONL file
jsonl_files = {
    "batch_comparison": os.path.join(results_dir, "batch_comparison.jsonl"),
    "cpu_results": os.path.join(results_dir, "cpu_results.jsonl"),
    "mixed_precision": os.path.join(results_dir, "mixed_precision_results.jsonl"),
    "absolute_results": os.path.join(results_dir, "absolute_results.jsonl")
}

results = {}
for name, file_path in jsonl_files.items():
    if os.path.exists(file_path):
        print(f"Loading {name}...")
        df = load_jsonl(file_path)
        
        # Print the columns for debugging
        print(f"  Columns in {name}: {list(df.columns)}")
        
        results[name] = df
    else:
        print(f"Warning: {name} file not found at {file_path}")

# Function to get standardized column name
def get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

# Create detailed analysis of batch size effects
if "batch_comparison" in results:
    df = results["batch_comparison"]
    
    # Define possible column names
    latency_names = ['latency_mean_ms', 'latency_ms', 'latency', 'mean_latency_ms']
    throughput_names = ['throughput_mean_sps', 'throughput_sps', 'throughput', 'mean_throughput_sps']
    batch_size_names = ['batch_size', 'batch']
    memory_names = ['gpu_memory_max_mb', 'gpu_memory_mb', 'memory_mb', 'max_memory_mb']
    
    # Get actual column names
    latency_col = get_column(df, latency_names)
    throughput_col = get_column(df, throughput_names)
    batch_size_col = get_column(df, batch_size_names)
    memory_col = get_column(df, memory_names)
    
    print(f"\nUsing columns:")
    print(f"  Latency: {latency_col}")
    print(f"  Throughput: {throughput_col}")
    print(f"  Batch size: {batch_size_col}")
    print(f"  Memory: {memory_col}")
    
    if not all([latency_col, throughput_col, batch_size_col]):
        print("ERROR: Could not find required columns in batch comparison results")
    else:
        # 1. Compute per-sample and efficiency metrics
        df['per_sample_latency_ms'] = df[latency_col] / df[batch_size_col]
        
        if memory_col:
            df['memory_per_sample_MB'] = df[memory_col] / df[batch_size_col]
        
        # 2. Calculate scaling efficiency (relative to ideal linear scaling)
        # Get the smallest batch size as baseline
        smallest_batch = df[batch_size_col].min()
        baseline = df[df[batch_size_col] == smallest_batch].iloc[0]
        
        # Calculate how close to ideal scaling we get (1.0 = perfect scaling)
        df['scaling_efficiency'] = (df[throughput_col] / baseline[throughput_col]) / (df[batch_size_col] / smallest_batch)
        
        if memory_col:
            # 3. Calculate efficiency metrics
            df['gpu_memory_efficiency'] = baseline[memory_col] / df[memory_col] * df[batch_size_col] / smallest_batch
        
        # Try to extract component times if available
        component_times = []
        for column in df.columns:
            if "_time_ms" in column and column not in [latency_col, f"{latency_col}_std"]:
                component_times.append(column)
        
        if component_times:
            # Calculate overhead percentage if component timing data is available
            compute_columns = [col for col in component_times if "forward" in col or "backward" in col]
            overhead_columns = [col for col in component_times if col not in compute_columns]
            
            if compute_columns and overhead_columns:
                df['compute_time_ms'] = df[compute_columns].sum(axis=1)
                df['overhead_time_ms'] = df[overhead_columns].sum(axis=1)
                df['overhead_percentage'] = (df['overhead_time_ms'] / df[latency_col]) * 100
        
        # 4. Select and reorder columns for clarity
        core_metrics = [
            batch_size_col,
            latency_col,
            'per_sample_latency_ms',
            throughput_col,
            'scaling_efficiency'
        ]
        
        # Add memory metrics if available
        memory_metrics = []
        if memory_col:
            memory_metrics = [
                memory_col,
                'memory_per_sample_MB',
                'gpu_memory_efficiency'
            ]
        
        # Add overhead metrics if available
        overhead_metrics = []
        if 'overhead_percentage' in df.columns:
            overhead_metrics = [
                'compute_time_ms',
                'overhead_time_ms',
                'overhead_percentage'
            ]
        
        # Combine all available metrics
        all_metrics = core_metrics + memory_metrics + overhead_metrics
        available_metrics = [col for col in all_metrics if col in df.columns]
        
        metrics_df = df[available_metrics].sort_values(batch_size_col)
        
        # Rename columns for consistent CSV output
        column_mapping = {
            batch_size_col: 'batch_size',
            latency_col: 'latency_ms',
            throughput_col: 'throughput_sps'
        }
        if memory_col:
            column_mapping[memory_col] = 'gpu_memory_mb'
            
        metrics_df = metrics_df.rename(columns=column_mapping)
        
        # 5. Save to CSV
        csv_path = os.path.join(output_dir, "batch_size_analysis.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved detailed batch size metrics to {csv_path}")
        
        # 6. Generate visualizations
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Scaling efficiency plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='batch_size', y='scaling_efficiency', data=metrics_df, marker='o')
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal scaling')
        plt.title('Scaling Efficiency vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Scaling Efficiency (closer to 1.0 is better)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'scaling_efficiency.png'))
        plt.close()
        
        # Per-sample latency plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='batch_size', y='per_sample_latency_ms', data=metrics_df, marker='o')
        plt.title('Per-Sample Latency vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency per Sample (ms)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'per_sample_latency.png'))
        plt.close()
        
        if 'memory_per_sample_MB' in metrics_df.columns:
            # Memory efficiency plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='batch_size', y='memory_per_sample_MB', data=metrics_df, marker='o')
            plt.title('Memory Usage per Sample vs Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Memory per Sample (MB)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plot_dir, 'memory_per_sample.png'))
            plt.close()
        
        # Generate bottleneck analysis report
        report_lines = []
        report_lines.append("# DistilBERT Batch Size Bottleneck Analysis\n")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add analysis of scaling efficiency
        report_lines.append("## Scaling Efficiency Analysis\n")
        most_efficient_idx = metrics_df['scaling_efficiency'].idxmax()
        most_efficient_batch = metrics_df.loc[most_efficient_idx]
        report_lines.append(f"- Most efficient batch size: {most_efficient_batch['batch_size']} (scaling efficiency: {most_efficient_batch['scaling_efficiency']:.2f})")
        
        last_idx = len(metrics_df) - 1
        report_lines.append(f"- Largest batch size efficiency: {metrics_df.iloc[last_idx]['scaling_efficiency']:.2f} " +
                            f"(batch size {metrics_df.iloc[last_idx]['batch_size']})")
        
        if metrics_df.iloc[last_idx]['scaling_efficiency'] < 0.7:
            report_lines.append("- **Significant scaling efficiency loss detected at larger batch sizes**")
            report_lines.append("  - This suggests memory bandwidth limitations or compute serialization")
        
        # Add throughput analysis
        report_lines.append("\n## Throughput Analysis\n")
        max_throughput_idx = metrics_df['throughput_sps'].idxmax()
        max_throughput_batch = metrics_df.loc[max_throughput_idx]
        report_lines.append(f"- Maximum throughput: {max_throughput_batch['throughput_sps']:.2f} samples/second " +
                            f"at batch size {max_throughput_batch['batch_size']}")
        
        # Calculate throughput ratio between batch sizes
        small_batch_throughput = metrics_df.iloc[0]['throughput_sps']
        optimal_batch_throughput = max_throughput_batch['throughput_sps']
        large_batch_throughput = metrics_df.iloc[last_idx]['throughput_sps']
        
        report_lines.append(f"- Small batch (size {metrics_df.iloc[0]['batch_size']}) throughput: {small_batch_throughput:.2f} samples/second")
        report_lines.append(f"- Optimal batch (size {max_throughput_batch['batch_size']}) throughput: {optimal_batch_throughput:.2f} samples/second")
        report_lines.append(f"- Large batch (size {metrics_df.iloc[last_idx]['batch_size']}) throughput: {large_batch_throughput:.2f} samples/second")
        report_lines.append(f"- Throughput improvement from small to optimal: {(optimal_batch_throughput/small_batch_throughput - 1)*100:.1f}%")
        
        if large_batch_throughput < optimal_batch_throughput:
            report_lines.append(f"- **Throughput decline from optimal to large batch: {(1 - large_batch_throughput/optimal_batch_throughput)*100:.1f}%**")
            report_lines.append("  - This indicates resource limitations (memory bandwidth, cache pressure, etc.)")
        
        # Add memory analysis if available
        if 'memory_per_sample_MB' in metrics_df.columns:
            report_lines.append("\n## Memory Efficiency Analysis\n")
            most_memory_efficient_idx = metrics_df['memory_per_sample_MB'].idxmin()
            most_memory_efficient = metrics_df.loc[most_memory_efficient_idx]
            report_lines.append(f"- Most memory-efficient batch size: {most_memory_efficient['batch_size']} " +
                                f"({most_memory_efficient['memory_per_sample_MB']:.2f} MB per sample)")
            
            first_to_last_ratio = metrics_df.iloc[0]['memory_per_sample_MB'] / metrics_df.iloc[last_idx]['memory_per_sample_MB']
            report_lines.append(f"- Memory efficiency improvement, small to large batch: {first_to_last_ratio:.2f}x")
            
            report_lines.append(f"- GPU memory usage at smallest batch size: {metrics_df.iloc[0]['gpu_memory_mb']:.1f} MB")
            report_lines.append(f"- GPU memory usage at largest batch size: {metrics_df.iloc[last_idx]['gpu_memory_mb']:.1f} MB")
            report_lines.append(f"- Memory usage increase factor: {metrics_df.iloc[last_idx]['gpu_memory_mb']/metrics_df.iloc[0]['gpu_memory_mb']:.1f}x")
        
        # Add latency analysis
        report_lines.append("\n## Latency Analysis\n")
        min_latency_idx = metrics_df['latency_ms'].idxmin()
        min_latency_batch = metrics_df.loc[min_latency_idx]
        report_lines.append(f"- Minimum absolute latency: {min_latency_batch['latency_ms']:.2f} ms at batch size {min_latency_batch['batch_size']}")
        
        min_per_sample_idx = metrics_df['per_sample_latency_ms'].idxmin()
        min_per_sample = metrics_df.loc[min_per_sample_idx]
        report_lines.append(f"- Minimum per-sample latency: {min_per_sample['per_sample_latency_ms']:.2f} ms at batch size {min_per_sample['batch_size']}")
        
        # Add component time analysis if available
        if 'overhead_percentage' in metrics_df.columns:
            report_lines.append("\n## Processing Overhead Analysis\n")
            for idx, row in metrics_df.iterrows():
                batch_size = row['batch_size']
                overhead_pct = row['overhead_percentage']
                report_lines.append(f"- Batch size {batch_size}: {overhead_pct:.1f}% overhead, {100-overhead_pct:.1f}% computation")
            
            small_batch = metrics_df.iloc[0]
            large_batch = metrics_df.iloc[last_idx]
            overhead_diff = small_batch['overhead_percentage'] - large_batch['overhead_percentage']
            
            if overhead_diff > 10:  # If overhead percentage drops significantly
                report_lines.append(f"\n- **Overhead decreases by {overhead_diff:.1f}% from small to large batches**")
                report_lines.append("  - This confirms that fixed costs (kernel launches, memory ops) are being amortized")
                report_lines.append("  - Small batches are overhead-dominated, explaining their lower throughput")
        
        # Conclusion and recommendations
        report_lines.append("\n## Root Causes of Observed Performance Patterns\n")
        
        # Small batch analysis
        report_lines.append("### 1. Low Throughput at Small Batch Sizes\n")
        report_lines.append("**Root causes:**")
        report_lines.append("- **Fixed overhead dominance**: Each batch incurs constant costs regardless of size")
        report_lines.append("  - CUDA kernel launch overhead (microseconds per launch)")
        report_lines.append("  - Memory allocation/deallocation costs")
        report_lines.append("  - Execution queue management")
        if 'overhead_percentage' in metrics_df.columns and metrics_df.iloc[0]['overhead_percentage'] > 40:
            report_lines.append(f"  - Measured overhead: {metrics_df.iloc[0]['overhead_percentage']:.1f}% at batch size {metrics_df.iloc[0]['batch_size']}")
        
        report_lines.append("- **GPU underutilization**: Small batches don't provide enough parallelism")
        report_lines.append("  - NVIDIA GPUs have thousands of CUDA cores that remain partly idle")
        report_lines.append("  - The GPU's SIMD architecture is most efficient with high parallelism")
        report_lines.append("  - Low occupancy means wasted computational capacity")
        
        report_lines.append("- **Data transfer inefficiency**: PCIe transfers have overhead costs")
        report_lines.append("  - Small transfers waste PCIe bandwidth due to protocol overhead")
        report_lines.append("  - Frequent small transfers can't achieve peak bandwidth utilization")
        
        # Optimal batch analysis
        report_lines.append("\n### 2. Optimal Throughput at Medium Batch Sizes\n")
        report_lines.append("**Root causes:**")
        report_lines.append("- **Parallelism saturation**: The GPU reaches high occupancy")
        report_lines.append("  - Most CUDA cores are active and processing useful work")
        report_lines.append("  - Warp execution efficiency is high")
        report_lines.append("  - Memory access patterns become more coalesced")
        
        report_lines.append("- **Balanced resource usage**: Balance between:")
        report_lines.append("  - Compute resources (CUDA cores)")
        report_lines.append("  - Memory bandwidth")
        report_lines.append("  - Cache utilization")
        report_lines.append("  - Register file usage")
        
        report_lines.append("- **Amortized overheads**: Fixed costs are spread across more samples")
        report_lines.append("  - Kernel launch overhead becomes negligible per sample")
        report_lines.append("  - Memory allocation costs are distributed effectively")
        
        # Large batch analysis
        if large_batch_throughput < optimal_batch_throughput:
            report_lines.append("\n### 3. Declining Throughput at Large Batch Sizes\n")
            report_lines.append("**Root causes:**")
            report_lines.append("- **Memory constraints**: ")
            if 'gpu_memory_mb' in metrics_df.columns:
                optimal_idx = metrics_df[metrics_df['batch_size'] == max_throughput_batch['batch_size']].index[0]
                report_lines.append(f"  - Memory usage increases from {metrics_df.iloc[optimal_idx]['gpu_memory_mb']:.1f}MB at optimal batch size to {metrics_df.iloc[last_idx]['gpu_memory_mb']:.1f}MB at largest batch")
            report_lines.append("  - Increased cache pressure and thrashing")
            report_lines.append("  - Internal memory fragmentation increases")
            
            report_lines.append("- **Execution serialization**:")
            report_lines.append("  - The scheduler faces more pressure as work grows")
            report_lines.append("  - More serialization of operations occurs")
            report_lines.append("  - Memory transfer operations start blocking compute operations")
            
            report_lines.append("- **Increased backward pass cost**:")
            report_lines.append("  - Gradient computation scales non-linearly with large batch sizes")
            report_lines.append("  - Memory requirements for storing activations increase")
            report_lines.append("  - The optimizer step becomes more expensive")
        
        # Recommendations
        report_lines.append("\n## Recommendations\n")
        report_lines.append(f"- **For maximum throughput**: Use batch size {max_throughput_batch['batch_size']}")
        report_lines.append(f"- **For maximum scaling efficiency**: Use batch size {most_efficient_batch['batch_size']}")
        report_lines.append(f"- **For minimum latency**: Use batch size {min_latency_batch['batch_size']}")
        
        if 'memory_per_sample_MB' in metrics_df.columns:
            report_lines.append(f"- **For optimal memory usage**: Use batch size {most_memory_efficient['batch_size']}")
        
        # Write out the report
        report_path = os.path.join(output_dir, "batch_size_bottleneck_analysis.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Generated detailed bottleneck analysis report: {report_path}")
        print(f"Generated plots in: {plot_dir}")

# Process CPU vs GPU comparison
if "cpu_results" in results and "absolute_results" in results:
    try:
        cpu_df = results["cpu_results"]
        gpu_df = results["absolute_results"]
        
        # Define possible column names
        latency_names = ['latency_mean_ms', 'latency_ms', 'latency', 'mean_latency_ms']
        throughput_names = ['throughput_mean_sps', 'throughput_sps', 'throughput', 'mean_throughput_sps']
        batch_size_names = ['batch_size', 'batch']
        
        # Get actual column names
        cpu_latency_col = get_column(cpu_df, latency_names)
        cpu_throughput_col = get_column(cpu_df, throughput_names)
        cpu_batch_size_col = get_column(cpu_df, batch_size_names)
        
        gpu_latency_col = get_column(gpu_df, latency_names)
        gpu_throughput_col = get_column(gpu_df, throughput_names)
        gpu_batch_size_col = get_column(gpu_df, batch_size_names)
        
        print(f"\nCPU vs GPU comparison columns:")
        print(f"  CPU: Latency={cpu_latency_col}, Throughput={cpu_throughput_col}, Batch size={cpu_batch_size_col}")
        print(f"  GPU: Latency={gpu_latency_col}, Throughput={gpu_throughput_col}, Batch size={gpu_batch_size_col}")
        
        # Ensure we have common batch sizes for comparison
        common_batch_sizes = set(cpu_df[cpu_batch_size_col]).intersection(set(gpu_df[gpu_batch_size_col]))
        
        if common_batch_sizes:
            # Filter to common batch sizes
            cpu_common = cpu_df[cpu_df[cpu_batch_size_col].isin(common_batch_sizes)]
            gpu_common = gpu_df[gpu_df[gpu_batch_size_col].isin(common_batch_sizes)]
            
            # Create a comparison dataframe
            comparison = []
            for bs in sorted(common_batch_sizes):
                cpu_row = cpu_df[cpu_df[cpu_batch_size_col] == bs].iloc[0]
                gpu_row = gpu_df[gpu_df[gpu_batch_size_col] == bs].iloc[0]
                
                comparison.append({
                    'batch_size': bs,
                    'cpu_latency_ms': cpu_row[cpu_latency_col],
                    'gpu_latency_ms': gpu_row[gpu_latency_col],
                    'cpu_throughput_sps': cpu_row[cpu_throughput_col],
                    'gpu_throughput_sps': gpu_row[gpu_throughput_col],
                    'speedup_factor': gpu_row[gpu_throughput_col] / cpu_row[cpu_throughput_col],
                    'latency_reduction': 1 - (gpu_row[gpu_latency_col] / cpu_row[cpu_latency_col])
                })
            
            # Convert to DataFrame
            comparison_df = pd.DataFrame(comparison)
            
            # Save to CSV
            comparison_csv = os.path.join(output_dir, "cpu_gpu_comparison.csv")
            comparison_df.to_csv(comparison_csv, index=False)
            print(f"Saved CPU vs GPU comparison to {comparison_csv}")
            
            # Create speedup visualization
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='batch_size', y='speedup_factor', data=comparison_df, marker='o')
            plt.title('GPU Speedup Factor vs. Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Speedup Factor (GPU/CPU)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plot_dir, 'gpu_speedup_factor.png'))
            plt.close()
    except Exception as e:
        print(f"Error processing CPU vs GPU comparison: {e}")

# Process mixed precision results if available
if "mixed_precision" in results:
    try:
        mp_df = results["mixed_precision"]
        
        # Check if we have regular precision results for the same batch sizes
        if "batch_comparison" in results:
            reg_df = results["batch_comparison"]
            
            # Get column names
            mp_latency_col = get_column(mp_df, latency_names)
            mp_throughput_col = get_column(mp_df, throughput_names)
            mp_batch_size_col = get_column(mp_df, batch_size_names)
            
            reg_latency_col = get_column(reg_df, latency_names)
            reg_throughput_col = get_column(reg_df, throughput_names)
            reg_batch_size_col = get_column(reg_df, batch_size_names)
            
            print(f"\nMixed precision comparison columns:")
            print(f"  Mixed: Latency={mp_latency_col}, Throughput={mp_throughput_col}, Batch size={mp_batch_size_col}")
            print(f"  Regular: Latency={reg_latency_col}, Throughput={reg_throughput_col}, Batch size={reg_batch_size_col}")
            
            # Find common batch sizes
            common_batch_sizes = set(mp_df[mp_batch_size_col]).intersection(set(reg_df[reg_batch_size_col]))
            
            if common_batch_sizes:
                # Create a precision comparison dataframe
                precision_comparison = []
                for bs in sorted(common_batch_sizes):
                    mp_row = mp_df[mp_df[mp_batch_size_col] == bs].iloc[0]
                    reg_row = reg_df[reg_df[reg_batch_size_col] == bs].iloc[0]
                    
                    precision_comparison.append({
                        'batch_size': bs,
                        'mixed_precision_latency_ms': mp_row[mp_latency_col],
                        'regular_precision_latency_ms': reg_row[reg_latency_col],
                        'mixed_precision_throughput': mp_row[mp_throughput_col],
                        'regular_precision_throughput': reg_row[reg_throughput_col],
                        'throughput_improvement': mp_row[mp_throughput_col] / reg_row[reg_throughput_col],
                        'latency_reduction': 1 - (mp_row[mp_latency_col] / reg_row[reg_latency_col])
                    })
                
                # Convert to DataFrame
                precision_df = pd.DataFrame(precision_comparison)
                
                # Save to CSV
                precision_csv = os.path.join(output_dir, "precision_comparison.csv")
                precision_df.to_csv(precision_csv, index=False)
                print(f"Saved precision comparison to {precision_csv}")
                
                # Create precision improvement visualization
                plt.figure(figsize=(10, 6))
                sns.lineplot(x='batch_size', y='throughput_improvement', data=precision_df, marker='o')
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No improvement')
                plt.title('Mixed Precision Throughput Improvement vs. Batch Size')
                plt.xlabel('Batch Size')
                plt.ylabel('Throughput Improvement Factor')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plot_dir, 'mixed_precision_improvement.png'))
                plt.close()
    except Exception as e:
        print(f"Error processing mixed precision comparison: {e}")

print(f"\nAll analyses completed. Results saved to {output_dir}") 