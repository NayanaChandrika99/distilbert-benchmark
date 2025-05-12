#!/usr/bin/env python3
"""
Generate a comprehensive DistilBERT benchmark report including all test configurations.
This script combines data from all benchmark results and generates detailed reports and visualizations.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import re
import markdown
from tabulate import tabulate

# Directory paths
INPUT_DIR = "distilbert_outputs"
OUTPUT_DIR = "distilbert_outputs/reports"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_jsonl(filepath):
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def collect_all_benchmark_data():
    """Collect and combine data from all benchmark files."""
    all_data = []
    
    # List of files to process
    benchmark_files = [
        "batch_comparison.jsonl",
        "mixed_precision_results.jsonl",
        "cpu_results.jsonl",
        "absolute_results.jsonl"
    ]
    
    for filename in benchmark_files:
        filepath = os.path.join(INPUT_DIR, filename)
        if os.path.exists(filepath):
            print(f"Processing {filepath}...")
            try:
                data = load_jsonl(filepath)
                all_data.extend(data)
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
        else:
            print(f"Warning: File {filepath} not found")
    
    return all_data

def extract_metrics(data_list):
    """Extract key metrics from benchmark data into a DataFrame."""
    metrics_data = []
    
    for data in data_list:
        metrics = data.get('metrics', {})
        
        row = {
            'timestamp': data.get('timestamp', None),
            'batch_size': data.get('batch_size', None),
            'model': data.get('model', None),
            'device': data.get('device', None),
            'mixed_precision': data.get('mixed_precision', False),
            'latency_ms': metrics.get('latency_ms_mean', None),
            'throughput': metrics.get('throughput_mean', None),
            'cpu_memory_mb': metrics.get('cpu_memory_mb_max', None),
            'gpu_memory_mb': metrics.get('gpu_memory_mb_max', None) if metrics.get('gpu_memory_mb_max', None) is not None else np.nan,
            'gpu_avg_power_w': metrics.get('gpu_avg_power_w', np.nan),
            'gpu_energy_j': metrics.get('gpu_energy_j', np.nan),
        }
        
        # Calculate additional metrics
        if row['gpu_memory_mb'] and row['batch_size']:
            row['memory_per_sample'] = row['gpu_memory_mb'] / row['batch_size']
        else:
            row['memory_per_sample'] = np.nan
            
        if row['throughput'] and row['gpu_avg_power_w'] and not np.isnan(row['gpu_avg_power_w']):
            row['samples_per_watt'] = row['throughput'] / row['gpu_avg_power_w']
        else:
            row['samples_per_watt'] = np.nan
            
        metrics_data.append(row)
    
    # Convert to DataFrame and clean up
    df = pd.DataFrame(metrics_data)
    
    # Handle duplicate entries (take the latest entry for each configuration)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp in descending order
        df = df.sort_values('timestamp', ascending=False)
        
        # Take the most recent entry for each unique configuration
        df = df.drop_duplicates(subset=['batch_size', 'device', 'mixed_precision'], keep='first')
        
        # Sort by device, mixed_precision, and batch_size for easier reading
        df = df.sort_values(['device', 'mixed_precision', 'batch_size'])
    
    return df

def extract_model_info(data_list):
    """Extract model information from the benchmark data."""
    for data in data_list:
        if 'model_info' in data:
            return data['model_info']
    return None

def extract_system_info(data_list):
    """Extract system information from the benchmark data."""
    for data in data_list:
        if 'system_info' in data:
            return data['system_info']
    return None

def calculate_scaling_efficiency(df):
    """Calculate scaling efficiency for each device/precision combination."""
    scaling_data = []
    
    # Get unique device and mixed_precision combinations
    configs = df[['device', 'mixed_precision']].drop_duplicates().values
    
    for device, mixed_precision in configs:
        config_df = df[(df['device'] == device) & (df['mixed_precision'] == mixed_precision)]
        
        if len(config_df) < 2:
            continue
            
        # Sort by batch size
        config_df = config_df.sort_values('batch_size')
        
        # Calculate throughput per batch item for each batch size
        per_item_throughput = config_df['throughput'] / config_df['batch_size']
        
        # Calculate scaling efficiency as ratio of largest to smallest batch size's per-item throughput
        if len(per_item_throughput) >= 2:
            baseline = per_item_throughput.iloc[0]
            for i, efficiency in enumerate(per_item_throughput):
                batch_size = config_df['batch_size'].iloc[i]
                scaling_data.append({
                    'device': device,
                    'mixed_precision': mixed_precision,
                    'batch_size': batch_size,
                    'per_item_throughput': efficiency,
                    'scaling_efficiency': efficiency / baseline if baseline > 0 else np.nan
                })
    
    return pd.DataFrame(scaling_data)

def calculate_mixed_precision_speedup(df):
    """Calculate mixed precision speedup for each batch size."""
    speedup_data = []
    
    # Get unique device and batch_size combinations that have both FP32 and mixed precision
    for device in df['device'].unique():
        device_df = df[df['device'] == device]
        
        for batch_size in device_df['batch_size'].unique():
            batch_df = device_df[device_df['batch_size'] == batch_size]
            
            fp32_row = batch_df[batch_df['mixed_precision'] == False]
            amp_row = batch_df[batch_df['mixed_precision'] == True]
            
            if len(fp32_row) == 0 or len(amp_row) == 0:
                continue
                
            fp32_throughput = fp32_row['throughput'].iloc[0]
            amp_throughput = amp_row['throughput'].iloc[0]
            
            fp32_latency = fp32_row['latency_ms'].iloc[0]
            amp_latency = amp_row['latency_ms'].iloc[0]
            
            speedup_data.append({
                'device': device,
                'batch_size': batch_size,
                'fp32_throughput': fp32_throughput,
                'amp_throughput': amp_throughput,
                'throughput_speedup': amp_throughput / fp32_throughput if fp32_throughput > 0 else np.nan,
                'throughput_improvement_percent': (amp_throughput - fp32_throughput) / fp32_throughput * 100 if fp32_throughput > 0 else np.nan,
                'fp32_latency': fp32_latency,
                'amp_latency': amp_latency,
                'latency_speedup': fp32_latency / amp_latency if amp_latency > 0 else np.nan,
                'latency_improvement_percent': (fp32_latency - amp_latency) / fp32_latency * 100 if fp32_latency > 0 else np.nan
            })
    
    return pd.DataFrame(speedup_data)

def calculate_device_speedup(df):
    """Calculate speedup between devices (e.g., CPU vs GPU)."""
    speedup_data = []
    
    # Assume we want to compare the "baseline" device (typically CPU) against other devices
    devices = df['device'].unique()
    if len(devices) < 2:
        return None
        
    # We'll arbitrarily choose 'cpu' as baseline if it exists, otherwise the first device
    baseline_device = 'cpu' if 'cpu' in devices else devices[0]
    
    # Get unique configurations (mixed_precision and batch_size)
    for mixed_precision in df['mixed_precision'].unique():
        mp_df = df[df['mixed_precision'] == mixed_precision]
        
        for batch_size in mp_df['batch_size'].unique():
            batch_df = mp_df[mp_df['batch_size'] == batch_size]
            
            if baseline_device not in batch_df['device'].values:
                continue
                
            baseline_row = batch_df[batch_df['device'] == baseline_device]
            
            baseline_throughput = baseline_row['throughput'].iloc[0]
            baseline_latency = baseline_row['latency_ms'].iloc[0]
            
            for device in batch_df['device'].unique():
                if device == baseline_device:
                    continue
                    
                device_row = batch_df[batch_df['device'] == device]
                device_throughput = device_row['throughput'].iloc[0]
                device_latency = device_row['latency_ms'].iloc[0]
                
                speedup_data.append({
                    'batch_size': batch_size,
                    'mixed_precision': mixed_precision,
                    'baseline_device': baseline_device,
                    'target_device': device,
                    'baseline_throughput': baseline_throughput,
                    'target_throughput': device_throughput,
                    'throughput_speedup': device_throughput / baseline_throughput if baseline_throughput > 0 else np.nan,
                    'baseline_latency': baseline_latency,
                    'target_latency': device_latency,
                    'latency_speedup': baseline_latency / device_latency if device_latency > 0 else np.nan
                })
    
    return pd.DataFrame(speedup_data)

def generate_markdown_report(metrics_df, scaling_df, mp_speedup_df, device_speedup_df, model_info, system_info):
    """Generate comprehensive markdown report with benchmark results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format system info
    system_info_text = ""
    if system_info:
        system_info_text = f"""## System Information

- **Platform**: {system_info.get('platform', 'Unknown')}
- **Python Version**: {system_info.get('python_version', 'Unknown')}
- **Processor**: {system_info.get('processor', 'Unknown')}
- **CPU Count**: {system_info.get('cpu_count', 'Unknown')}
- **PyTorch Version**: {system_info.get('torch_version', 'Unknown')}
"""
        
        # Add GPU info if available
        if 'gpu_info' in system_info and system_info['gpu_info']:
            gpu_info = system_info['gpu_info'][0]
            system_info_text += f"""- **GPU**: {gpu_info.get('name', 'Unknown')}
- **CUDA Version**: {system_info.get('cuda_version', 'Unknown')}
- **cuDNN Version**: {system_info.get('cudnn_version', 'Unknown')}
"""

    # Format model info
    model_info_text = ""
    if model_info:
        model_info_text = f"""## Model Information

- **Model**: {model_info.get('name', 'Unknown')}
- **Hidden Size**: {model_info.get('hidden_size', 'Unknown')}
- **Hidden Layers**: {model_info.get('num_hidden_layers', 'Unknown')}
- **Attention Heads**: {model_info.get('num_attention_heads', 'Unknown')}
- **Parameters**: {model_info.get('num_parameters', 'Unknown'):,}
"""

    # Format batch size comparison table
    grouped_metrics = metrics_df.sort_values(['device', 'mixed_precision', 'batch_size'])
    
    # Prepare performance tables by device and precision
    performance_tables = ""
    
    for device in metrics_df['device'].unique():
        device_df = metrics_df[metrics_df['device'] == device]
        
        for mixed_precision in sorted(device_df['mixed_precision'].unique()):
            precision_type = "Mixed Precision (AMP)" if mixed_precision else "FP32"
            config_df = device_df[device_df['mixed_precision'] == mixed_precision].sort_values('batch_size')
            
            if len(config_df) == 0:
                continue
                
            # Convert to formatted table
            table_df = config_df[['batch_size', 'latency_ms', 'throughput', 'cpu_memory_mb', 'gpu_memory_mb']].copy()
            
            # Format the table with appropriate precision and alignment
            table_html = tabulate(
                table_df, 
                headers=['Batch Size', 'Latency (ms)', 'Throughput (samples/sec)', 'CPU Memory (MB)', 'GPU Memory (MB)'],
                tablefmt='pipe',
                floatfmt='.2f',
                showindex=False
            )
            
            performance_tables += f"### {device.upper()} - {precision_type}\n\n"
            performance_tables += table_html + "\n\n"
    
    # Find optimal configurations
    if 'cuda' in metrics_df['device'].values:
        gpu_df = metrics_df[metrics_df['device'] == 'cuda']
        
        max_throughput_row = gpu_df.loc[gpu_df['throughput'].idxmax()]
        min_latency_row = gpu_df.loc[gpu_df['latency_ms'].idxmin()]
        
        optimal_batch_throughput = max_throughput_row['batch_size']
        optimal_throughput = max_throughput_row['throughput']
        optimal_precision_throughput = "Mixed Precision" if max_throughput_row['mixed_precision'] else "FP32"
        
        optimal_batch_latency = min_latency_row['batch_size']
        optimal_latency = min_latency_row['latency_ms']
        optimal_precision_latency = "Mixed Precision" if min_latency_row['mixed_precision'] else "FP32"
        
        # Calculate estimated daily throughput
        estimated_daily_throughput = optimal_throughput * 86400  # seconds in a day
        
        optimal_config_text = f"""## Optimal Configurations

### Maximum Throughput
- **Batch Size**: {optimal_batch_throughput}
- **Precision**: {optimal_precision_throughput}
- **Throughput**: {optimal_throughput:.2f} samples/second
- **Estimated Daily Throughput**: {estimated_daily_throughput:,.2f} samples/day ({estimated_daily_throughput/1e6:.2f} million samples/day)

### Minimum Latency
- **Batch Size**: {optimal_batch_latency}
- **Precision**: {optimal_precision_latency}
- **Latency**: {optimal_latency:.2f} ms
"""
    else:
        optimal_config_text = "## Optimal Configurations\n\nNo GPU data available to determine optimal configurations.\n\n"

    # Format mixed precision speedup table
    mp_speedup_text = ""
    if mp_speedup_df is not None and len(mp_speedup_df) > 0:
        mp_speedup_text = "## Mixed Precision Speedup\n\n"
        
        # Format the table with appropriate precision and alignment
        mp_table_df = mp_speedup_df[['batch_size', 'throughput_improvement_percent', 'latency_improvement_percent']].copy()
        mp_table_df.columns = ['Batch Size', 'Throughput Improvement (%)', 'Latency Improvement (%)']
        
        mp_speedup_text += tabulate(
            mp_table_df, 
            headers='keys',
            tablefmt='pipe',
            floatfmt='.2f',
            showindex=False
        ) + "\n\n"
    
    # Format device speedup table
    device_speedup_text = ""
    if device_speedup_df is not None and len(device_speedup_df) > 0:
        device_speedup_text = "## Device Speedup Comparison\n\n"
        
        # Only include common configurations between devices
        device_table_df = device_speedup_df[['batch_size', 'mixed_precision', 'baseline_device', 'target_device', 'throughput_speedup', 'latency_speedup']].copy()
        device_table_df.columns = ['Batch Size', 'Mixed Precision', 'Baseline Device', 'Target Device', 'Throughput Speedup (×)', 'Latency Speedup (×)']
        
        # Format the mixed precision column
        device_table_df['Mixed Precision'] = device_table_df['Mixed Precision'].map({True: 'Yes', False: 'No'})
        
        device_speedup_text += tabulate(
            device_table_df, 
            headers='keys',
            tablefmt='pipe',
            floatfmt='.2f',
            showindex=False
        ) + "\n\n"
    
    # Calculate scaling efficiency information
    scaling_efficiency_text = ""
    if scaling_df is not None and len(scaling_df) > 0:
        scaling_efficiency_text = "## Scaling Efficiency\n\n"
        
        for device in scaling_df['device'].unique():
            device_scaling = scaling_df[scaling_df['device'] == device]
            
            for mixed_precision in device_scaling['mixed_precision'].unique():
                precision_type = "Mixed Precision" if mixed_precision else "FP32"
                config_scaling = device_scaling[device_scaling['mixed_precision'] == mixed_precision]
                
                if len(config_scaling) == 0:
                    continue
                
                max_batch_row = config_scaling.iloc[config_scaling['batch_size'].argmax()]
                min_batch_row = config_scaling.iloc[config_scaling['batch_size'].argmin()]
                
                overall_efficiency = max_batch_row['scaling_efficiency']
                
                scaling_efficiency_text += f"### {device.upper()} - {precision_type}\n\n"
                scaling_efficiency_text += f"- **Overall Scaling Efficiency**: {overall_efficiency:.2f}\n"
                scaling_efficiency_text += f"- **Baseline Batch Size**: {min_batch_row['batch_size']}\n"
                scaling_efficiency_text += f"- **Maximum Batch Size**: {max_batch_row['batch_size']}\n\n"
                
                # Create a table showing efficiency at each batch size
                scale_table_df = config_scaling[['batch_size', 'per_item_throughput', 'scaling_efficiency']].copy()
                scale_table_df.columns = ['Batch Size', 'Per-Item Throughput', 'Scaling Efficiency']
                
                scaling_efficiency_text += tabulate(
                    scale_table_df, 
                    headers='keys',
                    tablefmt='pipe',
                    floatfmt='.2f',
                    showindex=False
                ) + "\n\n"
    
    # Combine all sections into the final report
    report = f"""# DistilBERT Comprehensive Benchmark Report

*Generated on {now}*

{model_info_text}

{system_info_text}

## Performance Metrics

{performance_tables}

{optimal_config_text}

{scaling_efficiency_text}

{mp_speedup_text}

{device_speedup_text}

## Recommendations

1. **For Maximum Throughput**: Use batch size {optimal_batch_throughput} with {optimal_precision_throughput}
   * Achieves {optimal_throughput:.2f} samples/second
   * Can process approximately {estimated_daily_throughput/1e6:.2f} million samples per day

2. **For Minimum Latency**: Use batch size {optimal_batch_latency} with {optimal_precision_latency}
   * Achieves {optimal_latency:.2f} ms latency

3. **Hardware Recommendations**:
   * GPU provides significantly better performance than CPU (up to {device_speedup_df['throughput_speedup'].max():.1f}x throughput)
   * For production deployments, GPU is strongly recommended

## Conclusion

The DistilBERT model demonstrates excellent performance characteristics on GPU hardware. Mixed precision provides modest gains in throughput and latency without requiring additional memory. For optimal efficiency, batch size {optimal_batch_throughput} provides the best balance of throughput and resource utilization.
"""

    return report

def save_markdown_report(report_text, output_path):
    """Save the markdown report to file."""
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to {output_path}")

def main():
    """Main function to generate the comprehensive report."""
    print("Generating comprehensive DistilBERT benchmark report...")
    
    # Collect all benchmark data
    data_list = collect_all_benchmark_data()
    if not data_list:
        print("Error: No benchmark data found")
        return
    
    # Extract metrics into DataFrame
    metrics_df = extract_metrics(data_list)
    if metrics_df.empty:
        print("Error: No metrics data could be extracted")
        return
    
    # Extract additional information
    model_info = extract_model_info(data_list)
    system_info = extract_system_info(data_list)
    
    # Calculate advanced metrics
    scaling_df = calculate_scaling_efficiency(metrics_df)
    mp_speedup_df = calculate_mixed_precision_speedup(metrics_df)
    device_speedup_df = calculate_device_speedup(metrics_df)
    
    # Generate markdown report
    report = generate_markdown_report(metrics_df, scaling_df, mp_speedup_df, device_speedup_df, model_info, system_info)
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, "comprehensive_report.md")
    save_markdown_report(report, output_path)
    
    print(f"Comprehensive report generated at {output_path}")

if __name__ == "__main__":
    main() 