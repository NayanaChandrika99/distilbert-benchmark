import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import gc
import psutil
from tqdm import tqdm

class PerformanceDiagnostics:
    def __init__(self, model, loss_fn, optimizer, device=None):
        """Set up the diagnostics tool with a model and training components.
        
        Args:
            model: PyTorch model to diagnose
            loss_fn: Loss function 
            optimizer: Optimizer
            device: Device to run on (auto-detects if None)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Figure out which device to use if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Where we'll store all our results
        self.results = []
    
    def _get_memory_usage(self):
        """Grab current memory usage for CPU and GPU."""
        # RAM usage
        cpu_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
        
        # GPU memory if we're using it
        gpu_memory = 0
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            
        return cpu_memory, gpu_memory
    
    def profile_batch_size(self, dataloader, batch_size, num_iterations=100):
        """Run profiling for a specific batch size.
        
        Args:
            dataloader: DataLoader to get data from
            batch_size: Batch size to test
            num_iterations: Number of iterations to run
            
        Returns:
            Dict with performance metrics
        """
        # Create dataloader with the batch size we want to test
        new_dataloader = DataLoader(
            dataloader.dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
        
        # Stuff we're measuring
        times = {
            'data_loading': [],
            'forward_pass': [],
            'backward_pass': [],
            'optimizer_step': [],
            'total_iteration': []
        }
        
        # Track memory usage
        initial_cpu_mem, initial_gpu_mem = self._get_memory_usage()
        peak_cpu_mem, peak_gpu_mem = initial_cpu_mem, initial_gpu_mem
        
        # Do a few warmup iterations first
        self.model.train()
        data_iter = iter(new_dataloader)
        for _ in range(min(5, len(new_dataloader))):
            try:
                inputs, targets = next(data_iter)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except StopIteration:
                data_iter = iter(new_dataloader)
                
        # Clean slate for actual testing
        self.optimizer.zero_grad()
        
        # Clear GPU cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Here's where the real testing happens
        self.model.train()
        data_iter = iter(new_dataloader)
        samples_processed = 0
        
        for i in range(num_iterations):
            try:
                # Time how long data loading takes
                start_time = time.time()
                inputs, targets = next(data_iter)
                data_load_time = time.time() - start_time
                times['data_loading'].append(data_load_time)
                
                # Move to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Time the forward pass
                start_time = time.time()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                forward_time = time.time() - start_time
                times['forward_pass'].append(forward_time)
                
                # Time the backward pass
                start_time = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                backward_time = time.time() - start_time
                times['backward_pass'].append(backward_time)
                
                # Time optimizer step
                start_time = time.time()
                self.optimizer.step()
                optim_time = time.time() - start_time
                times['optimizer_step'].append(optim_time)
                
                # Add up the total time
                times['total_iteration'].append(
                    data_load_time + forward_time + backward_time + optim_time
                )
                
                # Check memory usage
                cpu_mem, gpu_mem = self._get_memory_usage()
                peak_cpu_mem = max(peak_cpu_mem, cpu_mem)
                peak_gpu_mem = max(peak_gpu_mem, gpu_mem)
                
                # Keep track of how many samples we've processed
                samples_processed += inputs.size(0)
                
            except StopIteration:
                data_iter = iter(new_dataloader)
                
            # Clean up memory every now and then
            if i % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate all our metrics
        avg_times = {k: np.mean(v) for k, v in times.items()}
        throughput = samples_processed / sum(times['total_iteration'])
        
        # Gather everything into a nice results dict
        result = {
            'batch_size': batch_size,
            'device': self.device.type,
            'throughput': throughput,  # samples/second
            'avg_iteration_time': avg_times['total_iteration'],
            'avg_data_loading_time': avg_times['data_loading'],
            'avg_forward_time': avg_times['forward_pass'],
            'avg_backward_time': avg_times['backward_pass'],
            'avg_optimizer_time': avg_times['optimizer_step'],
            'memory_used_cpu': peak_cpu_mem - initial_cpu_mem,
            'memory_used_gpu': peak_gpu_mem - initial_gpu_mem,
            'peak_cpu_memory': peak_cpu_mem,
            'peak_gpu_memory': peak_gpu_mem,
        }
        
        self.results.append(result)
        return result
        
    def run_diagnostics(self, dataloader, batch_sizes, num_iterations=100, devices=None):
        """Run tests across multiple batch sizes and devices.
        
        Args:
            dataloader: Base dataloader with the dataset
            batch_sizes: List of batch sizes to test
            num_iterations: How many iterations per batch size
            devices: List of devices to test on (if None, uses current device)
        """
        original_device = self.device
        
        # Stick with current device if none specified
        if devices is None:
            devices = [self.device]
            
        for device in devices:
            print(f"Running diagnostics on {device}...")
            self.device = device
            self.model.to(device)
            
            for batch_size in tqdm(batch_sizes, desc="Testing batch sizes"):
                print(f"Testing batch size: {batch_size}")
                self.profile_batch_size(dataloader, batch_size, num_iterations)
                
            # Free up memory
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        # Put things back how we found them
        self.device = original_device
        self.model.to(original_device)
        
        return self.get_results()
    
    def get_results(self):
        """Get all results as a nice DataFrame."""
        return pd.DataFrame(self.results)
    
    def plot_throughput_vs_batch_size(self):
        """Plot how throughput changes with batch size for each device."""
        df = self.get_results()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot a line for each device
        for device in df['device'].unique():
            device_df = df[df['device'] == device]
            ax.plot(
                device_df['batch_size'], 
                device_df['throughput'], 
                'o-', 
                label=f'{device}'
            )
            
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (samples/second)')
        ax.set_title('Throughput vs Batch Size')
        ax.legend()
        ax.grid(True)
        
        # Use log scale if we have a wide range of batch sizes
        if max(df['batch_size']) / min(df['batch_size']) > 10:
            ax.set_xscale('log', base=2)
            
        return fig
    
    def plot_time_breakdown(self):
        """Create a stacked bar chart showing where time is being spent."""
        df = self.get_results()
        
        # Reshape data for plotting
        plot_data = []
        for _, row in df.iterrows():
            plot_data.append({
                'batch_size': row['batch_size'],
                'device': row['device'],
                'component': 'Data Loading',
                'time': row['avg_data_loading_time']
            })
            plot_data.append({
                'batch_size': row['batch_size'],
                'device': row['device'],
                'component': 'Forward Pass',
                'time': row['avg_forward_time']
            })
            plot_data.append({
                'batch_size': row['batch_size'],
                'device': row['device'],
                'component': 'Backward Pass',
                'time': row['avg_backward_time']
            })
            plot_data.append({
                'batch_size': row['batch_size'],
                'device': row['device'],
                'component': 'Optimizer Step',
                'time': row['avg_optimizer_time']
            })
            
        plot_df = pd.DataFrame(plot_data)
        
        # Create a figure for each device
        devices = df['device'].unique()
        fig, axes = plt.subplots(1, len(devices), figsize=(15, 6), sharey=True)
        if len(devices) == 1:
            axes = [axes]
            
        for i, device in enumerate(devices):
            device_df = plot_df[plot_df['device'] == device]
            pivot_df = device_df.pivot(index='batch_size', columns='component', values='time')
            
            # Make the stacked bar chart
            pivot_df.plot(kind='bar', stacked=True, ax=axes[i])
            axes[i].set_title(f'Time Breakdown ({device})')
            axes[i].set_xlabel('Batch Size')
            axes[i].set_ylabel('Time (seconds)')
            
        plt.tight_layout()
        return fig
    
    def plot_memory_usage(self):
        """Plot memory usage across batch sizes for each device."""
        df = self.get_results()
        
        # Only plot memory metrics that make sense for each device
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CPU memory plot
        ax = axes[0]
        for device in df['device'].unique():
            device_df = df[df['device'] == device]
            ax.plot(
                device_df['batch_size'],
                device_df['peak_cpu_memory'],
                'o-',
                label=f'{device}'
            )
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('CPU Memory Usage')
        ax.grid(True)
        ax.legend()
        
        # GPU memory plot - only for devices that have GPU metrics
        ax = axes[1]
        for device in df['device'].unique():
            device_df = df[df['device'] == device]
            if device == 'cuda' and 'peak_gpu_memory' in device_df.columns:
                ax.plot(
                    device_df['batch_size'],
                    device_df['peak_gpu_memory'],
                    'o-',
                    label=f'{device}'
                )
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_bottleneck_report(self):
        """Create a detailed report analyzing performance bottlenecks."""
        df = self.get_results()
        
        report = "# Performance Bottleneck Analysis\n\n"
        
        # Add timestamp
        from datetime import datetime
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall summary
        report += "## Summary\n\n"
        
        # Find best throughput batch size
        best_throughput_row = df.loc[df['throughput'].idxmax()]
        report += f"- Best throughput: {best_throughput_row['throughput']:.2f} samples/sec with batch size {best_throughput_row['batch_size']} on {best_throughput_row['device']}\n"
        
        # Device comparison if we have multiple devices
        if len(df['device'].unique()) > 1:
            report += "\n### Device Comparison\n\n"
            for device in df['device'].unique():
                device_df = df[df['device'] == device]
                best_device_row = device_df.loc[device_df['throughput'].idxmax()]
                report += f"- {device}: Best throughput {best_device_row['throughput']:.2f} samples/sec at batch size {best_device_row['batch_size']}\n"
            
            # GPU speedup if we have both CPU and GPU results
            if 'cpu' in df['device'].values and 'cuda' in df['device'].values:
                cpu_best = df[df['device'] == 'cpu']['throughput'].max()
                gpu_best = df[df['device'] == 'cuda']['throughput'].max()
                speedup = gpu_best / cpu_best if cpu_best > 0 else float('inf')
                report += f"\nGPU speedup over CPU: {speedup:.2f}x\n"
        
        # Batch size analysis
        report += "\n## Batch Size Analysis\n\n"
        
        # Look for patterns in the data
        for device in df['device'].unique():
            device_df = df[df['device'] == device].sort_values('batch_size')
            
            report += f"### {device.upper()} Performance\n\n"
            
            # Skip if we don't have enough batch sizes to analyze
            if len(device_df) < 2:
                report += "Not enough batch sizes to analyze patterns.\n\n"
                continue
                
            # Check for throughput pattern (small -> large batch sizes)
            throughputs = device_df['throughput'].values
            batch_sizes = device_df['batch_size'].values
            
            # Look for batch sizes where throughput drops
            peak_idx = np.argmax(throughputs)
            peak_batch = batch_sizes[peak_idx]
            
            report += f"Peak throughput at batch size {peak_batch}.\n\n"
            
            # Check for small batch size inefficiency
            if peak_idx > 0:
                report += "#### Small Batch Size Performance\n\n"
                small_inefficiency = 1 - (throughputs[0] / throughputs[peak_idx])
                report += f"Small batch inefficiency: {small_inefficiency:.1%} lower throughput at batch size {batch_sizes[0]} compared to optimal.\n\n"
                
                # Look at what's taking time with small batches
                small_batch_row = device_df.iloc[0]
                total_time = small_batch_row['avg_iteration_time']
                data_pct = small_batch_row['avg_data_loading_time'] / total_time * 100
                forward_pct = small_batch_row['avg_forward_time'] / total_time * 100
                backward_pct = small_batch_row['avg_backward_time'] / total_time * 100
                optim_pct = small_batch_row['avg_optimizer_time'] / total_time * 100
                
                report += f"Time breakdown at batch size {batch_sizes[0]}:\n"
                report += f"- Data loading: {data_pct:.1f}%\n"
                report += f"- Forward pass: {forward_pct:.1f}%\n"
                report += f"- Backward pass: {backward_pct:.1f}%\n"
                report += f"- Optimizer step: {optim_pct:.1f}%\n\n"
                
                # Diagnose the issue
                if data_pct > 30:
                    report += "Diagnosis: Data loading is taking significant time with small batches. "
                    report += "Consider increasing the number of workers or using GPU pinned memory.\n\n"
                elif forward_pct + backward_pct > 80:
                    report += "Diagnosis: Model compute time dominates with small batches, but GPU utilization is likely low. "
                    report += "Small batches don't fully utilize GPU parallel processing.\n\n"
            
            # Check for large batch size issues
            if peak_idx < len(batch_sizes) - 1:
                report += "#### Large Batch Size Performance\n\n"
                large_idx = len(batch_sizes) - 1
                large_inefficiency = 1 - (throughputs[large_idx] / throughputs[peak_idx])
                report += f"Large batch inefficiency: {large_inefficiency:.1%} lower throughput at batch size {batch_sizes[large_idx]} compared to optimal.\n\n"
                
                # Look at what's changed with large batches
                large_batch_row = device_df.iloc[large_idx]
                opt_batch_row = device_df.iloc[peak_idx]
                
                # Memory usage comparison
                if device == 'cuda':
                    opt_mem = opt_batch_row['peak_gpu_memory']
                    large_mem = large_batch_row['peak_gpu_memory']
                    mem_increase = (large_mem - opt_mem) / opt_mem * 100
                    report += f"Memory usage increased {mem_increase:.1f}% from optimal to largest batch size.\n\n"
                    
                    if mem_increase > 50:
                        report += "Diagnosis: High memory pressure is likely causing GPU throttling or memory swapping.\n\n"
                
                # Timing comparison
                for component in ['forward', 'backward', 'optimizer']:
                    opt_time = opt_batch_row[f'avg_{component}_time']
                    large_time = large_batch_row[f'avg_{component}_time']
                    time_ratio = large_time / opt_time if opt_time > 0 else float('inf')
                    
                    ideal_ratio = batch_sizes[large_idx] / batch_sizes[peak_idx]
                    efficiency = ideal_ratio / time_ratio if time_ratio > 0 else 0
                    
                    report += f"{component.capitalize()} pass scaling efficiency: {efficiency:.1%}\n"
                    
                report += "\n"
                
                # Overall diagnosis for large batches
                if large_inefficiency > 0.2:
                    if device == 'cuda':
                        report += "Diagnosis for large batches: "
                        if mem_increase > 50:
                            report += "Memory constraints are limiting performance. "
                            report += "Consider gradient accumulation to maintain effective batch size while reducing memory usage.\n\n"
                        else:
                            report += "Computational efficiency is dropping with larger batches. "
                            report += "This could be due to reduced cache efficiency or other architectural factors.\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        # General batch size recommendation
        report += f"- Recommended batch size for highest throughput: {best_throughput_row['batch_size']} on {best_throughput_row['device']}\n"
        
        # Add device-specific recommendations
        for device in df['device'].unique():
            device_df = df[df['device'] == device]
            best_device_batch = device_df.loc[device_df['throughput'].idxmax()]['batch_size']
            
            report += f"\n### {device.upper()} Optimization Tips\n\n"
            
            if device == 'cpu':
                report += "- Try setting **OMP_NUM_THREADS** and **MKL_NUM_THREADS** to optimize CPU threading\n"
                report += "- Consider using PyTorch's JIT compilation for CPU speedup\n"
                report += f"- Use batch size {best_device_batch} for best CPU throughput\n"
            elif device == 'cuda':
                report += "- Enable mixed precision training for GPU throughput boost\n"
                report += "- Increase dataloader workers (possibly to CPU core count) to avoid GPU starvation\n"
                report += f"- Use batch size {best_device_batch} for optimal GPU performance\n"
                
                # Only suggest gradient accumulation if we're memory bound with large batches
                if best_device_batch < max(device_df['batch_size']):
                    report += "- Consider gradient accumulation if you need a larger effective batch size\n"
                
        return report


def example_usage():
    """Example showing how to use the diagnostics tool."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )
    
    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize and run diagnostics
    profiler = PerformanceDiagnostics(model, loss_fn, optimizer)
    batch_sizes = [8, 16, 32, 64, 128]
    profiler.run_diagnostics(dataloader, batch_sizes, num_iterations=10)
    
    # Generate visualizations
    profiler.plot_throughput_vs_batch_size()
    profiler.plot_time_breakdown()
    profiler.plot_memory_usage()
    
    # Generate report
    report = profiler.generate_bottleneck_report()
    print(report)
    
    
if __name__ == "__main__":
    example_usage() 