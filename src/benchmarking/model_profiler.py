import torch
import matplotlib.pyplot as plt
from perf_diagnostics import PerformanceDiagnostics

def run_kaggle_analysis(model, train_loader, batch_sizes=None):
    """
    Run performance diagnostics to figure out what's going on with those weird batch size patterns.
    
    Args:
        model: Your PyTorch model
        train_loader: DataLoader with training data
        batch_sizes: List of batch sizes to test (defaults to [8, 16, 32, 64, 128, 256])
    """
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128, 256]
    
    # We'll need these for profiling - nothing fancy, just basic stuff
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Let's see what devices we can use
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    
    # Fire up the profiler
    print("Setting up performance diagnostics...")
    profiler = PerformanceDiagnostics(model, loss_fn, optimizer)
    
    print(f"Running tests on batch sizes: {batch_sizes}")
    profiler.run_diagnostics(train_loader, batch_sizes, num_iterations=30, devices=devices)
    
    # Let's make some pretty charts
    print("Creating charts and reports...")
    
    # Throughput chart - the most important one
    throughput_fig = profiler.plot_throughput_vs_batch_size()
    throughput_fig.savefig('throughput_vs_batch_size.png')
    print("✓ Saved throughput chart")
    
    # Time breakdown - helps spot bottlenecks
    time_fig = profiler.plot_time_breakdown()
    time_fig.savefig('time_breakdown.png')
    print("✓ Saved time breakdown chart")
    
    # Memory usage - crucial for understanding GPU limits
    memory_fig = profiler.plot_memory_usage()
    memory_fig.savefig('memory_usage.png')
    print("✓ Saved memory usage chart")
    
    # Let's get a detailed report too
    report = profiler.generate_bottleneck_report()
    with open('performance_report.md', 'w') as f:
        f.write(report)
    print("✓ Created performance report")
    
    print("\nAll done! Check out the charts and report to see what's happening with your model.")
    
    # Return the profiler object so you can dig deeper if needed
    return profiler


def analyze_gpu_utilization():
    """
    Dig into GPU utilization to figure out why there's that weird batch size pattern.
    This helps explain why batch=8 is slow, batch=32 is fast, and batch=128 slows down again.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        # How many GPUs we got?
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            print("No NVIDIA GPUs found. Are you sure you're on a GPU machine?")
            return
            
        print(f"Found {device_count} NVIDIA GPU{'s' if device_count > 1 else ''}")
        
        # Let's look at the first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # What are we working with?
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"GPU: {name.decode('utf-8')}")
        
        # Memory info - this matters a lot
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = meminfo.total / (1024**3)  # GB
        print(f"Total memory: {total_memory:.2f} GB")
        
        # What architecture is this?
        compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        print(f"Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        
        print("\nHere's what's probably limiting your GPU:")
        print("- Small batches (like 8): Probably kernel launch overhead & low utilization")
        print("- Big batches (like 128): Probably hitting memory limits")
        print("- Medium batches (like 32): Usually the sweet spot!")
        
        print("\nTry these commands for more insight:")
        print("1. nvidia-smi dmon -s pucvmet -i 0 -d 1   # real-time monitoring")
        print("2. Nsight Systems if you want timeline visualization")
        print("3. Nsight Compute for kernel-level analysis (but it's overkill for most)")
    except ImportError:
        print("You need pynvml to check GPU stats - try 'pip install nvidia-ml-py'")
    except Exception as e:
        print(f"Oops, something went wrong: {str(e)}")


if __name__ == "__main__":
    print("Import this module in your Kaggle notebook to use these tools.")
    print("Quick example:")
    print("```python")
    print("from model_profiler import run_kaggle_analysis, analyze_gpu_utilization")
    print("")
    print("# Run batch size analysis")
    print("diagnostics = run_kaggle_analysis(model, train_loader, [8, 16, 32, 64, 128, 256])")
    print("")
    print("# Check GPU details")
    print("analyze_gpu_utilization()")
    print("```") 