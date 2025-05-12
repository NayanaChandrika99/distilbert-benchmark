# DistilBERT Benchmark Suite

A comprehensive benchmarking toolkit for evaluating DistilBERT performance across CPUs and GPUs with detailed reporting capabilities.

## Overview

This toolkit provides an automated pipeline for benchmarking the DistilBERT language model on various hardware configurations. It measures critical performance metrics such as latency, throughput, memory usage, and energy consumption across different batch sizes. The suite generates publication-ready reports and visualizations for easy analysis and comparison.

## Features

- **Comprehensive Benchmarking**: Measures latency, throughput, memory usage, and energy consumption
- **Device Support**: Tests on both CPU and GPU with automatic device detection
- **Batch Size Sweeps**: Evaluates performance across varying batch sizes
- **Visualization**: Generates insightful plots for all metrics
- **Reporting**: Creates HTML reports, PDF documents, and PowerPoint presentations
- **Kaggle Integration**: Primarily deployed on Kaggle with notebooks included
- **Dataset Support**: Uses GLUE/SST-2 for standardized testing

## Requirements

- Python 3.10+
- PyTorch 1.12+
- Transformers 4.25+
- CUDA-compatible GPU (optional for GPU testing)
- Additional dependencies listed in `environment.yml`

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/distilbert-benchmark.git
   cd distilbert-benchmark
   ```

2. Create and activate the conda environment:
```bash
   make setup
   conda activate distilbert-benchmark
   ```

3. Run a smoke test to verify your setup:
```bash
   make smoke
   ```

## Usage

### Basic Benchmarking

Run benchmarks on CPU:
```bash
make benchmark
```

Run benchmarks on GPU (if available):
```bash
make benchmark-gpu
```

Run both CPU and GPU benchmarks with comparison:
```bash
make benchmark-all
```

### Generating Reports

Create HTML reports from benchmark results:
```bash
make report
```

Create PDF reports (requires Pandoc):
```bash
make report-pdf
```

Create PowerPoint slide decks:
```bash
make report-pptx
```

Generate all report formats:
```bash
make report-full
```

### Quick Testing

Generate a quick report from minimal smoke test results:
```bash
make quick-report
```

### HPC Cluster Integration

Note: HPC-related files are stored in the `hpc_related_files` directory but not included in the GitHub repository as the project was primarily deployed on Kaggle.

Submit a job to SLURM:
```bash
make slurm
```

For CPU-only or GPU-specific SLURM jobs:
```bash
make slurm-cpu
make slurm-gpu
```

## Project Structure

The project has been organized for clarity:

- `src/` - Core implementation files
  - `analysis/` - Scripts for analyzing and visualizing benchmark results
  - `benchmarking/` - Benchmarking implementation and profiling
  - `visualization/` - Data visualization tools
  - `metrics/` - Performance metrics collection
- `tests/` - Test files to ensure code quality
- `config/` - Configuration files for the project
- `docs/` - Documentation files
- `examples/` - Example notebooks and usage demos
- `kaggle/` - Scripts and notebooks for running benchmarks on Kaggle
- `scripts/` - Utility scripts for running benchmarks
- `results/` - Directory for storing benchmark results

## Key Components

### Analysis Tools
See `src/analysis/README.md` for details on:
- Generating comparison plots
- Creating comprehensive reports
- Analyzing mixed precision results

### Kaggle Integration
See `kaggle/README.md` for details on:
- Running benchmarks on Kaggle
- Mixed precision benchmarking
- Downloading results

### Core Implementation
The core benchmarking code is in `src/benchmarking/` and includes:
- Model loading and configuration
- Metrics collection
- Benchmarking pipeline
- Data processing

## Results

Benchmark results are stored in the `results/` directory:
- `batch_comparison.jsonl` - Results from GPU batch size comparisons
- `cpu_results.jsonl` - Results from CPU benchmarks
- `mixed_precision_results.jsonl` - Results from mixed precision benchmarks
- `absolute_results.jsonl` - Reference benchmark results

Visualization and analysis:
- `results/analysis/batch_profile_charts/` - Detailed batch profiling charts
- `results/analysis/` - Analysis reports and artifacts

## Development

### Testing

Run all tests:
```bash
make test-all
```

Run analysis components tests:
```bash
make test-analysis
```

Run reporting components tests:
```bash
make test-reporting
```

### Code Quality

Set up pre-commit hooks:
```bash
make setup-pre-commit
```

Run linting and code quality checks:
```bash
make lint
```

### Artifact Integrity

Generate SHA256 manifest for project artifacts:
```bash
make manifest
```

Verify artifacts against manifest:
```bash
make verify-manifest
```

## Creating a Release

Generate a complete release package with reports, figures, and integrity verification:
```bash
make release
```

This creates a `release` directory with all deliverables and a SHA256 manifest for verification.

## Limitations and Known Issues

- PDF generation requires Pandoc to be installed
- GPU metrics collection requires NVIDIA GPUs with NVML support
- Energy measurements may not be available on all hardware platforms
- Some functionality requires root/admin access for hardware counters

## Future Work

- Support for additional language models (BERT, RoBERTa, etc.)
- Integration with MLPerf benchmarking suite
- Support for distributed and multi-GPU benchmarking
- Integration with cloud cost calculators
- More fine-grained energy profiling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{distilbert_benchmark,
  author = {Your Name},
  title = {DistilBERT Benchmark Suite},
  year = {2023},
  url = {https://github.com/yourusername/distilbert-benchmark}
}
```

## Acknowledgments

- Hugging Face for the Transformers library
- The GLUE benchmark creators for the dataset
- DistilBERT authors for the model architecture

# Model Performance Diagnostics

This tool helps diagnose deep learning model performance patterns across different batch sizes and hardware. It's designed to identify the causes of common performance patterns like:

1. Low throughput at small batch sizes (e.g., batch size 8)
2. Optimal throughput at medium batch sizes (e.g., batch size 32)
3. Declining throughput at large batch sizes (e.g., batch size 128)
4. CPU vs. GPU performance differences

## Installation

```bash
# Clone this repository
git clone <repository-url>
cd model-diagnostics

# Install dependencies
pip install torch matplotlib numpy pandas psutil tqdm
pip install nvidia-ml-py  # Optional: for GPU utilization analysis
```

## Usage in Kaggle

1. Download the key files to your Kaggle notebook:
   - `src/benchmarking/perf_diagnostics.py` - Performance diagnostics class
   - `src/benchmarking/model_profiler.py` - Kaggle integration utilities
   - `src/metrics/` - Performance metrics collection
   
   Or use the pre-packaged Kaggle notebook in the `kaggle/` directory:
   - `kaggle/run_distilbert_on_kaggle.py` - Pre-configured Kaggle benchmark
   - `kaggle/distilbert_kaggle_notebook.ipynb` - Notebook version

2. Import and use the diagnostic tools:

```python
# If using the standalone files
from model_profiler import run_kaggle_analysis, analyze_gpu_utilization

# Run comprehensive batch size performance analysis
diagnostics = run_kaggle_analysis(model, train_loader, [8, 16, 32, 64, 128, 256])

# Check GPU details and utilization patterns
analyze_gpu_utilization()
```

## Understanding Output

The tool generates several visualizations and a detailed report:

1. **throughput_vs_batch_size.png**: Shows how samples/second varies with batch size
2. **time_breakdown.png**: Breaks down execution time into data loading, forward pass, backward pass, and optimizer steps
3. **memory_usage.png**: Shows CPU and GPU memory usage across batch sizes
4. **performance_report.md**: Comprehensive analysis of bottlenecks and recommendations

## Common Performance Patterns

### 1. Low Throughput at Small Batch Sizes (e.g., Batch Size 8)

**Typical causes:**
- Fixed per-batch overhead dominates processing time
- GPU underutilization (low occupancy)
- Kernel launch overhead outweighs computation

**Solutions:**
- Use larger batch sizes if memory allows
- Optimize data loading pipeline
- Reduce CPU-GPU synchronization points

### 2. Optimal Throughput at Medium Batch Sizes (e.g., Batch Size 32)

**Why it happens:**
- Balances parallelism and memory usage
- Good GPU utilization without memory constraints
- Efficient use of tensor cores and CUDA cores

**Tips:**
- Use this batch size range for production training
- Consider accumulating gradients if you need a larger effective batch size

### 3. Declining Throughput at Large Batch Sizes (e.g., Batch Size 128)

**Typical causes:**
- GPU memory pressure causing throttling
- Memory swapping between GPU and CPU
- Inefficient memory access patterns
- Increased backward pass time

**Solutions:**
- Use gradient accumulation to maintain effective batch size while reducing memory usage
- Try mixed precision training (FP16/BF16)
- Implement gradient checkpointing
- Optimize model architecture

### 4. CPU vs. GPU Performance

**Analysis focus:**
- Speed-up ratio between GPU and CPU
- Data transfer bottlenecks
- Model size appropriateness for GPU acceleration

**Optimization tips:**
- For small models with low GPU speedup: consider CPU deployment
- For transfer bottlenecks: use `pin_memory=True` and increase `num_workers` 
- Enable CUDA graphs for repeated operations
- Use GPU-optimized libraries and operators

## Advanced Usage

For more detailed analysis, you can use the `PerformanceDiagnostics` class directly:

```python
from perf_diagnostics import PerformanceDiagnostics

# Initialize
profiler = PerformanceDiagnostics(model, loss_fn, optimizer)

# Run diagnostics
profiler.run_diagnostics(dataloader, batch_sizes=[8, 16, 32, 64, 128], 
                        num_iterations=50, 
                        devices=[torch.device('cpu'), torch.device('cuda')])

# Analyze results
results_df = profiler.get_results()
print(results_df)

# Generate visualizations
profiler.plot_throughput_vs_batch_size()
profiler.plot_time_breakdown()
profiler.plot_memory_usage()

# Get bottleneck report
report = profiler.generate_bottleneck_report()
print(report)
```

## Requirements

- Python 3.6+
- PyTorch
- matplotlib
- pandas
- numpy
- psutil
- tqdm
- nvidia-ml-py (optional, for GPU diagnostics)
