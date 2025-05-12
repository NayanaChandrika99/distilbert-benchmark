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
- **HPC Integration**: Includes SLURM scripts for high-performance computing clusters
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

```
distilbert-benchmark/
├── analysis/              # Analysis scripts and reporting tools
├── cluster/               # SLURM templates and HPC scripts
├── data/                  # Dataset handling and caching
├── src/                   # Core benchmarking implementation
│   ├── metrics/           # Performance metric collectors
│   ├── model.py           # Model loading utilities
│   ├── runner.py          # Benchmark orchestration
│   └── manifest.py        # Artifact integrity verification
├── tests/                 # Unit and integration tests
├── config.yaml            # Benchmark configuration
├── environment.yml        # Conda environment definition
└── Makefile               # Build automation
```

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
