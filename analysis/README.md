# Analysis Scripts for DistilBERT Benchmark Results

This directory contains scripts for analyzing and visualizing DistilBERT benchmark results.

## Files

- `generate_comparison_plots.py` - Creates visualization plots for batch size, precision, and CPU/GPU comparisons
- `generate_comprehensive_report.py` - Produces a detailed analysis report combining all benchmark data
- `run_mixed_precision_sweep.py` - Script to run mixed precision benchmarks (copy of the one in the kaggle directory)

## Usage

### Generating Comparison Plots

```bash
python generate_comparison_plots.py --input distilbert_outputs/all_benchmark_results --output distilbert_outputs/new_plots
```

### Generating Comprehensive Reports

```bash
python generate_comprehensive_report.py --input distilbert_outputs/all_benchmark_results --output distilbert_outputs/new_reports
```

## Inputs

The scripts expect benchmark results in JSONL format with the following key files:
- `batch_comparison.jsonl` - Results from GPU batch size comparisons
- `cpu_results.jsonl` - Results from CPU benchmarks
- `mixed_precision_results.jsonl` - Results from mixed precision benchmarks
- `absolute_results.jsonl` - Reference benchmark results

## Outputs

- Visualizations showing performance comparisons across:
  - Different batch sizes
  - CPU vs GPU
  - FP32 vs mixed precision
- Markdown reports with key findings and recommendations
