# Kaggle Scripts for DistilBERT Benchmarking

This directory contains scripts and notebooks for running DistilBERT benchmarks on Kaggle.

## Files

- `run_distilbert_on_kaggle.py` - Script to run DistilBERT benchmarks on Kaggle
- `distilbert_kaggle_notebook.ipynb` - Jupyter notebook version of the benchmark for Kaggle
- `kaggle_download_notebook.ipynb` - Notebook to download benchmark results from Kaggle
- `KAGGLE_INSTRUCTIONS.md` - Detailed instructions for running benchmarks on Kaggle
- `run_mixed_precision_sweep.py` - Script to run mixed precision benchmarks with different batch sizes

## Usage

### Running the benchmark on Kaggle

1. Upload the necessary files to Kaggle:
   
   ```bash
   # Follow the instructions in KAGGLE_INSTRUCTIONS.md
   ```

2. Run the mixed precision sweep:

   ```python
   # In a Kaggle notebook
   !python run_mixed_precision_sweep.py
   ```

3. Download the results:
   - Use the download button in the Kaggle UI
   - Or follow the download instructions in the notebook

## Requirements

- Kaggle account with GPU access
- Sufficient Kaggle session time for benchmarks 