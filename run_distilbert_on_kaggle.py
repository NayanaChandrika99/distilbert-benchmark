#!/usr/bin/env python3
"""
DistilBERT Benchmarking on Kaggle GPU

This script provides instructions and code snippets for running the 
DistilBERT benchmarking project on Kaggle's GPU environment.

How to use:
1. Upload distilbert_kaggle.zip to Kaggle
2. Create a new notebook and enable GPU
3. Copy and paste sections of this script as needed
"""

# ========================================================================
# SECTION 1: Setup
# ========================================================================
# Extract the uploaded zip file
# !mkdir -p /kaggle/working/distilbert
# !unzip -q /kaggle/input/distilbert-benchmarking/distilbert_kaggle.zip -d /kaggle/working/distilbert
# !ls -la /kaggle/working/distilbert

# ========================================================================
# SECTION 2: Set Up Environment
# ========================================================================
# Install dependencies
# !pip install torch transformers datasets numpy pandas matplotlib seaborn

# ========================================================================
# SECTION 3: Verify GPU Availability
# ========================================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ========================================================================
# SECTION 4: Run Benchmarking
# ========================================================================
# Change to the working directory
import os
# os.chdir('/kaggle/working/distilbert')

# Run the benchmark
# !python -m src.main --device cuda --batch_sizes 8 16 32 64 --output /kaggle/working/results.jsonl

# ========================================================================
# SECTION 5: Analyze Results
# ========================================================================
# Generate analysis report
# !python -m analysis.generate_report --input /kaggle/working/results.jsonl --output /kaggle/working/report.md

# Display the report (in a notebook)
"""
with open('/kaggle/working/report.md', 'r') as f:
    report_content = f.read()

from IPython.display import Markdown
Markdown(report_content)
"""

# ========================================================================
# SECTION 6: Plot Performance Metrics
# ========================================================================
# Plot metrics
# !python -m analysis.plot_metrics --input /kaggle/working/results.jsonl --output /kaggle/working/plots

# Display the plots (in a notebook)
"""
import matplotlib.pyplot as plt
import glob

plot_files = glob.glob('/kaggle/working/plots/*.png')
for plot_file in plot_files:
    plt.figure(figsize=(10, 6))
    img = plt.imread(plot_file)
    plt.imshow(img)
    plt.axis('off')
    plt.title(os.path.basename(plot_file))
    plt.show()
"""

# ========================================================================
# SECTION 7: Compare with Different Batch Sizes
# ========================================================================
# Run benchmark with larger batch sizes
# !python -m src.main --device cuda --batch_sizes 128 256 --output /kaggle/working/results_large_batch.jsonl

# Combine results
# !cat /kaggle/working/results.jsonl /kaggle/working/results_large_batch.jsonl > /kaggle/working/results_combined.jsonl

# Generate comparison report
# !python -m analysis.compare_results --input /kaggle/working/results_combined.jsonl --output /kaggle/working/comparison_report.md

# Display the comparison report (in a notebook)
"""
with open('/kaggle/working/comparison_report.md', 'r') as f:
    comparison_content = f.read()

Markdown(comparison_content)
""" 