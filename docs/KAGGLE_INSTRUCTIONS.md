# Running DistilBERT Benchmarking on Kaggle GPU

This document provides step-by-step instructions for running your DistilBERT benchmarking project on Kaggle's GPU platform.

## Prerequisites

1. A Kaggle account (free)
2. The `distilbert_kaggle.zip` file (created with the included script)

## Step 1: Set Up Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com/) and sign in
2. Click **Create** → **New Notebook**
3. In the notebook settings:
   - Click the **⋮** menu in the top right
   - Select **Settings**
   - Under **Accelerator**, choose **GPU** (T4 GPU)
   - Click **Save**

## Step 2: Upload Your Project Files

1. In the notebook interface, click the **Data** tab in the right sidebar
2. Click **Add Data** → **Upload Dataset**
3. Upload the `distilbert_kaggle.zip` file and give it a name (e.g., "distilbert-benchmarking")
4. Set privacy to "Private" and click **Create**

## Step 3: Configure Your Notebook

Create cells in your notebook and add the following code in sequence:

### Setup & Extraction

```python
# Extract the uploaded zip file
!mkdir -p /kaggle/working/distilbert
!unzip -q /kaggle/input/distilbert-benchmarking/distilbert_kaggle.zip -d /kaggle/working/distilbert
!ls -la /kaggle/working/distilbert
```

### Install Dependencies

```python
# Install dependencies
!pip install torch transformers datasets numpy pandas matplotlib seaborn
```

### Verify GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Run Benchmarking

```python
# Change to the working directory
import os
os.chdir('/kaggle/working/distilbert')

# Run the benchmark
!python -m src.main --device cuda --batch_sizes 8 16 32 64 --output /kaggle/working/results.jsonl
```

### Analyze Results

```python
# Generate analysis report
!python -m analysis.generate_report --input /kaggle/working/results.jsonl --output /kaggle/working/report.md

# Display the report
with open('/kaggle/working/report.md', 'r') as f:
    report_content = f.read()

from IPython.display import Markdown
Markdown(report_content)
```

### Visualize Results

```python
# Plot metrics
!python -m analysis.plot_metrics --input /kaggle/working/results.jsonl --output /kaggle/working/plots

# Display the plots
import matplotlib.pyplot as plt
import glob

# Create the plots directory if it doesn't exist
!mkdir -p /kaggle/working/plots

plot_files = glob.glob('/kaggle/working/plots/*.png')
for plot_file in plot_files:
    plt.figure(figsize=(10, 6))
    img = plt.imread(plot_file)
    plt.imshow(img)
    plt.axis('off')
    plt.title(os.path.basename(plot_file))
    plt.show()
```

## Step 4: Run Additional Experiments (Optional)

You can run additional experiments with different parameters:

```python
# Run benchmark with larger batch sizes
!python -m src.main --device cuda --batch_sizes 128 256 --output /kaggle/working/results_large_batch.jsonl

# Combine results
!cat /kaggle/working/results.jsonl /kaggle/working/results_large_batch.jsonl > /kaggle/working/results_combined.jsonl

# Generate comparison report
!python -m analysis.compare_results --input /kaggle/working/results_combined.jsonl --output /kaggle/working/comparison_report.md

# Display the comparison report
with open('/kaggle/working/comparison_report.md', 'r') as f:
    comparison_content = f.read()

Markdown(comparison_content)
```

## Saving Results

To save the results from your Kaggle session:

1. The plots and results will be automatically saved in the Kaggle notebook output
2. You can download the results files by:
   - Going to the **Output** tab in the right sidebar
   - Clicking the download icon next to each file you want to save
3. To save the entire notebook with results:
   - Click **Save Version** in the top right
   - Add a version name and description
   - Click **Save**

## Troubleshooting

- If you encounter memory issues with larger batch sizes, try reducing them
- Ensure your Kaggle session has GPU enabled (you should see "GPU" in the top right corner)
- If packages are missing, install them with `!pip install [package-name]`

## Advantages of Kaggle

- Free GPU access
- Interactive development environment
- Easy visualization
- No complex setup or job scheduling
- Results and notebook can be shared easily

## Limitations

- GPU sessions limited to 12 hours
- Single GPU only
- Limited disk space (75GB)
- Less computational power than dedicated HPC clusters 