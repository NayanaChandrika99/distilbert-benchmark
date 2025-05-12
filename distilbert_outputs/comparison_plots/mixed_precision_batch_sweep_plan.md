# Mixed Precision Batch Size Sweep Plan

## Objective
Extend the existing benchmark with mixed precision (AMP) tests across all batch sizes to identify optimal throughput and latency configurations.

## Current Findings
- Mixed precision has been tested with batch sizes 8, 16, and 32
- Initial results show ~4% throughput improvement at batch size 8
- Larger batch sizes should be tested to determine if scaling improves with mixed precision

## Proposed Test Matrix

| Batch Size | FP32 (done) | Mixed Precision |
|------------|-------------|-----------------|
| 8          | ✅          | ✅              |
| 16         | ✅          | ✅              |
| 32         | ✅          | ✅              |
| 64         | ✅          | 🔲              |
| 128        | ✅          | 🔲              |
| 256        | 🔲          | 🔲              |

## Implementation Steps

1. Extend the existing benchmark script to support larger batch sizes with mixed precision
2. Run the following command for each missing configuration:

```bash
python benchmark_distilbert.py \
  --model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english" \
  --batch_size=64 \
  --device="cuda" \
  --mixed_precision \
  --output_dir="distilbert_outputs"
```

3. Run the same for batch size 128 and 256 (if GPU memory allows)
4. Rerun the comparison analysis with the new complete dataset

## Expected Benefits

- Confirm if mixed precision provides greater scaling efficiency at larger batch sizes
- Identify the optimal batch size for mixed precision inference
- Quantify power efficiency improvements with mixed precision

## Data Analysis

Update the plotting scripts to include all new data points and generate:
1. Combined throughput plots showing FP32 vs AMP across all batch sizes
2. Scaling efficiency analysis comparing FP32 and AMP
3. Energy efficiency metrics (samples/joule) for each configuration

Generated: 2025-05-06 12:15:48
