# DistilBERT Benchmark Insights Report

*Generated on 2025-05-06 14:55:33*

## Model Information

- **Model**: distilbert-base-uncased-finetuned-sst-2-english
- **Hidden Size**: 768
- **Hidden Layers**: 6
- **Attention Heads**: 12
- **Parameters**: 66,955,010

## System Information

- **Platform**: Linux-6.6.56+-x86_64-with-glibc2.35
- **Processor**: x86_64
- **CPU Count**: 4
- **Python Version**: 3.11.11
- **PyTorch Version**: 2.5.1+cu124

## Benchmark Summary

- **Device**: CUDA
- **Batch Sizes Tested**: 8
- **Sequence Length**: 128
- **Iterations**: 20
- **Warmup Runs**: 5

## Key Performance Metrics

|   Batch Size |   Latency (ms) |   Throughput (samples/sec) |   CPU Memory (MB) |   GPU Memory (MB) |
|-------------:|---------------:|---------------------------:|------------------:|------------------:|
|         8.00 |          16.10 |                     497.94 |           1599.68 |            746.88 |

## Performance Insights

- **Optimal Batch Size for Throughput**: 8
  - Achieves 497.94 samples/second
- **Optimal Batch Size for Latency**: 8
  - Achieves 16.10 ms latency
- **Most Memory-Efficient Batch Size**: 8
  - Uses 199.96 MB per sample
- **Estimated Daily Throughput**: 43,022,037 samples/day
  - Equivalent to 43.02 million samples/day

## Recommendations

- For maximum throughput, use batch size **8**
- For minimum latency, use batch size **8**

## Conclusion

The DistilBERT model demonstrates 497.94 samples/second maximum throughput and 16.10 ms minimum latency on CUDA. At optimal settings, this configuration can process approximately 43.02 million samples per day.
