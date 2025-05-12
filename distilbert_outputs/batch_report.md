# DistilBERT Benchmark Insights Report

*Generated on 2025-05-06 15:01:30*

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
- **Batch Sizes Tested**: 8, 16, 32, 64, 128
- **Sequence Length**: 128
- **Iterations**: 20
- **Warmup Runs**: 5

## Key Performance Metrics

|   Batch Size |   Latency (ms) |   Throughput (samples/sec) |   CPU Memory (MB) |   GPU Memory (MB) |
|-------------:|---------------:|---------------------------:|------------------:|------------------:|
|         8.00 |          15.56 |                     514.72 |           1601.66 |            746.88 |
|        16.00 |          26.54 |                     602.75 |           1602.78 |            796.88 |
|        32.00 |          51.91 |                     616.49 |           1616.91 |            892.88 |
|        64.00 |         109.80 |                     582.91 |           1617.16 |           1084.88 |
|       128.00 |         221.10 |                     578.92 |           1617.66 |           1468.88 |

## Performance Insights

- **Optimal Batch Size for Throughput**: 32
  - Achieves 616.49 samples/second
- **Optimal Batch Size for Latency**: 8
  - Achieves 15.56 ms latency
- **Most Memory-Efficient Batch Size**: 128
  - Uses 12.64 MB per sample
- **Scaling Efficiency**: 0.07
  - Scaling starts to decline after batch size 8
- **Estimated Daily Throughput**: 53,264,538 samples/day
  - Equivalent to 53.26 million samples/day

## Potential Bottlenecks

- Throughput plateaus for large batch sizes, possibly due to memory bandwidth limitations

## Recommendations

- For maximum throughput, use batch size **32**
- For minimum latency, use batch size **8**
- Address the identified bottlenecks to improve performance:
  - Consider using a device with more memory or optimizing memory usage

## Conclusion

The DistilBERT model demonstrates 616.49 samples/second maximum throughput and 15.56 ms minimum latency on CUDA. The model shows poor scaling efficiency (0.07) with larger batch sizes, suggesting resource limitations. At optimal settings, this configuration can process approximately 53.26 million samples per day.
