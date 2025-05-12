# DistilBERT Benchmark: CPU vs. GPU Comparison

## Summary

This report compares the performance of DistilBERT on CPU and GPU devices.

## System Information

### CPU

- Platform: Linux-6.6.56+-x86_64-with-glibc2.35
- Processor: x86_64
- CPU Count: 4
- Python: 3.11.11
- PyTorch: 2.5.1+cu124

### GPU

No GPU information available or CUDA not available.

## Model Information

- Model: distilbert-base-uncased-finetuned-sst-2-english
- Hidden Size: 768
- Hidden Layers: 6
- Attention Heads: 12
- Parameters: 66,955,010

## Performance Comparison

|   batch_size |   cpu_latency_ms_mean |   gpu_latency_ms_mean |   speedup_latency_ms_mean |   cpu_throughput_mean |   gpu_throughput_mean |   speedup_throughput_mean |   cpu_cpu_memory_mb_max |   gpu_cpu_memory_mb_max |   gpu_gpu_memory_mb_max |   gpu_gpu_avg_power_w |
|-------------:|----------------------:|----------------------:|--------------------------:|----------------------:|----------------------:|--------------------------:|------------------------:|------------------------:|------------------------:|----------------------:|
|         8.00 |                449.57 |                 16.10 |                     27.93 |                 17.83 |                497.94 |                     27.93 |                 1671.68 |                 1599.68 |                  746.88 |                216.24 |

### Interpretation

- **Latency**: GPU is on average 27.93x faster than CPU.
  - The maximum speedup of 27.93x is achieved at batch size 8.

- **Throughput**: GPU achieves on average 27.93x higher throughput than CPU.
  - The maximum throughput improvement of 27.93x is achieved at batch size 8.

- **Memory Usage**:
  - Batch size 8.0: CPU 1671.68 MB, GPU 746.88 MB

- **Power Consumption**:
  - Batch size 8.0: GPU 216.24 W

- **Recommendation**: For optimal performance, use batch size 8 on GPU.

## Conclusion

The GPU provides significant performance benefits with an average of 27.93x higher throughput compared to CPU execution. For production deployments, GPU is recommended for maximum performance.

