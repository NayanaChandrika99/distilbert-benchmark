# DistilBERT Comprehensive Benchmark Report

*Generated on 2025-05-06 12:12:37*

## Model Information

- **Model**: distilbert-base-uncased-finetuned-sst-2-english
- **Hidden Size**: 768
- **Hidden Layers**: 6
- **Attention Heads**: 12
- **Parameters**: 66,955,010


## System Information

- **Platform**: Linux-6.6.56+-x86_64-with-glibc2.35
- **Python Version**: 3.11.11
- **Processor**: x86_64
- **CPU Count**: 4
- **PyTorch Version**: 2.5.1+cu124
- **GPU**: Tesla P100-PCIE-16GB
- **CUDA Version**: 12.4
- **cuDNN Version**: 90300


## Performance Metrics

### CPU - FP32

|   Batch Size |   Latency (ms) |   Throughput (samples/sec) |   CPU Memory (MB) |   GPU Memory (MB) |
|-------------:|---------------:|---------------------------:|------------------:|------------------:|
|         8.00 |         449.57 |                      17.83 |           1671.68 |               nan |
|        16.00 |         937.44 |                      17.07 |           1751.04 |               nan |

### CUDA - FP32

|   Batch Size |   Latency (ms) |   Throughput (samples/sec) |   CPU Memory (MB) |   GPU Memory (MB) |
|-------------:|---------------:|---------------------------:|------------------:|------------------:|
|         8.00 |          15.56 |                     514.72 |           1601.66 |            746.88 |
|        16.00 |          26.54 |                     602.75 |           1602.78 |            796.88 |
|        32.00 |          51.91 |                     616.49 |           1616.91 |            892.88 |
|        64.00 |         109.80 |                     582.91 |           1617.16 |           1084.88 |
|       128.00 |         221.10 |                     578.92 |           1617.66 |           1468.88 |

### CUDA - Mixed Precision (AMP)

|   Batch Size |   Latency (ms) |   Throughput (samples/sec) |   CPU Memory (MB) |   GPU Memory (MB) |
|-------------:|---------------:|---------------------------:|------------------:|------------------:|
|         8.00 |          15.45 |                     517.79 |           1601.30 |            746.88 |
|        16.00 |          26.46 |                     604.79 |           1602.42 |            796.88 |
|        32.00 |          51.73 |                     618.65 |           1616.55 |            892.88 |



## Optimal Configurations

### Maximum Throughput
- **Batch Size**: 32
- **Precision**: Mixed Precision
- **Throughput**: 618.65 samples/second
- **Estimated Daily Throughput**: 53,451,187.85 samples/day (53.45 million samples/day)

### Minimum Latency
- **Batch Size**: 8
- **Precision**: Mixed Precision
- **Latency**: 15.45 ms


## Scaling Efficiency

### CPU - FP32

- **Overall Scaling Efficiency**: 0.48
- **Baseline Batch Size**: 8
- **Maximum Batch Size**: 16

|   Batch Size |   Per-Item Throughput |   Scaling Efficiency |
|-------------:|----------------------:|---------------------:|
|         8.00 |                  2.23 |                 1.00 |
|        16.00 |                  1.07 |                 0.48 |

### CUDA - FP32

- **Overall Scaling Efficiency**: 0.07
- **Baseline Batch Size**: 8
- **Maximum Batch Size**: 128

|   Batch Size |   Per-Item Throughput |   Scaling Efficiency |
|-------------:|----------------------:|---------------------:|
|         8.00 |                 64.34 |                 1.00 |
|        16.00 |                 37.67 |                 0.59 |
|        32.00 |                 19.27 |                 0.30 |
|        64.00 |                  9.11 |                 0.14 |
|       128.00 |                  4.52 |                 0.07 |

### CUDA - Mixed Precision

- **Overall Scaling Efficiency**: 0.30
- **Baseline Batch Size**: 8
- **Maximum Batch Size**: 32

|   Batch Size |   Per-Item Throughput |   Scaling Efficiency |
|-------------:|----------------------:|---------------------:|
|         8.00 |                 64.72 |                 1.00 |
|        16.00 |                 37.80 |                 0.58 |
|        32.00 |                 19.33 |                 0.30 |



## Mixed Precision Speedup

|   Batch Size |   Throughput Improvement (%) |   Latency Improvement (%) |
|-------------:|-----------------------------:|--------------------------:|
|         8.00 |                         0.60 |                      0.66 |
|        16.00 |                         0.34 |                      0.34 |
|        32.00 |                         0.35 |                      0.35 |



## Device Speedup Comparison

|   Batch Size | Mixed Precision   | Baseline Device   | Target Device   |   Throughput Speedup (×) |   Latency Speedup (×) |
|-------------:|:------------------|:------------------|:----------------|-------------------------:|----------------------:|
|            8 | No                | cpu               | cuda            |                    28.87 |                 28.90 |
|           16 | No                | cpu               | cuda            |                    35.30 |                 35.32 |



## Recommendations

1. **For Maximum Throughput**: Use batch size 32 with Mixed Precision
   * Achieves 618.65 samples/second
   * Can process approximately 53.45 million samples per day

2. **For Minimum Latency**: Use batch size 8 with Mixed Precision
   * Achieves 15.45 ms latency

3. **Hardware Recommendations**:
   * GPU provides significantly better performance than CPU (up to 35.3x throughput)
   * For production deployments, GPU is strongly recommended

## Conclusion

The DistilBERT model demonstrates excellent performance characteristics on GPU hardware. Mixed precision provides modest gains in throughput and latency without requiring additional memory. For optimal efficiency, batch size 32 provides the best balance of throughput and resource utilization.
