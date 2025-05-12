# DistilBERT Batch Size Performance Analysis

Generated on: 2025-05-11 17:12:17

## Throughput Analysis

- Small batch (size 8) throughput: 514.72 samples/second
- Optimal batch (size 32) throughput: 616.49 samples/second
- Large batch (size 128) throughput: 578.92 samples/second
- Throughput improvement from small to optimal: 19.8%
- **Throughput decline from optimal to large batch: 6.1%**

## Root Causes of Observed Performance Patterns

### 1. Low Throughput at Small Batch Sizes

**Root causes:**
- **Fixed overhead dominance**: Each batch incurs constant costs regardless of size
  - CUDA kernel launch overhead (microseconds per launch)
  - Memory allocation/deallocation costs
  - Execution queue management
- **GPU underutilization**: Small batches don't provide enough parallelism
  - NVIDIA GPUs have thousands of CUDA cores that remain partly idle
  - The GPU's SIMD architecture is most efficient with high parallelism

### 2. Optimal Throughput at Medium Batch Sizes

**Root causes:**
- **Parallelism saturation**: The GPU reaches high occupancy
  - Most CUDA cores are active and processing useful work
  - Memory access patterns become more coalesced
- **Balanced resource usage**: Balance between:
  - Compute resources (CUDA cores)
  - Memory bandwidth
  - Cache utilization

### 3. Declining Throughput at Large Batch Sizes

**Root causes:**
- **Memory constraints**: 
  - Memory usage increases from 892.9MB at optimal batch to 1468.9MB at largest batch
  - Increased cache pressure and thrashing
  - Internal memory fragmentation increases
- **Execution serialization**:
  - More serialization of operations occurs as batch size grows
  - Memory transfer operations start blocking compute operations

## Recommendations

- **For maximum throughput**: Use batch size 32