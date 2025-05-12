# DistilBERT Benchmarking on HPC Clusters

This directory contains SLURM job templates and submission scripts for running DistilBERT benchmarks on high-performance computing (HPC) clusters.

## Available Scripts

- `bench_distilbert.slurm` - SLURM job script for GPU-based benchmarking
- `bench_distilbert_cpu.slurm` - SLURM job script for CPU-only benchmarking
- `submit_benchmark.sh` - Wrapper script to submit benchmark jobs with custom parameters

## Quick Start

To run a benchmark on both CPU and GPU with default settings:

```bash
./submit_benchmark.sh
```

To run a benchmark on GPU only with custom settings:

```bash
./submit_benchmark.sh --device gpu --gpu-count 2 --batch-sizes "1,4,16,64,128" --analyze
```

## Environment Variables

Both SLURM scripts support customization via environment variables:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `JOB_NAME` | Name of the SLURM job | distilbert-bench-gpu/cpu |
| `OUTPUT_FILE` | Output file path | slurm-%j.out |
| `ERROR_FILE` | Error file path | slurm-%j.err |
| `PARTITION` | SLURM partition to use | gpu/compute |
| `TIME` | Job time limit | 01:00:00 (GPU), 02:00:00 (CPU) |
| `CPUS` | Number of CPUs to allocate | 4 (GPU), 8 (CPU) |
| `MEMORY` | Memory to allocate | 16G (GPU), 32G (CPU) |
| `GPU_COUNT` | Number of GPUs (GPU script only) | 1 |
| `GPU_TYPE` | Type of GPU to request (GPU script only) | * |
| `NODES` | Number of nodes | 1 |
| `TASKS_PER_NODE` | Tasks per node | 1 |
| `ANACONDA_VERSION` | Anaconda module version | latest |
| `CUDA_VERSION` | CUDA module version (GPU script only) | 11.8 |
| `GCC_VERSION` | GCC module version | latest |
| `CONDA_ENV` | Conda environment name | distilbert-benchmark |
| `BATCH_SIZES` | Batch sizes to benchmark | 1,2,4,8,16,32,64 |
| `WARMUP_RUNS` | Warmup runs before measuring | 5 |
| `ITERATIONS` | Benchmark iterations | 20 |
| `SHARED_DIR` | Directory to copy results to | None |
| `ANALYZE` | Whether to run analysis | false |
| `CLEANUP` | Whether to clean up temporary files | true |

## Wrapper Script Options

The `submit_benchmark.sh` script supports the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--device DEVICE` | Device to run on: 'cpu', 'gpu', or 'both' | both |
| `--cpus CPUS` | Number of CPUs to allocate | auto |
| `--memory MEMORY` | Memory to allocate | auto |
| `--time TIME` | Time limit | auto |
| `--partition PART` | SLURM partition to use | auto |
| `--batch-sizes SIZES` | Comma-separated list of batch sizes | 1,2,4,8,16,32,64 |
| `--iterations ITER` | Number of benchmark iterations | 20 |
| `--analyze` | Run analysis after benchmark | false |
| `--shared-dir DIR` | Directory to copy results to | ~/benchmark-results |
| `--env NAME` | Conda environment name | distilbert-benchmark |
| `--gpu-count COUNT` | Number of GPUs to use | 1 |
| `--gpu-type TYPE` | Type of GPU to request | * |
| `--help` | Show help message | - |

## Examples

### Basic GPU Benchmark

```bash
./submit_benchmark.sh --device gpu
```

### CPU Benchmark with Increased Resources

```bash
./submit_benchmark.sh --device cpu --cpus 16 --memory 64G
```

### Full Benchmark Suite with Analysis

```bash
./submit_benchmark.sh --device both --analyze --shared-dir /path/to/results
```

### Custom Batch Sizes and Iterations

```bash
./submit_benchmark.sh --batch-sizes "1,8,32,128" --iterations 50
```

## Cluster-Specific Customization

Different HPC clusters may require specific modifications:

1. **Module Names**: Adjust module names in the SLURM scripts based on your cluster's available modules.
2. **Partitions**: Change the partition names based on your cluster's configuration.
3. **Resource Limits**: Adjust memory, CPU, and time limits based on your cluster's policies.
4. **GPU Types**: Specify GPU types if your cluster has different GPU models.

For example, for a cluster with A100 GPUs in a partition named "a100":

```bash
./submit_benchmark.sh --device gpu --partition a100 --gpu-type A100
```

## Viewing Results

Results are stored in timestamped directories. If `--analyze` is enabled, analysis reports and figures will be generated in these directories.

If both CPU and GPU jobs are submitted with `--analyze`, a comparison report will be generated automatically after both jobs complete.

## Troubleshooting

- **Job Fails Immediately**: Check SLURM output files for errors. Common issues include module load failures or conda environment problems.
- **Out of Memory**: Increase `--memory` parameter or optimize batch sizes.
- **Time Limit Exceeded**: Increase `--time` parameter or reduce batch sizes/iterations.
- **Module Load Errors**: Adjust module names to match your cluster's available modules.

## Extending

To add support for additional HPC schedulers (e.g., PBS, LSF), create new template scripts following a similar structure to the SLURM scripts.
