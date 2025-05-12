#!/bin/bash
# Wrapper script to submit SLURM jobs for DistilBERT benchmarking

set -euo pipefail

# ---------- defaults ----------
DEVICE=${DEVICE:-"both"}  # "cpu", "gpu", or "both"
SCRIPT_DIR=$(dirname "$0")
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-64G}
TIME=${TIME:-02:00:00}
CONDA_ENV=${CONDA_ENV:-ml-benchmark}
GPU_COUNT=${GPU_COUNT:-1}
GPU_TYPE=${GPU_TYPE:-a100}            # or v100 / k80 depending on node type
SHARED_DIR=${SHARED_DIR:-/scratch/zt1/$USER/benchmarks}
PARTITION_CPU=${PARTITION_CPU:-compute}
PARTITION_GPU=${PARTITION_GPU:-gpu}
ANALYZE=${ANALYZE:-"false"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition-cpu)
            PARTITION_CPU="$2"
            shift 2
            ;;
        --partition-gpu)
            PARTITION_GPU="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --analyze)
            ANALYZE="true"
            shift
            ;;
        --shared-dir)
            SHARED_DIR="$2"
            shift 2
            ;;
        --env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "DistilBERT Benchmark Job Submission Script for Zaratan HPC"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --device DEVICE      Device to run on: 'cpu', 'gpu', or 'both' (default: both)"
            echo "  --cpus CPUS          Number of CPUs to allocate (default: 8)"
            echo "  --memory MEMORY      Memory to allocate (default: 64G)"
            echo "  --time TIME          Time limit (default: 02:00:00)"
            echo "  --partition-cpu PART SLURM CPU partition to use (default: compute)"
            echo "  --partition-gpu PART SLURM GPU partition to use (default: gpu)"
            echo "  --batch-sizes SIZES  Comma-separated list of batch sizes (default: 1,2,4,8,16,32,64)"
            echo "  --iterations ITER    Number of benchmark iterations (default: 20)"
            echo "  --analyze            Run analysis after benchmark (default: false)"
            echo "  --shared-dir DIR     Directory to copy results to (default: /scratch/zt1/\$USER/benchmarks)"
            echo "  --env NAME           Conda environment name (default: ml-benchmark)"
            echo "  --gpu-count COUNT    Number of GPUs to use (default: 1)"
            echo "  --gpu-type TYPE      Type of GPU to request (default: a100)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Create shared directory if it doesn't exist
if [ ! -d "$SHARED_DIR" ]; then
    echo "Creating shared directory: $SHARED_DIR"
    mkdir -p "$SHARED_DIR"
fi

# Initialize job IDs storage
CPU_JOB_ID=""
GPU_JOB_ID=""

# Submit CPU job if requested
if [ "$DEVICE" = "cpu" ] || [ "$DEVICE" = "both" ]; then
    echo "Submitting CPU benchmark job..."

    # Prepare environment variables
    export JOB_NAME="distilbert-bench-cpu"
    export SHARED_DIR="$SHARED_DIR"
    export ANALYZE="$ANALYZE"
    export CONDA_ENV="$CONDA_ENV"

    # Submit the job
    CPU_JOB_ID=$(sbatch \
        --cpus-per-task=$CPUS \
        --mem=$MEMORY \
        --time=$TIME \
        --partition=$PARTITION_CPU \
        "$SCRIPT_DIR/bench_distilbert_cpu.slurm" | awk '{print $NF}')
    echo "CPU benchmark job submitted with ID: $CPU_JOB_ID"
fi

# Submit GPU job if requested
if [ "$DEVICE" = "gpu" ] || [ "$DEVICE" = "both" ]; then
    echo "Submitting GPU benchmark job..."

    # Prepare environment variables
    export JOB_NAME="distilbert-bench-gpu"
    export SHARED_DIR="$SHARED_DIR"
    export ANALYZE="$ANALYZE"
    export CONDA_ENV="$CONDA_ENV"

    # Submit the job
    GPU_JOB_ID=$(sbatch \
        --cpus-per-task=$CPUS \
        --mem=$MEMORY \
        --time=$TIME \
        --partition=$PARTITION_GPU \
        --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
        "$SCRIPT_DIR/bench_distilbert.slurm" | awk '{print $NF}')
    echo "GPU benchmark job submitted with ID: $GPU_JOB_ID"
fi

# Submit comparison job if both CPU and GPU jobs were submitted
if [ -n "$CPU_JOB_ID" ] && [ -n "$GPU_JOB_ID" ] && [ "$ANALYZE" = "true" ]; then
    echo "Will submit comparison job after CPU and GPU jobs complete..."

    # Create a temporary script for the comparison job
    COMPARE_SCRIPT=$(mktemp)
    cat > "$COMPARE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=distilbert-compare
#SBATCH --output=slurm-compare-%j.out
#SBATCH --error=slurm-compare-%j.err
#SBATCH --partition=$PARTITION_CPU
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

echo "Running comparison analysis at \$(date)"

# Load modules and activate environment
module purge
module load anaconda3
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# Get result paths from the completed jobs
CPU_RESULTS=\$(ls -t ${SHARED_DIR}/${CPU_JOB_ID}-*/*.jsonl | head -1)
GPU_RESULTS=\$(ls -t ${SHARED_DIR}/${GPU_JOB_ID}-*/*.jsonl | head -1)

# Create comparison directory
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
COMPARISON_DIR="${SHARED_DIR}/comparison-\${TIMESTAMP}"
mkdir -p "\$COMPARISON_DIR"

# Run comparison
echo "Comparing CPU results (\$CPU_RESULTS) with GPU results (\$GPU_RESULTS)"
python analysis/compare_results.py --cpu "\$CPU_RESULTS" --gpu "\$GPU_RESULTS" --output "\$COMPARISON_DIR/comparison.md"

echo "Comparison completed. Results in \$COMPARISON_DIR"
EOF

    chmod +x "$COMPARE_SCRIPT"

    # Submit the comparison job with dependencies on the CPU and GPU jobs
    COMPARE_JOB_ID=$(sbatch --dependency=afterok:$CPU_JOB_ID:$GPU_JOB_ID "$COMPARE_SCRIPT" | awk '{print $NF}')
    echo "Comparison job submitted with ID: $COMPARE_JOB_ID"
    echo "It will run after both CPU and GPU jobs complete successfully."

    # Clean up the temporary script
    rm "$COMPARE_SCRIPT"
fi

echo "All jobs submitted successfully. Use 'squeue -u $USER' to check status."
echo "Results will be stored in: $SHARED_DIR"

# End of script
