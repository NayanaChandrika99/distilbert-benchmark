#!/bin/bash
#SBATCH --job-name=${JOB_NAME:-distilbert-bench-cpu}
#SBATCH --output=${OUTPUT_FILE:-slurm-cpu-%j.out}
#SBATCH --error=${ERROR_FILE:-slurm-cpu-%j.err}
#SBATCH --comment="DistilBERT Benchmark on CPU"

# Print job information
echo "================ Job Information ================"
echo "Running on host: $(hostname)"
echo "Job id: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "Temporary directory: $TMPDIR"
echo "Current time: $(date)"
echo "================================================="

# Load modules (Zaratan HPC specific configuration)
echo "Loading modules..."
module purge
module load anaconda3
echo "Modules loaded."

# Create a temporary directory for intermediate results if needed
TEMP_DIR=${TEMP_DIR:-${TMPDIR:-/tmp}/distilbert-cpu-${SLURM_JOB_ID}}
mkdir -p $TEMP_DIR
echo "Created temporary directory: $TEMP_DIR"

# Set OpenMP and MKL variables for optimal CPU performance
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
echo "Set OMP_NUM_THREADS=$OMP_NUM_THREADS, MKL_NUM_THREADS=$MKL_NUM_THREADS"

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-ml-benchmark}"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment activated."

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=${RESULTS_DIR:-"results-${TIMESTAMP}"}
mkdir -p $RESULTS_DIR
echo "Created results directory: $RESULTS_DIR"

# Prepare config file if custom configuration provided
CONFIG_FILE=${CONFIG_FILE:-config.yaml}
if [ -n "$CUSTOM_CONFIG" ]; then
    echo "Using custom configuration: $CUSTOM_CONFIG"
    echo "$CUSTOM_CONFIG" > $CONFIG_FILE
fi

# Run benchmark
echo "================================================="
echo "Starting benchmark at $(date)"
python src/runner.py \
    --config $CONFIG_FILE \
    --device cpu \
    --batch-sizes ${BATCH_SIZES:-"1,2,4,8,16,32,64"} \
    --warmup-runs ${WARMUP_RUNS:-5} \
    --iterations ${ITERATIONS:-20} \
    --output $RESULTS_DIR/cpu_results.jsonl
echo "Benchmark completed at $(date)"

# Run analysis if requested
if [ "${ANALYZE:-false}" = "true" ]; then
    echo "================================================="
    echo "Running analysis at $(date)"
    python analysis/plot_metrics.py --input $RESULTS_DIR/cpu_results.jsonl --output $RESULTS_DIR/figures
    python analysis/generate_report.py --input $RESULTS_DIR/cpu_results.jsonl --output $RESULTS_DIR/insights.md
    echo "Analysis completed at $(date)"
fi

# Copy results to shared storage if needed
if [ -n "${SHARED_DIR}" ]; then
    echo "================================================="
    echo "Copying results to shared storage at $(date)"
    # Create target directory with job ID for better organization
    TARGET_DIR=${SHARED_DIR}/${SLURM_JOB_ID}-${TIMESTAMP}
    mkdir -p $TARGET_DIR
    rsync -avz $RESULTS_DIR/ $TARGET_DIR/
    echo "Results copied to: $TARGET_DIR"
fi

# Clean up temporary files if needed
if [ "${CLEANUP:-true}" = "true" ]; then
    echo "================================================="
    echo "Cleaning up temporary files..."
    rm -rf $TEMP_DIR
    echo "Cleanup completed."
fi

echo "================================================="
echo "Job completed at $(date)"
echo "Results are in: $RESULTS_DIR"
if [ -n "${SHARED_DIR}" ]; then
    echo "and copied to: $TARGET_DIR"
fi
echo "================================================="

# End of script
