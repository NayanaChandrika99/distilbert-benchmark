#!/bin/bash
# Mock SLURM environment for testing SLURM scripts

echo "Setting up mock SLURM environment for testing"

# Create a temporary directory for mock SLURM binaries
MOCK_DIR=$(mktemp -d)
echo "Created mock SLURM directory: $MOCK_DIR"

# Add mock SLURM directory to PATH
export PATH="$MOCK_DIR:$PATH"

# Create mock sbatch command
cat > "$MOCK_DIR/sbatch" << 'EOF'
#!/bin/bash
# Mock sbatch command

# Parse command line options
job_script=""
test_only=false
dependency=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --test-only)
            test_only=true
            shift
            ;;
        --dependency=*)
            dependency="${key#*=}"
            shift
            ;;
        *)
            # Assume it's the job script if not recognized as an option
            if [[ "$key" == *.slurm ]] || [[ -f "$key" ]]; then
                job_script="$key"
            fi
            shift
            ;;
    esac
done

# If no script was found, assume it's the last argument
if [ -z "$job_script" ] && [ -n "$1" ]; then
    job_script="$1"
fi

if [ -z "$job_script" ]; then
    echo "Error: No job script specified" >&2
    exit 1
fi

if [ ! -f "$job_script" ]; then
    echo "Error: Job script $job_script not found" >&2
    exit 1
fi

# In test-only mode, just check if the script exists and is executable
if [ "$test_only" = true ]; then
    if [ -x "$job_script" ]; then
        echo "Job script $job_script is valid"
        exit 0
    else
        echo "Warning: Job script $job_script is not executable" >&2
        exit 0
    fi
fi

# Generate a random job ID
job_id=$((1000000 + RANDOM % 9000000))

# Extract job attributes with standard grep
out_file=$(grep -E '#SBATCH\s+--output=' "$job_script" | sed -E 's/.*--output=([^ ]*).*/\1/' | head -1)
err_file=$(grep -E '#SBATCH\s+--error=' "$job_script" | sed -E 's/.*--error=([^ ]*).*/\1/' | head -1)
job_name=$(grep -E '#SBATCH\s+--job-name=' "$job_script" | sed -E 's/.*--job-name=([^ ]*).*/\1/' | head -1)

# Replace %j with job ID
out_file=${out_file//%j/$job_id}
err_file=${err_file//%j/$job_id}

# Use default names if not specified
out_file=${out_file:-slurm-$job_id.out}
err_file=${err_file:-slurm-$job_id.err}
job_name=${job_name:-mock_job}

echo "Submitted batch job $job_id"
echo "Job name: $job_name"
echo "Output file: $out_file"
echo "Error file: $err_file"

# For testing purposes, you can actually run the script if desired
if [ "${MOCK_EXECUTE:-false}" = "true" ]; then
    echo "Mock executing job script..."
    bash "$job_script" > "$out_file" 2> "$err_file" &
    echo "Job script execution started in background"
fi

exit 0
EOF

# Make mock sbatch executable
chmod +x "$MOCK_DIR/sbatch"

# Create mock squeue command
cat > "$MOCK_DIR/squeue" << 'EOF'
#!/bin/bash
# Mock squeue command

# Parse command line options
user=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -u)
            user="$2"
            shift 2
            ;;
        --user=*)
            user="${key#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# If no user specified, use current user
user=${user:-$USER}

# Print header
printf "%-10s %-15s %-8s %-10s %-15s\n" "JOBID" "NAME" "USER" "STATE" "TIME"

# Print mock job info (random job for demo)
job_id=$((1000000 + RANDOM % 9000000))
job_name="mock_job"
state="PENDING"
time="0:01"

printf "%-10s %-15s %-8s %-10s %-15s\n" "$job_id" "$job_name" "$user" "$state" "$time"

exit 0
EOF

# Make mock squeue executable
chmod +x "$MOCK_DIR/squeue"

# Create mock module command if not already available
if ! command -v module &> /dev/null; then
    cat > "$MOCK_DIR/module" << 'EOF'
#!/bin/bash
# Mock module command

if [ "$1" = "purge" ]; then
    echo "Purged all modules"
elif [ "$1" = "load" ]; then
    shift
    echo "Loaded module(s): $*"
elif [ "$1" = "list" ]; then
    echo "Currently loaded modules:"
    echo "  1) anaconda3/latest"
    echo "  2) cuda/11.8"
    echo "  3) gcc/latest"
fi

exit 0
EOF

    chmod +x "$MOCK_DIR/module"
    echo "Created mock module command"
fi

# Set environment variables that SLURM would typically set
export SLURM_JOB_ID=12345
export SLURM_JOB_NAME="mock_job"
export SLURM_JOB_NUM_NODES=1
export SLURM_NTASKS=1
export SLURM_CPUS_PER_TASK=4
export TMPDIR="/tmp/mock-slurm-$SLURM_JOB_ID"
export SLURM_USER=$USER

# Create mock TMPDIR
mkdir -p "$TMPDIR"

echo "Mock SLURM environment ready. To test scripts, use:"
echo "  sbatch cluster/bench_distilbert.slurm"
echo "  sbatch cluster/bench_distilbert_cpu.slurm"
echo "  ./cluster/submit_benchmark.sh"
echo ""
echo "To execute scripts (dry run), set:"
echo "  export MOCK_EXECUTE=true"
echo ""
echo "To clean up the mock environment when done:"
echo "  rm -rf $MOCK_DIR $TMPDIR"
echo "  unset SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK TMPDIR"

# End of script
