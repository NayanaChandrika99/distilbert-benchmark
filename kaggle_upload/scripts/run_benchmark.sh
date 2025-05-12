#!/bin/bash
# Run benchmark script for DistilBERT benchmarking on HPC cluster
# Usage: ./scripts/run_benchmark.sh --device cpu --batch-sizes 1,2,4,8,16,32,64 --output results/benchmark.jsonl

# Default parameters
DEVICE="cpu"
BATCH_SIZES="1,2,4,8,16,32,64"  # Keeping all batch sizes for comprehensive benchmarking
OUTPUT="results/benchmark.jsonl"
WARMUP_RUNS=5
ITERATIONS=20
MIXED_PRECISION=false
VERBOSE="info"

# Convert arguments with equal signs to space-separated format
args=()
for arg in "$@"; do
  if [[ $arg == *"="* ]]; then
    # Split on first equals sign
    key="${arg%%=*}"
    value="${arg#*=}"
    args+=("$key" "$value")
  else
    args+=("$arg")
  fi
done

# Use the processed arguments
set -- "${args[@]}"

# Parse arguments, forwarding unknown ones to the Python runner
args=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-sizes)
      BATCH_SIZES="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --warmup-runs)
      WARMUP_RUNS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --mixed-precision)
      MIXED_PRECISION=true
      shift
      ;;
    --verbose)
      VERBOSE="$2"
      shift 2
      ;;
    --optimal)
      # New flag for using only the optimal batch size based on our analysis
      if [[ "$DEVICE" == "cpu" ]]; then
        BATCH_SIZES="4"
        echo "Using optimal batch size 4 for CPU"
      else
        BATCH_SIZES="8,16,32"
        echo "Using typical good batch sizes for GPU: 8,16,32"
      fi
      shift
      ;;
    *)
      # Forward any other parameter directly
      args+=("$1")
      shift
      ;;
  esac
done
set -- "${args[@]}"

# Create output directory if it doesn't exist
mkdir -p $(dirname "$OUTPUT")

# Print parameters
echo "Running benchmark with the following parameters:"
echo "  Device: $DEVICE"
echo "  Batch sizes: $BATCH_SIZES"
echo "  Output: $OUTPUT"
echo "  Warmup runs: $WARMUP_RUNS"
echo "  Iterations: $ITERATIONS"
echo "  Mixed precision: $MIXED_PRECISION"

# Build command properly - batch-sizes needs to be passed as a comma-separated string
COMMAND="python src/runner.py --device=$DEVICE --batch-sizes=$BATCH_SIZES --output=$OUTPUT --warmup-runs=$WARMUP_RUNS --iterations=$ITERATIONS --verbose=$VERBOSE"

if [ "$MIXED_PRECISION" = true ] && [[ "$DEVICE" == cuda* ]]; then
  COMMAND="$COMMAND --mixed-precision"
fi

echo "Executing: $COMMAND"
eval $COMMAND

echo "Benchmark completed. Results saved to: $OUTPUT"

# Add recommendation about batch size 4 for CPU
if [[ "$DEVICE" == "cpu" && "$BATCH_SIZES" != "4" ]]; then
  echo ""
  echo "NOTE: Our analysis shows that batch size 4 is optimal for CPU inference with this model."
  echo "To run with just the optimal batch size, use: ./scripts/run_benchmark.sh --device cpu --batch-sizes 4"
  echo "or use the --optimal flag: ./scripts/run_benchmark.sh --device cpu --optimal"
fi 