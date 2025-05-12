#!/usr/bin/env python3
"""
Run mixed precision benchmarks for DistilBERT with larger batch sizes.
This script automates the execution of benchmarks for batch sizes 64 and 128
with mixed precision enabled.
"""

import os
import subprocess
import argparse
import json
from datetime import datetime

# Default settings
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_OUTPUT_DIR = "distilbert_outputs"
BATCH_SIZES = [64, 128]

def run_benchmark(model_name, batch_size, output_dir, device="cuda", mixed_precision=True):
    """Run a single benchmark configuration."""
    cmd = [
        "python", "distilbert_benchmarking/src/main.py",
        "--model-name", model_name,
        "--device", device,
        "--batch-sizes", str(batch_size),
        "--output-file", os.path.join(output_dir, "mixed_precision_results.jsonl")
    ]
    
    if mixed_precision:
        cmd.append("--mixed-precision")
    
    print(f"\n{'='*80}")
    print(f"Running benchmark with batch size {batch_size} on {device} with mixed_precision={mixed_precision}")
    print(f"{'='*80}\n")
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()
    
    print(f"\nCompleted in {(end_time - start_time).total_seconds():.2f} seconds")
    
    if result.returncode != 0:
        print(f"ERROR: Benchmark failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"Benchmark completed successfully")
    return True

def verify_output_file(output_dir, batch_size, mixed_precision=True):
    """Verify that the output file was created."""
    # Determine the expected filename pattern based on configuration
    expected_file = os.path.join(output_dir, f"mixed_precision_results.jsonl")
    
    if not os.path.exists(expected_file):
        print(f"WARNING: Expected output file {expected_file} not found")
        return False
    
    # Verify the file has the expected batch size entry
    found = False
    try:
        with open(expected_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('batch_size') == batch_size and data.get('mixed_precision', False) == mixed_precision:
                    found = True
                    break
    except Exception as e:
        print(f"ERROR: Failed to read {expected_file}: {str(e)}")
        return False
    
    if not found:
        print(f"WARNING: No entry found for batch size {batch_size} with mixed_precision={mixed_precision}")
        return False
    
    print(f"Successfully verified output for batch size {batch_size}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run mixed precision benchmarks for DistilBERT")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to benchmark (default: {DEFAULT_MODEL})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES, 
                        help=f"Batch sizes to benchmark (default: {BATCH_SIZES})")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting mixed precision benchmark sweep for {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Track success/failure
    results = []
    
    # Run benchmarks for each batch size
    for batch_size in args.batch_sizes:
        success = run_benchmark(
            model_name=args.model,
            batch_size=batch_size,
            output_dir=args.output_dir,
            device=args.device,
            mixed_precision=True
        )
        
        if success:
            verified = verify_output_file(
                output_dir=args.output_dir,
                batch_size=batch_size,
                mixed_precision=True
            )
        else:
            verified = False
        
        results.append({
            "batch_size": batch_size,
            "success": success,
            "verified": verified
        })
    
    # Print summary
    print("\n\n" + "="*40)
    print("BENCHMARK SWEEP SUMMARY")
    print("="*40)
    
    all_success = True
    for result in results:
        status = "✅ SUCCESS" if result["success"] and result["verified"] else "❌ FAILED"
        all_success = all_success and result["success"] and result["verified"]
        print(f"Batch size {result['batch_size']}: {status}")
    
    print("\nNext steps:")
    if all_success:
        print("1. Run the generate_comparison_plots.py script to update plots with new data")
        print("2. Analyze the results to identify optimal configurations")
    else:
        print("1. Review logs to identify and fix benchmark failures")
        print("2. Re-run failed configurations")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 