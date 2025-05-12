#!/bin/bash
# DistilBERT Benchmarking Project Component List
# Run this script to get an overview of the project components

echo "======================================================================"
echo "                DistilBERT Benchmarking Project                       "
echo "======================================================================"
echo

# Function to print directory info if it exists
print_dir_info() {
  local dir=$1
  local description=$2
  
  if [ -d "$dir" ]; then
    echo "📁 $dir"
    echo "   $description"
    echo "   Files: $(find "$dir" -type f | wc -l)"
    echo
  fi
}

# Print key directories
print_dir_info "analysis" "Scripts for analyzing and visualizing benchmark results"
print_dir_info "distilbert_benchmarking" "Core benchmarking implementation"
print_dir_info "distilbert_outputs" "Benchmark results and generated reports"
print_dir_info "hpc" "Scripts for deploying and running benchmarks on HPC systems"
print_dir_info "kaggle" "Scripts and notebooks for running benchmarks on Kaggle"

echo "======================================================================"
echo "                           Results                                    "
echo "======================================================================"
echo

# Print key result files
if [ -d "distilbert_outputs" ]; then
  echo "Key result files:"
  for file in distilbert_outputs/*.jsonl; do
    if [ -f "$file" ]; then
      base=$(basename "$file")
      size=$(du -h "$file" | cut -f1)
      echo "📄 $base ($size)"
    fi
  done
  
  echo
  echo "Visualization directories:"
  for dir in distilbert_outputs/*/; do
    if [ -d "$dir" ]; then
      base=$(basename "$dir")
      count=$(find "$dir" -type f -name "*.png" | wc -l)
      echo "📊 $base ($count plots)"
    fi
  done
fi

echo
echo "======================================================================"
echo "                          Documentation                               "
echo "======================================================================"
echo

# Print README files
echo "Documentation files:"
find . -name "README.md" -maxdepth 2 | while read -r readme; do
  dir=$(dirname "$readme")
  if [ "$dir" = "." ]; then
    echo "📑 Main README.md - Project overview"
  else
    echo "📑 $dir/README.md - $(echo "$dir" | sed 's/\.\///') documentation"
  fi
done

echo
echo "Run 'python analysis/generate_comparison_plots.py --help' for plotting options"
echo "Run 'python analysis/generate_comprehensive_report.py --help' for report generation options"
echo 