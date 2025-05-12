#!/bin/bash

# Script to organize the codebase structure before uploading to GitHub

echo "Organizing codebase structure..."

# Create necessary directories if they don't exist
mkdir -p src/analysis
mkdir -p src/benchmarking
mkdir -p src/utils
mkdir -p src/visualization
mkdir -p scripts
mkdir -p docs
mkdir -p config
mkdir -p results

# Move analysis scripts to src/analysis
echo "Moving analysis scripts..."
mv analyze_nested_metrics.py src/analysis/ 2>/dev/null
mv analyze_benchmark_results.py src/analysis/ 2>/dev/null
mv generate_comprehensive_report.py src/analysis/ 2>/dev/null

# Move benchmarking scripts to src/benchmarking
echo "Moving benchmarking scripts..."
mv run_mixed_precision_sweep.py src/benchmarking/ 2>/dev/null
mv model_profiler.py src/benchmarking/ 2>/dev/null
mv perf_diagnostics.py src/benchmarking/ 2>/dev/null
mv smoke_test.py src/benchmarking/ 2>/dev/null
mv test_model.py src/benchmarking/ 2>/dev/null

# Move visualization scripts to src/visualization
echo "Moving visualization scripts..."
mv generate_comparison_plots.py src/visualization/ 2>/dev/null

# Move utility scripts to src/utils
echo "Moving utility scripts..."
mv list_components.sh scripts/ 2>/dev/null
mv improved_prepare_zaratan.sh scripts/ 2>/dev/null
mv prepare_zaratan_deployment.sh scripts/ 2>/dev/null

# Move configuration files to config
echo "Moving configuration files..."
mv config.yaml config/ 2>/dev/null
mv environment.yml config/ 2>/dev/null
mv mypy.ini config/ 2>/dev/null
mv .pre-commit-config.yaml config/ 2>/dev/null

# Move documentation to docs
echo "Moving documentation..."
mv DistilBERT-README.md docs/README.md 2>/dev/null
mv KAGGLE_INSTRUCTIONS.md docs/ 2>/dev/null

# Move result files to results directory
echo "Moving result files..."
mv *_results.jsonl results/ 2>/dev/null
mv batch_comparison.jsonl results/ 2>/dev/null
mv absolute_results.jsonl results/ 2>/dev/null

# Clean up large files and archives
echo "Cleaning up large files and archives..."
mkdir -p archive/backups
mv *.tar.gz archive/backups/ 2>/dev/null
mv *.zip archive/backups/ 2>/dev/null

# Update imports in Python files to reflect new structure
echo "Updating imports in Python files..."
find src -name "*.py" -type f -exec sed -i '' 's/from analysis\./from src.analysis./g' {} \;
find src -name "*.py" -type f -exec sed -i '' 's/import analysis\./import src.analysis./g' {} \;

# Create a basic README for each directory
echo "Creating README files for directories..."

# src/analysis README
cat > src/analysis/README.md << 'EOF'
# Analysis Module

This directory contains scripts for analyzing benchmark results and generating reports.

## Files:
- `analyze_nested_metrics.py`: Analyzes nested metrics from benchmark results
- `analyze_benchmark_results.py`: Processes and analyzes benchmark results
- `generate_comprehensive_report.py`: Creates comprehensive reports from benchmark data
EOF

# src/benchmarking README
cat > src/benchmarking/README.md << 'EOF'
# Benchmarking Module

This directory contains scripts for running benchmarks and performance tests.

## Files:
- `run_mixed_precision_sweep.py`: Runs mixed precision benchmarks
- `model_profiler.py`: Profiles model performance
- `perf_diagnostics.py`: Diagnoses performance issues
- `smoke_test.py`: Quick validation test
- `test_model.py`: Model testing utilities
EOF

# src/visualization README
cat > src/visualization/README.md << 'EOF'
# Visualization Module

This directory contains scripts for visualizing benchmark results.

## Files:
- `generate_comparison_plots.py`: Creates comparison plots from benchmark results
EOF

# src/utils README
cat > src/utils/README.md << 'EOF'
# Utilities Module

This directory contains utility scripts and helper functions.
EOF

# scripts README
cat > scripts/README.md << 'EOF'
# Scripts

This directory contains utility scripts for deployment and maintenance.

## Files:
- `list_components.sh`: Lists components of the project
- `improved_prepare_zaratan.sh`: Improved script for Zaratan deployment
- `prepare_zaratan_deployment.sh`: Script for Zaratan deployment
EOF

# docs README
cat > docs/README.md << 'EOF'
# Documentation

This directory contains project documentation.

## Files:
- `KAGGLE_INSTRUCTIONS.md`: Instructions for running benchmarks on Kaggle
EOF

# config README
cat > config/README.md << 'EOF'
# Configuration

This directory contains configuration files for the project.

## Files:
- `config.yaml`: Main configuration file
- `environment.yml`: Conda environment configuration
- `mypy.ini`: MyPy type checking configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
EOF

# results README
cat > results/README.md << 'EOF'
# Results

This directory contains benchmark results.

## Files:
- Various `*_results.jsonl` files containing benchmark results
- `batch_comparison.jsonl`: Batch size comparison results
- `absolute_results.jsonl`: Reference benchmark results
EOF

echo "Codebase organization complete!"
echo "Please review the changes before proceeding with GitHub upload." 