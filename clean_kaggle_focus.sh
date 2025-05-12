#!/bin/bash

# Script to clean the repository focusing only on Kaggle deployment

echo "Cleaning repository to focus on Kaggle deployment only..."

# First run the cursor reference cleanup
if [ -f "clean_cursor_references.sh" ]; then
  echo "Running cursor reference cleanup first..."
  ./clean_cursor_references.sh
fi

# Remove HPC-related directories
echo "Removing HPC-related directories..."
rm -rf hpc/
rm -rf cluster/
rm -rf deployment/

# Remove HPC-related scripts
echo "Removing HPC-related scripts..."
rm -f scripts/prepare_hpc_deployment.sh
rm -f scripts/improved_prepare_zaratan.sh
rm -f scripts/prepare_zaratan_deployment.sh
rm -f prepare_zaratan_deployment.sh
rm -f improved_prepare_zaratan.sh

# Update README.md to remove HPC mentions
echo "Updating README.md to focus on Kaggle deployment..."
if [ -f "README.md" ]; then
  # Create a backup
  cp README.md README.md.bak
  
  # Remove HPC sections from README
  sed -i '' '/## HPC Cluster Integration/,/^##/d' README.md
  sed -i '' 's/HPC Integration: Includes SLURM scripts for high-performance computing clusters/Kaggle Integration: Includes notebooks for running on Kaggle/g' README.md
  
  # Update the overview to focus on Kaggle
  sed -i '' 's/across various hardware configurations/on Kaggle/g' README.md
  
  # Update any mentions of HPC to Kaggle where appropriate
  sed -i '' 's/high-performance computing clusters/Kaggle notebooks/g' README.md
  sed -i '' 's/HPC systems/Kaggle/g' README.md
fi

# Update any import statements in src files to reflect removal of HPC components
echo "Updating import statements in Python files..."
find src -name "*.py" -type f -exec grep -l "import.*hpc" {} \; | xargs -I{} sed -i '' 's/import.*hpc//g' {} 2>/dev/null
find src -name "*.py" -type f -exec grep -l "from.*hpc" {} \; | xargs -I{} sed -i '' 's/from.*hpc.*import//g' {} 2>/dev/null

# Clean up any other references to HPC
echo "Cleaning up any other HPC references..."
find src docs config -type f -name "*.py" -o -name "*.md" | xargs grep -l "hpc\|zaratan\|slurm" 2>/dev/null | xargs -I{} sed -i '' 's/hpc integration/Kaggle integration/g; s/HPC integration/Kaggle integration/g; s/HPC systems/Kaggle/g; s/slurm/Kaggle notebook/g; s/Zaratan/Kaggle/g; s/zaratan/Kaggle/g' {} 2>/dev/null

# Create readme files to explain the focus on Kaggle
echo "Creating explanatory README files..."

# Create a Kaggle-focused README for scripts
cat > scripts/README.md << 'EOF'
# Scripts

This directory contains utility scripts for running the DistilBERT benchmark suite on Kaggle.

## Files:
- `list_components.sh`: Lists components of the project
- `profile_batch_sizes.py`: Script for profiling different batch sizes
- `run_benchmark.sh`: Script for running benchmarks locally before deploying to Kaggle
EOF

echo "Cleaning complete! Repository now focuses on Kaggle deployment only." 