#!/bin/bash

# Script to organize the repository for GitHub without deleting files
# We'll move HPC-related files to a separate folder and gitignore them

echo "Organizing repository for GitHub (Kaggle-focused)..."

# First run the cursor reference cleanup
if [ -f "clean_cursor_references.sh" ]; then
  echo "Running cursor reference cleanup first..."
  ./clean_cursor_references.sh
fi

# Create a directory for HPC-related files
echo "Creating directory for HPC-related files..."
mkdir -p hpc_related_files

# Move HPC-related directories to hpc_related_files instead of deleting
echo "Moving HPC-related directories to hpc_related_files/..."
if [ -d "hpc" ]; then
  mv hpc hpc_related_files/
fi
if [ -d "cluster" ]; then
  mv cluster hpc_related_files/
fi
if [ -d "deployment" ]; then
  mv deployment hpc_related_files/
fi

# Move HPC-related scripts to hpc_related_files
echo "Moving HPC-related scripts to hpc_related_files/..."
if [ -f "scripts/prepare_hpc_deployment.sh" ]; then
  mv scripts/prepare_hpc_deployment.sh hpc_related_files/
fi
if [ -f "scripts/improved_prepare_zaratan.sh" ]; then
  mv scripts/improved_prepare_zaratan.sh hpc_related_files/
fi
if [ -f "scripts/prepare_zaratan_deployment.sh" ]; then
  mv scripts/prepare_zaratan_deployment.sh hpc_related_files/
fi
if [ -f "prepare_zaratan_deployment.sh" ]; then
  mv prepare_zaratan_deployment.sh hpc_related_files/
fi
if [ -f "improved_prepare_zaratan.sh" ]; then
  mv improved_prepare_zaratan.sh hpc_related_files/
fi

# Update README.md to focus on Kaggle deployment
echo "Updating README.md to focus on Kaggle deployment..."
if [ -f "README.md" ]; then
  # Create a backup
  cp README.md README.md.bak
  
  # Instead of removing HPC sections, add a note about HPC files location
  sed -i '' '/## HPC Cluster Integration/a\\nNote: HPC-related files are stored in the `hpc_related_files` directory but not included in the GitHub repository as the project was primarily deployed on Kaggle.\n' README.md
  
  # Add a note about Kaggle being the primary deployment platform
  sed -i '' 's/HPC Integration: Includes SLURM scripts for high-performance computing clusters/Kaggle Integration: Primarily deployed on Kaggle with notebooks included/g' README.md
fi

# Update .gitignore to exclude the hpc_related_files directory
echo "Updating .gitignore to exclude hpc_related_files/..."
if ! grep -q "hpc_related_files/" .gitignore; then
  echo "" >> .gitignore
  echo "# HPC related files (not used in final deployment)" >> .gitignore
  echo "hpc_related_files/" >> .gitignore
fi

# Create a README in the hpc_related_files directory
echo "Creating README for hpc_related_files/..."
cat > hpc_related_files/README.md << 'EOF'
# HPC-Related Files

This directory contains files related to High-Performance Computing (HPC) deployment of the DistilBERT benchmark suite. These files are preserved for reference but were not used in the final project deployment, which primarily focused on Kaggle.

## Contents
- HPC deployment scripts
- SLURM job scripts
- Cluster configuration files

**Note**: This directory is excluded from the GitHub repository via .gitignore.
EOF

# Create a Kaggle-focused README for scripts
echo "Creating Kaggle-focused README for scripts/..."
cat > scripts/README.md << 'EOF'
# Scripts

This directory contains utility scripts for running the DistilBERT benchmark suite with a focus on Kaggle deployment.

## Files:
- `list_components.sh`: Lists components of the project
- `profile_batch_sizes.py`: Script for profiling different batch sizes
- `run_benchmark.sh`: Script for running benchmarks locally before deploying to Kaggle
EOF

echo "Organization complete! Repository now focuses on Kaggle deployment, with HPC files preserved in hpc_related_files/ but excluded from GitHub." 