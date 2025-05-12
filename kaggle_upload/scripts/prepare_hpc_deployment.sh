#!/bin/bash
# Script to prepare the DistilBERT benchmarking codebase for HPC deployment

set -e  # Exit on any error

# Configuration
PACKAGE_NAME="distilbert-benchmark"
TARGET_DIR="deployment"
VERSION=$(date +%Y%m%d)
PACKAGE_FILE="${PACKAGE_NAME}-${VERSION}.tar.gz"

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Preparing DistilBERT Benchmarking for HPC deployment...${NC}"

# Create a clean target directory
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Essential directories to include
echo -e "${GREEN}Copying essential files...${NC}"
cp -r src/ "$TARGET_DIR/"
cp -r data/ "$TARGET_DIR/"
cp -r cluster/ "$TARGET_DIR/"
cp -r analysis/ "$TARGET_DIR/"
cp -r scripts/ "$TARGET_DIR/"
mkdir -p "$TARGET_DIR/results"

# Essential files
cp config.yaml "$TARGET_DIR/"
cp environment.yml "$TARGET_DIR/"
cp README.md "$TARGET_DIR/"
cp DistilBERT-README.md "$TARGET_DIR/"

# Remove any cached files
find "$TARGET_DIR" -name "__pycache__" -type d -exec rm -rf {} +  2>/dev/null || true
find "$TARGET_DIR" -name "*.pyc" -delete
find "$TARGET_DIR" -name ".DS_Store" -delete

# Create a README specifically for HPC deployment
cat > "$TARGET_DIR/HPC-README.md" << 'EOF'
# DistilBERT Benchmarking HPC Deployment

This package contains the essential files for running DistilBERT benchmarking on an HPC cluster.

## Quick Start

1. **Set up the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate distilbert-benchmark
   ```

2. **Submit benchmark jobs:**
   ```bash
   cd cluster
   ./submit_benchmark.sh --device both --analyze
   ```

## Directory Structure

- `src/` - Core implementation code
- `data/` - Dataset utilities
- `analysis/` - Analysis and visualization scripts
- `cluster/` - SLURM job scripts
- `scripts/` - Utility scripts
- `results/` - Directory for benchmark results

## Important Files

- `config.yaml` - Default configuration
- `environment.yml` - Conda environment definition
- `cluster/submit_benchmark.sh` - Main job submission script
- `cluster/README.md` - Detailed HPC usage documentation

## Optimal Settings

Based on our analysis, we recommend:
- For CPU: Use batch size 4 (--batch-sizes "4" or --optimal flag)
- For GPU: Test multiple batch sizes (--batch-sizes "4,8,16,32,64")

See `cluster/README.md` for complete documentation on available options.
EOF

# Create a compressed archive
echo -e "${GREEN}Creating deployment package...${NC}"
tar -czf "$PACKAGE_FILE" "$TARGET_DIR"

echo -e "${GREEN}Deployment package created: ${PACKAGE_FILE}${NC}"
echo "To deploy to HPC:"
echo "1. Copy $PACKAGE_FILE to your HPC environment"
echo "2. Extract with: tar -xzf $PACKAGE_FILE"
echo "3. Follow instructions in $TARGET_DIR/HPC-README.md" 