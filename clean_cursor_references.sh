#!/bin/bash

# Script to clean any remaining Cursor-specific content from files we want to keep

echo "Cleaning Cursor references from files..."

# Clean up scripts/instructions.txt if needed
if [ -f "scripts/instructions.txt" ]; then
  echo "Cleaning scripts/instructions.txt..."
  grep -v "cursor\|Cursor\|task-master\|taskmaster" scripts/instructions.txt > scripts/clean_instructions.txt
  mv scripts/clean_instructions.txt scripts/instructions.txt
fi

# Remove README-task-master.md
if [ -f "scripts/README-task-master.md" ]; then
  echo "Removing scripts/README-task-master.md..."
  rm scripts/README-task-master.md
fi

# Create empty directories for git
echo "Creating empty directory placeholders for git..."
mkdir -p results
touch results/.gitkeep

# Clean up any remaining dev.js files (task-master related)
if [ -f "scripts/dev.js" ]; then
  echo "Removing scripts/dev.js..."
  rm scripts/dev.js
fi

# Clean up any references to task-master in scripts we want to keep
echo "Cleaning task-master references from utility scripts..."
for script in scripts/list_components.sh scripts/profile_batch_sizes.py scripts/run_benchmark.sh scripts/prepare_hpc_deployment.sh
do
  if [ -f "$script" ]; then
    echo "Cleaning $script..."
    sed -i '' -e 's/task-master/benchmark-tool/g' -e 's/taskmaster/benchmark-tool/g' -e 's/Task Master/Benchmark Tool/g' "$script"
  fi
done

echo "Cleaning complete!" 