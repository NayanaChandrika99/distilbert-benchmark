#!/bin/bash

# Script to clean the repository of Cursor-specific files before uploading to GitHub

echo "Cleaning repository of Cursor-specific files..."

# Remove Cursor-specific directories
echo "Removing .cursor directory..."
rm -rf .cursor/

# Remove Cursor-specific files
echo "Removing Cursor rule files..."
rm -f .cursorrules .windsurfrules

# Remove Cursor logs
echo "Removing Cursor logs..."
rm -f logs/cursor_edits.log

# Remove task-related files
echo "Removing task-related files..."
rm -rf tasks/
rm -f tasks.json

# Remove package files related to Cursor/task-master
echo "Removing package files related to Cursor/task-master..."
rm -f package.json package-lock.json

# Remove Cursor-specific entries from README.md
echo "Creating clean README.md without Cursor references..."
if [ -f README.md ]; then
  # Create a backup
  cp README.md README.md.bak
  
  # Filter out any lines with Cursor references
  grep -v -i "cursor\|task-master\|taskmaster" README.md > README.md.clean
  mv README.md.clean README.md
fi

# Remove manifest.json which may contain Cursor-specific information
echo "Removing manifest.json..."
rm -f manifest.json

echo "Clean-up complete! The repository is now ready for GitHub."
echo "Please review the changes before pushing to GitHub." 