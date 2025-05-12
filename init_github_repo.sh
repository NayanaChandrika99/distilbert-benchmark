#!/bin/bash

# Script to initialize a Git repository and prepare for GitHub

echo "Setting up Git repository for GitHub..."

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
  echo "Initializing Git repository..."
  git init
else
  echo "Git repository already initialized."
fi

# Add all files to Git
echo "Adding files to Git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: DistilBERT Benchmark Suite with Kaggle deployment focus"

echo ""
echo "Repository is ready for GitHub!"
echo ""
echo "To push to GitHub, follow these steps:"
echo ""
echo "1. Create a new repository on GitHub (do not initialize with README, .gitignore, or license)"
echo "2. Run the following commands:"
echo "   git remote add origin https://github.com/yourusername/your-repo-name.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Replace 'yourusername' and 'your-repo-name' with your GitHub username and repository name." 