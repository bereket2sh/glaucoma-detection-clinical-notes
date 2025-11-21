#!/bin/bash
# GitHub Repository Setup Script

echo "========================================================================"
echo "GITHUB REPOSITORY SETUP"
echo "========================================================================"

# Repository name
REPO_NAME="glaucoma-detection-clinical-notes"

echo ""
echo "Recommended repository name: $REPO_NAME"
echo ""
echo "Steps to create GitHub repository:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: $REPO_NAME"
echo "3. Description: Deep learning for glaucoma detection from clinical notes with fairness evaluation"
echo "4. Make it: Public (recommended) or Private"
echo "5. DO NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
echo "Then run this script or the commands below:"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Use the GitHub README
if [ -f "README_GITHUB.md" ]; then
    echo "Using GitHub-specific README..."
    cp README_GITHUB.md README.md
    echo "✓ README updated for GitHub"
fi

# Add all files
echo ""
echo "Adding files to git..."
git add .gitignore
git add requirements.txt
git add README.md
git add models.py
git add train_all_models.py
git add 5_fairness_evaluation.py
git add 2_data_preprocessing.py
git add 1_data_exploration.py
git add *.py
git add *.md
echo "✓ Files added"

# First commit
echo ""
echo "Creating initial commit..."
git commit -m "Initial commit: Glaucoma detection project with fairness evaluation"
echo "✓ Commit created"

# Rename branch to main
git branch -M main
echo "✓ Branch renamed to main"

echo ""
echo "========================================================================"
echo "NEXT STEPS:"
echo "========================================================================"
echo ""
echo "1. After creating the repository on GitHub, copy your repository URL"
echo "   Example: https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo ""
echo "2. Run these commands (replace YOUR_USERNAME):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "3. Your code will be uploaded to GitHub!"
echo ""
echo "========================================================================"
echo ""
echo "Optional: To include trained models and results (if not too large):"
echo "   git add best_*.pt"
echo "   git add *_results.json"
echo "   git add *.png"
echo "   git commit -m 'Add trained models and results'"
echo "   git push"
echo ""
echo "========================================================================"
