#!/bin/bash
# Package final project submission

echo "================================================"
echo "CSCE566 Final Project - Submission Packager"
echo "================================================"
echo ""

# Create submission directory
SUBMISSION_DIR="CSCE566_Final_Project_Submission"
rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"

echo "[1/5] Copying report..."
cp FINAL_REPORT.md "$SUBMISSION_DIR/"

echo "[2/5] Copying reflection..."
cp REFLECTION.txt "$SUBMISSION_DIR/"

echo "[3/5] Creating README with GitHub link..."
cat > "$SUBMISSION_DIR/README.txt" << 'EOF'
CSCE566 - Data Mining Final Project
Glaucoma Detection from Clinical Notes using Deep Learning

Student: [Your Name]
Date: November 25, 2025

GitHub Repository: https://github.com/bereket2sh/glaucoma-detection-clinical-notes

Files Included:
1. FINAL_REPORT.md - Complete project report (4 pages)
2. REFLECTION.txt - Project reflection
3. README.txt - This file

Instructions:
- The report is in Markdown format. You can view it in any text editor or Markdown viewer.
- All code is available in the GitHub repository linked above.
- Results and trained models are included in the repository's outputs/ folder.

Project Summary:
- Implemented 3 deep learning models: LSTM, GRU, Transformer
- Best model: GRU with 85.91% AUC
- Comprehensive fairness evaluation across racial groups
- 10,000 clinical notes from FairCLIP dataset
- Achieved balanced performance: 81.92% sensitivity, 72.26% specificity
EOF

echo "[4/5] Copying key results..."
mkdir -p "$SUBMISSION_DIR/results"
cp outputs/model_comparison_table.csv "$SUBMISSION_DIR/results/"
cp outputs/training_summary.csv "$SUBMISSION_DIR/results/"
cp outputs/figures/fairness_comparison.png "$SUBMISSION_DIR/results/" 2>/dev/null || echo "  (fairness visualization not copied - may not exist)"

echo "[5/5] Creating zip file..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ZIP_NAME="CSCE566_Final_Project_${TIMESTAMP}.zip"
zip -r "$ZIP_NAME" "$SUBMISSION_DIR"

echo ""
echo "================================================"
echo "✓ Submission package created successfully!"
echo "================================================"
echo ""
echo "Submission file: $ZIP_NAME"
echo "Contents:"
unzip -l "$ZIP_NAME"
echo ""
echo "To extract: unzip $ZIP_NAME"
echo ""
echo "IMPORTANT: Please update the following before submission:"
echo "  1. Add your name to FINAL_REPORT.md (line 4)"
echo "  2. Add your name to REFLECTION.txt (line 2)"
echo "  3. Update README.txt with your information"
echo ""
echo "Ready to submit!"
