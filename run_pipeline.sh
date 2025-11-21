#!/bin/bash
# Run the complete glaucoma detection pipeline

echo "========================================"
echo "Glaucoma Detection Pipeline"
echo "========================================"

# Change to src directory
cd src

# Step 1: Data Preprocessing
echo ""
echo "[1/3] Running data preprocessing..."
python data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Error in data preprocessing!"
    exit 1
fi

# Step 2: Model Training
echo ""
echo "[2/3] Training models..."
python train.py
if [ $? -ne 0 ]; then
    echo "Error in model training!"
    exit 1
fi

# Step 3: Fairness Evaluation
echo ""
echo "[3/3] Evaluating fairness..."
python evaluate_fairness.py
if [ $? -ne 0 ]; then
    echo "Error in fairness evaluation!"
    exit 1
fi

cd ..

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Results can be found in:"
echo "  - outputs/models/        (trained models)"
echo "  - outputs/figures/       (visualizations)"
echo "  - outputs/               (metrics and summaries)"
