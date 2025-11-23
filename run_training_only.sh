#!/bin/bash
# Run training only (skip preprocessing since data is already processed)

echo "========================================"
echo "Glaucoma Detection - Training Only"
echo "========================================"

# Change to src directory
cd src

# Model Training
echo ""
echo "Training models..."
python train.py
if [ $? -ne 0 ]; then
    echo "Error in model training!"
    exit 1
fi

# Fairness Evaluation
echo ""
echo "Evaluating fairness..."
python evaluate_fairness.py
if [ $? -ne 0 ]; then
    echo "Error in fairness evaluation!"
    exit 1
fi

cd ..

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "========================================"
echo ""
echo "Results can be found in:"
echo "  - outputs/models/        (trained models)"
echo "  - outputs/figures/       (visualizations)"
echo "  - outputs/               (metrics and summaries)"
