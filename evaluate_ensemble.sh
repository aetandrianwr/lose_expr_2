#!/bin/bash

################################################################################
# Ensemble Evaluation - Shell Script Wrapper
#
# Evaluates ensemble of Model 1 and Frequency-Aware Model
# Target: 37.24% test Acc@1
################################################################################

echo "=========================================="
echo "Ensemble Evaluation"
echo "=========================================="
echo ""
echo "This will evaluate an ensemble of:"
echo "  - Model 1 (Temporal Fusion)"
echo "  - Frequency-Aware Model"
echo ""
echo "Expected result: ~37.24% test Acc@1"
echo ""

# Check if checkpoints exist
if [ ! -f "./checkpoints/best_model.pt" ]; then
    echo "Error: Model 1 checkpoint not found at ./checkpoints/best_model.pt"
    echo "Please train Model 1 first using: bash train_model1.sh"
    exit 1
fi

if [ ! -f "./checkpoints_freq/best_model.pt" ]; then
    echo "Error: Frequency-Aware checkpoint not found at ./checkpoints_freq/best_model.pt"
    echo "Please train Frequency-Aware model first using: bash train_freq.sh"
    exit 1
fi

echo "Both checkpoints found. Starting ensemble evaluation..."
echo ""

# Run ensemble evaluation
python src/evaluate_ensemble.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Ensemble evaluation completed successfully!"
else
    echo ""
    echo "Ensemble evaluation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
