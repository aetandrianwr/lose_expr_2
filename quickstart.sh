#!/usr/bin/env bash
# Quick start script for running experiments

set -e  # Exit on error

echo "=========================================="
echo "Next-Location Prediction - Quick Start"
echo "=========================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import torch, yaml, numpy" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

echo "✓ Dependencies OK"
echo

# Run verification
echo "Running framework verification..."
python3 verify_framework.py || {
    echo "✗ Framework verification failed"
    exit 1
}

echo
echo "=========================================="
echo "Select an experiment to run:"
echo "=========================================="
echo "1) Train on Geolife dataset (default config)"
echo "2) Train on DIY dataset"
echo "3) Run unit tests"
echo "4) Quick test run (10 epochs)"
echo "5) Exit"
echo

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Training on Geolife dataset..."
        python3 src/train_production.py --config configs/default.yaml
        ;;
    2)
        echo "Training on DIY dataset..."
        python3 src/train_production.py --config configs/diy_experiment.yaml
        ;;
    3)
        echo "Running unit tests..."
        python3 tests/test_metrics.py
        python3 tests/test_reproducibility.py
        python3 tests/test_dataset.py
        echo "✓ All tests passed!"
        ;;
    4)
        echo "Quick test run (10 epochs on Geolife)..."
        python3 src/train_production.py \
            --config configs/default.yaml \
            --epochs 10 \
            --batch_size 128
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir experiments/"
echo
echo "To see results:"
echo "  ls -lh experiments/"
echo
