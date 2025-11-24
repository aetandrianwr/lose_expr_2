# Training Scripts Guide

This guide explains how to use the configurable training scripts to reproduce the best results.

## Quick Start

### 1. Train Model 1 (Best Single Model: ~37% test Acc@1)

```bash
chmod +x train_model1.sh
bash train_model1.sh
```

### 2. Train Frequency-Aware Model (~36% test Acc@1)

```bash
chmod +x train_freq.sh
bash train_freq.sh
```

### 3. Evaluate Ensemble (Best Result: ~37.24% test Acc@1)

```bash
chmod +x evaluate_ensemble.sh
bash evaluate_ensemble.sh
```

## Architecture Overview

The new training system follows the standard approach:

```
┌─────────────────────┐
│   Shell Script      │  ← Edit parameters here
│   (train_model1.sh) │
└──────────┬──────────┘
           │ Passes arguments
           ↓
┌─────────────────────┐
│   Python Script     │  ← All logic here
│   (train_*.py)      │
└──────────┬──────────┘
           │ Prints config & trains
           ↓
┌─────────────────────┐
│   Checkpoint        │
│   (best_model.pt)   │
└─────────────────────┘
```

## Files Overview

### Shell Scripts (Easy to edit)
- `train_model1.sh` - Train Temporal Fusion Model (Model 1)
- `train_freq.sh` - Train Frequency-Aware Model
- `train_model1_custom_example.sh` - Examples of custom configurations
- `evaluate_ensemble.sh` - Evaluate ensemble of two models

### Python Scripts (Contains all logic)
- `src/train_model1_configurable.py` - Model 1 training with argparse
- `src/train_freq_configurable.py` - Freq-Aware training with argparse
- `src/evaluate_ensemble.py` - Ensemble evaluation

## Configuration Parameters

### Dataset Parameters
```bash
DATASET="geolife"           # Dataset name (geolife or diy)
TRAIN_PATH="path/to/train"  # Path to training data
VAL_PATH="path/to/val"      # Path to validation data
TEST_PATH="path/to/test"    # Path to test data
BATCH_SIZE=128              # Batch size
MAX_SEQ_LEN=50              # Maximum sequence length
```

### Model Architecture (Model 1)
```bash
D_MODEL=96                  # Model dimension (64, 96, 128, 256)
NUM_LAYERS=2                # Number of temporal layers (2, 3, 4)
NUM_HEADS=4                 # Number of attention heads (4, 8, 16)
KERNEL_SIZE=3               # Convolution kernel size (3, 5, 7)
DROPOUT=0.2                 # Dropout rate (0.1 - 0.4)
```

### Model Architecture (Frequency-Aware)
```bash
D_MODEL=96                  # Model dimension (64, 96, 128, 256)
NUM_HEADS=4                 # Number of attention heads (4, 8, 16)
DROPOUT=0.25                # Dropout rate (0.1 - 0.4)
```

### Training Hyperparameters
```bash
EPOCHS=100                  # Number of training epochs
LR=0.001                    # Learning rate (0.0001 - 0.01)
WEIGHT_DECAY=0.0001         # Weight decay / L2 regularization
GRAD_CLIP=1.0               # Gradient clipping value
LABEL_SMOOTHING=0.1         # Label smoothing (0.0 - 0.2)
```

### Learning Rate Scheduler
```bash
SCHEDULER="onecycle"        # Options: onecycle, cosine, step
PCT_START=0.1               # Warmup percentage (for OneCycleLR)
```

### Early Stopping
```bash
PATIENCE=15                 # Early stopping patience (epochs)
MONITOR="val_acc"           # Metric to monitor: val_acc or val_loss
```

**Important:** Choose whether to save the best model based on:
- `val_acc` - Higher validation accuracy (default, recommended)
- `val_loss` - Lower validation loss

### Output Configuration
```bash
CHECKPOINT_DIR="./checkpoints"  # Where to save checkpoints
NAME="model1"                   # Experiment name
```

### Other Parameters
```bash
SEED=42                     # Random seed for reproducibility
DEVICE="cuda"               # Device: cuda or cpu
```

## Usage Examples

### Example 1: Default Training (Reproduces best result)

```bash
bash train_model1.sh
```

This uses the proven configuration that achieved 36.98% test Acc@1.

### Example 2: Quick Experiment (Faster training)

Edit `train_model1.sh`:
```bash
EPOCHS=50
BATCH_SIZE=256
D_MODEL=64
```

Then run:
```bash
bash train_model1.sh
```

### Example 3: Larger Model with More Regularization

Edit `train_model1.sh`:
```bash
D_MODEL=128
NUM_LAYERS=3
NUM_HEADS=8
DROPOUT=0.3
LABEL_SMOOTHING=0.15
CHECKPOINT_DIR="./checkpoints_large"
```

### Example 4: Different Learning Rate Schedule

Edit `train_model1.sh`:
```bash
SCHEDULER="cosine"
LR=0.0005
EPOCHS=150
```

### Example 5: Monitor val_loss instead of val_acc

Edit `train_model1.sh`:
```bash
MONITOR="val_loss"
PATIENCE=20
```

### Example 6: Different Random Seed

```bash
SEED=123
CHECKPOINT_DIR="./checkpoints_seed123"
```

### Example 7: Use DIY Dataset

```bash
DATASET="diy"
TRAIN_PATH="/content/lose_expr_2/data/diy/diy_transformer_7_train.pk"
VAL_PATH="/content/lose_expr_2/data/diy/diy_transformer_7_validation.pk"
TEST_PATH="/content/lose_expr_2/data/diy/diy_transformer_7_test.pk"
```

## Direct Python Usage

You can also call the Python scripts directly with arguments:

```bash
python src/train_model1_configurable.py \
    --d-model 128 \
    --num-layers 3 \
    --num-heads 8 \
    --dropout 0.3 \
    --epochs 100 \
    --lr 0.001 \
    --monitor val_acc \
    --checkpoint-dir ./checkpoints_custom \
    --name my_experiment
```

### View All Available Arguments

```bash
python src/train_model1_configurable.py --help
python src/train_freq_configurable.py --help
```

## Configuration Printing

All Python scripts automatically print the full configuration at the start of training:

```
================================================================================
CONFIGURATION
================================================================================

Dataset:
  Dataset name:       geolife
  Train path:         /content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk
  Val path:           /content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk
  Test path:          /content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk
  Batch size:         128
  Max sequence len:   50

Model Architecture:
  Model type:         Temporal Fusion
  d_model:            96
  num_layers:         2
  num_heads:          4
  kernel_size:        3
  dropout:            0.2

Training:
  Epochs:             100
  Learning rate:      0.001
  Weight decay:       0.0001
  Grad clip:          1.0
  Label smoothing:    0.1
  Scheduler:          onecycle
  PCT start:          0.1

Early Stopping:
  Patience:           15
  Monitor metric:     val_acc

Output:
  Checkpoint dir:     ./checkpoints
  Experiment name:    model1

Other:
  Random seed:        42
  Device:             cuda

================================================================================
```

## Reproducing Best Result (37.24% Test Acc@1)

Follow these steps:

### Step 1: Train Model 1
```bash
bash train_model1.sh
```
Expected: ~37% test Acc@1, checkpoint saved to `./checkpoints/best_model.pt`

### Step 2: Train Frequency-Aware Model
```bash
bash train_freq.sh
```
Expected: ~36% test Acc@1, checkpoint saved to `./checkpoints_freq/best_model.pt`

### Step 3: Evaluate Ensemble
```bash
bash evaluate_ensemble.sh
```
Expected: ~37.24% test Acc@1 (ensemble with weights [0.7, 0.3])

## Tips and Best Practices

### 1. Experimentation
- Always use different `CHECKPOINT_DIR` for each experiment
- Use meaningful `NAME` values to track experiments
- Keep `SEED` fixed for reproducibility

### 2. Hyperparameter Tuning
- Start with default values (they're proven to work well)
- Adjust one parameter at a time
- Common tuning order: lr → dropout → batch_size → model_size

### 3. Monitoring
- Use `MONITOR="val_acc"` for most cases (default)
- Use `MONITOR="val_loss"` if validation accuracy is noisy
- Increase `PATIENCE` for slower converging models

### 4. Regularization
- Increase `DROPOUT` if overfitting (val >> test)
- Increase `LABEL_SMOOTHING` if overfitting
- Increase `WEIGHT_DECAY` for better generalization

### 5. Model Size
- Larger models (higher `D_MODEL`, `NUM_LAYERS`) may overfit
- Keep total parameters < 500K
- The printed parameter count helps track this

### 6. Learning Rate
- OneCycleLR (default) works well for most cases
- Use Cosine for longer training runs
- Lower `LR` if training is unstable

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=64  # or 32
```

### Training Too Slow
```bash
# Reduce epochs for testing
EPOCHS=50

# Use smaller model
D_MODEL=64
NUM_LAYERS=2

# Increase batch size (if memory allows)
BATCH_SIZE=256
```

### Model Overfitting (Val >> Test)
```bash
# Increase regularization
DROPOUT=0.3
LABEL_SMOOTHING=0.15
WEIGHT_DECAY=0.0005
```

### Model Underfitting (Both Val and Test Low)
```bash
# Increase model capacity
D_MODEL=128
NUM_LAYERS=3

# Train longer
EPOCHS=150

# Reduce regularization
DROPOUT=0.1
LABEL_SMOOTHING=0.05
```

### Can't Reproduce Results
```bash
# Ensure same seed
SEED=42

# Check all parameters match
# Compare with FINAL_REPORT.md for reference values
```

## Output Files

After training, you'll find:

```
checkpoints/
├── best_model.pt          # Model checkpoint
└── config.json            # Full configuration used

logs/
└── *.log                  # Training logs (if redirected)
```

The `config.json` contains:
- All hyperparameters
- Dataset information
- Model parameter count
- Random seed

## Next Steps

1. **Run default configs** to verify everything works
2. **Experiment** with different hyperparameters
3. **Track results** by using different checkpoint directories
4. **Ensemble** your best models for better performance

## Support

For issues or questions:
- Check `README_NEW.md` for detailed project documentation
- Check `FINAL_REPORT.md` for performance benchmarks
- Review `train_model1_custom_example.sh` for configuration examples
