# Training Scripts Guide

This guide explains all available training and test scripts for both Geolife and DIY datasets.

## Training Scripts (Full Training)

### Model 1 (Temporal Fusion)
- **`train_model1_geolife.sh`** - Train Model 1 on Geolife dataset (100 epochs)
- **`train_model1_diy.sh`** - Train Model 1 on DIY dataset (100 epochs)

### Frequency-Aware Model
- **`train_freq_geolife.sh`** - Train Frequency-Aware model on Geolife (150 epochs)
- **`train_freq_diy.sh`** - Train Frequency-Aware model on DIY (150 epochs)

## Test Scripts (Quick 2-Epoch Validation)

### Model 1 (Temporal Fusion)
- **`test_model1_geolife.sh`** - Quick test of Model 1 on Geolife (2 epochs)
- **`test_model1_diy.sh`** - Quick test of Model 1 on DIY (2 epochs)

### Frequency-Aware Model
- **`test_freq_geolife.sh`** - Quick test of Frequency-Aware on Geolife (2 epochs)
- **`test_freq_diy.sh`** - Quick test of Frequency-Aware on DIY (2 epochs)

## Legacy Scripts (Generic - Geolife by default)

- **`train_model1.sh`** - Generic Model 1 training (Geolife by default)
- **`train_freq.sh`** - Generic Frequency-Aware training (Geolife by default)
- **`test_model1.sh`** - Generic Model 1 test (Geolife by default)
- **`test_freq.sh`** - Generic Frequency-Aware test (Geolife by default)

## Dataset Information

### Geolife Dataset
- **Locations:** 1,187 unique locations
- **Users:** 46 users
- **Max Sequence Length:** 54 (auto-inferred)
- **Data Path:** `/content/lose_expr_2/data/geolife/`

### DIY Dataset
- **Locations:** 2,810 unique locations
- **Users:** 1,156 users
- **Max Sequence Length:** 126 (auto-inferred)
- **Data Path:** `/content/lose_expr_2/data/diy/`

## Quick Start

### Run Full Training

```bash
# Model 1 on Geolife
bash train_model1_geolife.sh

# Model 1 on DIY
bash train_model1_diy.sh

# Frequency-Aware on Geolife
bash train_freq_geolife.sh

# Frequency-Aware on DIY
bash train_freq_diy.sh
```

### Run Quick Tests (2 epochs)

```bash
# Test Model 1 on Geolife
bash test_model1_geolife.sh

# Test Model 1 on DIY
bash test_model1_diy.sh

# Test Frequency-Aware on Geolife
bash test_freq_geolife.sh

# Test Frequency-Aware on DIY
bash test_freq_diy.sh
```

## Output Directories

### Training Checkpoints
- Model 1 Geolife: `./checkpoints_geolife/`
- Model 1 DIY: `./checkpoints_diy/`
- Frequency-Aware Geolife: `./checkpoints_freq_geolife/`
- Frequency-Aware DIY: `./checkpoints_freq_diy/`

### Test Checkpoints
- Model 1 Geolife Test: `./checkpoints_test_geolife/`
- Model 1 DIY Test: `./checkpoints_test_diy/`
- Frequency-Aware Geolife Test: `./checkpoints_freq_test_geolife/`
- Frequency-Aware DIY Test: `./checkpoints_freq_test_diy/`

## Key Features

✅ **Auto-inferred Parameters:** All scripts automatically infer:
- `max_seq_len` - Maximum sequence length from all datasets
- `num_locations` - Number of unique location classes
- `num_users` - Number of unique users

✅ **Dataset-Specific:** Each script is pre-configured for its specific dataset

✅ **Easy to Use:** Just run the script - no manual parameter configuration needed

✅ **Reproducible:** Fixed random seed (42) ensures reproducibility

## Customization

To customize any script, simply edit the parameters at the top of the file:
- Adjust `EPOCHS`, `LR`, `BATCH_SIZE`, etc.
- Change `CHECKPOINT_DIR` to save results elsewhere
- Modify model architecture parameters (`D_MODEL`, `NUM_LAYERS`, etc.)

## For More Details

See `TRAINING_SCRIPTS_README.md` for detailed information about:
- All available parameters
- Hyperparameter tuning tips
- Troubleshooting
- Advanced usage examples
