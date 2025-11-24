# Next-Location Prediction - Production Research Framework

A production-level PhD research codebase for next-location prediction using deep learning on trajectory data.

## Features

✅ **Reproducible Research**
- Deterministic seed setting across all libraries (PyTorch, NumPy, Python random)
- CUDA deterministic mode support
- Configuration-based experiments with full tracking

✅ **Production-Grade Code**
- Comprehensive logging and experiment tracking
- Automatic parameter inference (vocabulary sizes, sequence lengths)
- Support for multiple datasets (Geolife, DIY)
- Mixed precision training (AMP)
- Gradient clipping and learning rate scheduling
- Early stopping and model checkpointing

✅ **Well-Organized Structure**
- Clean separation of concerns (data, models, utils, configs)
- Extensive unit tests
- Type hints and documentation
- YAML-based configuration management

✅ **Experiment Management**
- TensorBoard integration
- Automatic metric tracking and logging
- Best model saving
- Experiment result archiving

## Project Structure

```
lose_expr_2/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default configuration
│   └── diy_experiment.yaml    # DIY dataset experiment
├── data/                      # Dataset directory
│   ├── geolife/              # Geolife dataset
│   └── diy/                  # DIY dataset
├── src/                       # Source code
│   ├── data/                 # Dataset loaders
│   │   ├── dataset.py        # Original dataset loader
│   │   └── dataset_v2.py     # Production dataset loader
│   ├── models/               # Model architectures
│   │   └── temporal_fusion.py
│   ├── utils/                # Utilities
│   │   ├── config.py         # Configuration management
│   │   ├── experiment_tracker.py  # Experiment tracking
│   │   ├── metrics_v2.py     # Comprehensive metrics
│   │   └── reproducibility.py     # Seed setting utilities
│   ├── train.py              # Original training script
│   └── train_production.py   # Production training script
├── tests/                     # Unit tests
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_reproducibility.py
├── experiments/               # Experiment logs (auto-created)
├── checkpoints/              # Model checkpoints (auto-created)
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README_NEW.md            # This file
```

## Installation

### 1. Clone the Repository

```bash
cd /content/lose_expr_2
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Basic Training (Geolife Dataset)

```bash
python src/train_production.py --config configs/default.yaml
```

### Training with DIY Dataset

```bash
python src/train_production.py --config configs/diy_experiment.yaml
```

### Custom Configuration

```bash
python src/train_production.py \
    --config configs/default.yaml \
    --dataset diy \
    --epochs 150 \
    --batch_size 128 \
    --lr 0.0005 \
    --seed 42
```

## Configuration

All experiments are controlled via YAML configuration files in `configs/`. Key parameters:

```yaml
# Reproducibility
seed: 42

# Dataset
data:
  dataset_name: "geolife"  # or "diy"
  batch_size: 256
  max_seq_len: null  # Auto-inferred from data

# Model Architecture
model:
  embedding_dim: 128
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1

# Training
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adamw"
  use_amp: true  # Mixed precision training
```

See `configs/default.yaml` for all available options.

## Running Tests

### Run All Tests

```bash
python tests/test_dataset.py
python tests/test_metrics.py
python tests/test_reproducibility.py
```

### Run with pytest

```bash
pip install pytest
pytest tests/ -v
```

## Experiment Tracking

Each experiment automatically creates a directory in `experiments/` with:

- `config.yaml` - Exact configuration used
- `metrics.json` - All metrics logged during training
- `summary.json` - Final results summary
- `experiment.log` - Detailed logs
- `tensorboard/` - TensorBoard logs
- `checkpoints/` - Model checkpoints

### View Results with TensorBoard

```bash
tensorboard --logdir experiments/
```

Then open http://localhost:6006 in your browser.

## Datasets

### Geolife Dataset

Located in `data/geolife/`:
- `geolife_transformer_7_train.pk`
- `geolife_transformer_7_validation.pk`
- `geolife_transformer_7_test.pk`

### DIY Dataset

Located in `data/diy/`:
- `diy_h3_res8_transformer_7_train.pk`
- `diy_h3_res8_transformer_7_validation.pk`
- `diy_h3_res8_transformer_7_test.pk`

### Dataset Format

Each pickle file contains a list of dictionaries with:
```python
{
    'X': [loc_1, loc_2, ..., loc_n],          # Location sequence
    'user_X': [user_1, user_2, ..., user_n],  # User IDs
    'weekday_X': [wd_1, wd_2, ..., wd_n],     # Weekdays (0-6)
    'start_min_X': [min_1, min_2, ..., min_n],# Start time (minutes)
    'dur_X': [dur_1, dur_2, ..., dur_n],      # Duration
    'diff': [diff_1, diff_2, ..., diff_n],    # Time differences
    'Y': next_location                         # Target location
}
```

## Reproducibility

The codebase ensures reproducibility through:

1. **Deterministic Seed Setting**: All random operations use the same seed
2. **CUDA Deterministic Mode**: GPU operations are deterministic
3. **Configuration Tracking**: All experiments save their exact config
4. **Version Logging**: System info and library versions are logged

Example:

```python
from utils.reproducibility import set_seed

# Set seed for all libraries
set_seed(42, cuda_deterministic=True)
```

## Model Architecture

**Temporal Fusion Model**: A modern non-RNN architecture combining:
- Multi-scale temporal convolutions (causal, dilated)
- Multi-head self-attention mechanisms
- Gated residual connections
- Feature fusion for trajectory modeling

Key features:
- <500K parameters (configurable)
- GPU-optimized
- No RNN/LSTM/GRU (fully parallelizable)

## Metrics

The framework computes comprehensive metrics:

- **Accuracy Metrics**: Top-1, Top-5, Top-10 accuracy
- **Ranking Metrics**: Mean Reciprocal Rank (MRR), NDCG@k
- **Classification Metrics**: Precision, Recall, F1-score

## Advanced Usage

### Creating a New Experiment Configuration

1. Copy `configs/default.yaml`
2. Modify parameters
3. Run with `--config path/to/your_config.yaml`

### Resuming from Checkpoint

```python
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Using Custom Datasets

1. Prepare data in the required format (see Dataset Format above)
2. Place in `data/your_dataset/`
3. Update `get_dataset_paths()` in `src/data/dataset_v2.py`
4. Create config file with `dataset_name: "your_dataset"`

## Best Practices

1. **Always use configuration files** for experiments (don't hardcode)
2. **Set seeds** for reproducibility
3. **Track experiments** with meaningful names
4. **Run tests** before major experiments
5. **Save best models** based on validation metrics
6. **Use version control** (git) for code changes

## Git Integration

The repository is already configured with git. To push changes:

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Description of changes"

# Push to GitHub
git push -u origin main
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `embedding_dim` or `hidden_dim`
- Disable mixed precision (`use_amp: false`)

### Slow Data Loading
- Increase `num_workers` in config
- Ensure data is on SSD
- Use `pin_memory: true` for GPU training

### Reproducibility Issues
- Ensure `cuda_deterministic: true`
- Same PyTorch/CUDA versions
- Same hardware (GPU model)

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{next_location_prediction,
  title={Production-Level Next-Location Prediction Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/aetandrianwr/lose_expr_2}
}
```

## License

[Add your license here]

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Last Updated**: November 2024
**Version**: 1.0.0
