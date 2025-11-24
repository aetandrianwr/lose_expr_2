# Installation Guide

## Quick Install

### 1. Prerequisites

- Python 3.9 or higher
- CUDA 11.7+ (for GPU support)
- Git

### 2. Install Dependencies

```bash
cd /content/lose_expr_2
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python verify_framework.py
```

You should see:
```
✓ All tests passed! Framework is ready to use.
```

## Detailed Installation

### For Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### For Production

```bash
# Install without dev dependencies
pip install .
```

## GPU Support

The framework automatically detects and uses CUDA if available. To verify:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running Your First Experiment

```bash
# Train on Geolife dataset
python src/train_production.py --config configs/default.yaml

# Train on DIY dataset
python src/train_production.py --config configs/diy_experiment.yaml
```

## Troubleshooting

### ImportError: No module named 'yaml'

```bash
pip install pyyaml
```

### CUDA out of memory

Reduce batch size in config:
```yaml
data:
  batch_size: 128  # or lower
```

### Permission denied

```bash
chmod +x verify_framework.py
chmod +x src/train_production.py
```

## Next Steps

1. Read [README_NEW.md](README_NEW.md) for detailed documentation
2. Run tests: `python tests/test_*.py`
3. Start experimenting with different configurations
4. Check TensorBoard logs: `tensorboard --logdir experiments/`
