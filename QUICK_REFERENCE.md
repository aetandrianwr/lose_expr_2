# Quick Reference Card

## 🚀 Common Commands

### Run Experiments
```bash
# Geolife dataset (default)
python src/train_production.py --config configs/default.yaml

# DIY dataset
python src/train_production.py --config configs/diy_experiment.yaml

# Custom parameters
python src/train_production.py \
    --config configs/default.yaml \
    --dataset diy \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --seed 42

# Quick test (10 epochs)
python src/train_production.py --config configs/default.yaml --epochs 10
```

### Testing
```bash
# Verify framework
python verify_framework.py

# Run all tests
python tests/test_dataset.py
python tests/test_metrics.py
python tests/test_reproducibility.py

# Interactive quick start
./quickstart.sh
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir experiments/

# List experiments
ls -lh experiments/

# View results
cat experiments/*/summary.json
```

### Git
```bash
# Check status
git status

# Commit changes
git add .
git commit -m "Your message"

# Push to GitHub
git push -u origin main

# View history
git log --oneline -10
```

## 📁 File Structure

```
lose_expr_2/
├── configs/              # Configuration files
│   ├── default.yaml     # Geolife config
│   └── diy_experiment.yaml
├── src/
│   ├── data/
│   │   └── dataset_v2.py      # Production dataset loader
│   ├── models/
│   │   └── temporal_fusion.py
│   ├── utils/
│   │   ├── config.py          # Config management
│   │   ├── experiment_tracker.py
│   │   ├── metrics_v2.py      # Metrics
│   │   └── reproducibility.py # Seed setting
│   ├── train.py              # Original
│   └── train_production.py   # Production script ⭐
├── tests/                    # Unit tests
├── experiments/              # Results (auto-created)
├── README_NEW.md            # Main docs ⭐
└── verify_framework.py      # Verification ⭐
```

## 🔧 Configuration Quick Edit

Edit `configs/default.yaml`:

```yaml
# Change seed
seed: 42  # Try 123, 456, etc.

# Change batch size
data:
  batch_size: 256  # Try 128, 512

# Change epochs
training:
  epochs: 100  # Try 50, 150

# Change learning rate
training:
  learning_rate: 0.001  # Try 0.0005, 0.002

# Change model size
model:
  embedding_dim: 128  # Try 64, 256
  hidden_dim: 256     # Try 128, 512
```

## 📊 Key Metrics

The framework computes:
- `accuracy` - Top-1 accuracy
- `top5_accuracy` - Top-5 accuracy
- `top10_accuracy` - Top-10 accuracy
- `mrr` - Mean Reciprocal Rank
- `ndcg@5` - NDCG at 5
- `ndcg@10` - NDCG at 10
- `precision` - Macro precision
- `recall` - Macro recall
- `f1_score` - Macro F1

## 🐍 Python API Quick Start

```python
from utils.reproducibility import set_seed, get_device
from utils.config import load_config
from data.dataset_v2 import create_dataloaders
from utils.experiment_tracker import ExperimentTracker

# Set seed
set_seed(42, cuda_deterministic=True)

# Load config
config = load_config('configs/default.yaml')

# Get device
device = get_device('cuda')

# Create dataloaders
train_loader, val_loader, test_loader, info = create_dataloaders(
    dataset_name='geolife',
    data_dir='data',
    batch_size=256
)

# Track experiment
with ExperimentTracker('my_exp', config=config) as tracker:
    # Training code here
    tracker.log_metric('val_acc', 85.3, epoch=10)
```

## 🔍 Troubleshooting

**CUDA Out of Memory**
```yaml
# Reduce batch size in config
data:
  batch_size: 128  # or 64
```

**Slow Training**
```yaml
# Use mixed precision
training:
  use_amp: true

# Increase workers
data:
  num_workers: 8
```

**Import Errors**
```bash
pip install -r requirements.txt
```

**Reproducibility Issues**
```yaml
# Enable deterministic mode
hardware:
  cuda_deterministic: true
```

## 📖 Documentation Files

- `README_NEW.md` - Comprehensive guide (start here!)
- `INSTALL.md` - Installation instructions
- `UPDATE_SUMMARY.md` - What changed
- `TRANSFORMATION_COMPLETE.md` - Complete overview
- `QUICK_REFERENCE.md` - This file

## ✅ Pre-Flight Checklist

Before running experiments:

- [ ] Run `python verify_framework.py`
- [ ] Check config file exists
- [ ] Set appropriate seed
- [ ] Check GPU availability
- [ ] Ensure enough disk space for checkpoints
- [ ] TensorBoard running (optional)

## 🎯 Typical Workflow

1. **Create config**
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   # Edit my_experiment.yaml
   ```

2. **Run experiment**
   ```bash
   python src/train_production.py --config configs/my_experiment.yaml
   ```

3. **Monitor**
   ```bash
   tensorboard --logdir experiments/
   ```

4. **Check results**
   ```bash
   cat experiments/my_experiment_*/summary.json
   ```

5. **Commit**
   ```bash
   git add configs/my_experiment.yaml
   git commit -m "Add my_experiment config"
   git push
   ```

## 💡 Best Practices

1. ✅ Always use configuration files
2. ✅ Set seed for reproducibility
3. ✅ Use meaningful experiment names
4. ✅ Track all experiments
5. ✅ Monitor with TensorBoard
6. ✅ Save best models
7. ✅ Document findings
8. ✅ Commit to git regularly

---

**Quick Help**: For detailed information, see `README_NEW.md`
