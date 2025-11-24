# Next-Location Prediction on Geolife Dataset

## Project Structure

```
lose_expr_2/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Dataset loaders
│   ├── utils/           # Metrics and utilities
│   └── train.py         # Training script
├── configs/             # Configuration files
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── notebooks/           # Analysis notebooks
```

## Architecture

**Temporal Fusion Model** - A modern, non-RNN architecture combining:
- Multi-scale temporal convolutions (causal, dilated)
- Multi-head self-attention mechanisms
- Gated residual connections
- Feature fusion for trajectory modeling

**Key Features:**
- <500K parameters
- GPU-optimized
- Mixed precision training
- Label smoothing + OneCycleLR scheduler
- No RNN/LSTM/GRU components

## Dataset

Geolife trajectory dataset with:
- 1,156 unique locations
- 45 users
- Train: 7,424 samples
- Val: 3,334 samples  
- Test: 3,502 samples

## Usage

```bash
cd /content/lose_expr_2
python src/train.py
```

## Target Performance

**Objective:** Stable 40% test Acc@1 on next-location prediction

## Dependencies

- PyTorch 1.12+ with CUDA
- NumPy, Pandas, scikit-learn
- tqdm, matplotlib, seaborn
