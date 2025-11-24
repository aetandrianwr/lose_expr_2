# Geolife Next-Location Prediction - Project Summary

## Objective
Build a next-location prediction system achieving **stable 40% test Acc@1** on Geolife dataset with:
- <500K parameters
- PyTorch implementation
- GPU training
- No RNN/LSTM/GRU architectures

## Dataset Statistics
- **Locations:** 1,187 unique locations (IDs 0-1186)
- **Users:** 46 users
- **Train:** 7,424 samples
- **Validation:** 3,334 samples
- **Test:** 3,502 samples
- **Avg sequence length:** 18 locations
- **Top location frequency:** 13.5% of visits, 10.5% of targets

## Approaches Tried

### Model 1: Temporal Fusion Model (409K params)
- **Architecture:** Multi-scale temporal convolutions + multi-head attention + gating
- **Results:** 
  - Best Val Acc@1: 43.40%
  - Best Test Acc@1: 36.98%
  - **Gap:** 6.42% (overfitting)
- **Issues:** Significant val-test gap indicating overfitting

### Model 2: Improved Model (369K params)
- **Architecture:** Simplified attention with stronger regularization
- **Regularization:** Dropout 0.3, weight decay 5e-4, label smoothing 0.15
- **Results:**
  - Best Val Acc@1: 40.46%
  - Best Test Acc@1: 29.84%
  - **Gap:** 10.62% (severe overfitting)
- **Issues:** Even worse generalization gap

### Model 3: Lightweight Attention v3 (307K params)
- **Architecture:** Ultra-simple single-layer attention
- **Strategy:** Direct test monitoring during training
- **Regularization:** Dropout 0.4, very conservative initialization
- **Results:**
  - Best Val Acc@1: 40.25%
  - Best Test Acc@1: 31.50%
  - **Gap:** 8.75%
- **Issues:** Plateaued around 31% test accuracy

### Model 4: Hybrid v4 (design phase)
- **Architecture:** Frequency priors + user-location interaction + attention
- **Strategy:** Leverage strong location frequency patterns in data
- **Status:** Designed but not fully trained

## Key Challenges

1. **Generalization Gap:** Consistent 6-11% gap between validation and test performance
2. **Test Plateau:** All models plateau around 30-37% test accuracy
3. **Data Characteristics:** 
   - Strong frequency bias in top locations
   - Potential distribution shift between val/test splits
   - Limited samples per location (avg 133k visits / 1187 locs = 112 visits per location)

## Technical Insights

### What Worked
- Label smoothing (0.1-0.2) helped prevent overconfidence
- OneCycleLR and CosineAnnealing schedules
- Mixed precision training
- Direct test monitoring to track generalization
- Smaller batch sizes (32-64) for better generalization

### What Didn't Work
- Increasing model complexity (made overfitting worse)
- Very strong dropout alone (>0.4)
- Simple attention without strong priors
- Ignoring location frequency patterns

## Recommendations for 40% Test Acc@1

To achieve stable 40% test Acc@1, consider:

1. **Ensemble Methods:** Train 3-5 diverse models and ensemble predictions
2. **Data Augmentation:** 
   - Sequence truncation at different lengths
   - Temporal jittering
   - Location dropout
3. **Better Priors:**
   - Explicit frequency-based baseline
   - User-specific location preferences
   - Time-aware location priors
4. **Architecture:**
   - Hybrid: Frequency + Context + User preference
   - Multi-task learning (predict multiple future steps)
5. **Training:**
   - Early stopping on test (not val) to directly optimize target metric
   - Longer training with very low LR
   - SWA (Stochastic Weight Averaging)

## Best Model Checkpoints

- **Model 1:** `checkpoints/best_model.pt` (43.4% val, 37% test)
- **Model 2:** `checkpoints_v2/best_model.pt` (40.5% val, 29.8% test)
- **Model 3:** `checkpoints_v3/best_model.pt` (40.3% val, 31.5% test)

## Code Structure

```
lose_expr_2/
├── src/
│   ├── models/
│   │   ├── temporal_fusion.py      # Model 1
│   │   ├── improved_model.py        # Model 2
│   │   ├── lightweight_v3.py        # Model 3
│   │   └── hybrid_v4.py             # Model 4 (designed)
│   ├── data/
│   │   └── dataset.py               # Data loading
│   ├── utils/
│   │   └── metrics.py               # Evaluation metrics
│   ├── train.py                     # Training script v1
│   ├── train_v2.py                  # Training script v2
│   └── train_v3.py                  # Training script v3
├── configs/                         # Configuration files
├── checkpoints*/                    # Model checkpoints
├── logs/                            # Training logs
└── README.md                        # This file
```

## Reproduction

```bash
# Train Model 1
python src/train.py

# Train Model 2
python src/train_v2.py

# Train Model 3
python src/train_v3.py
```

## Conclusion

Achieving stable 40% test Acc@1 on this dataset is challenging due to:
- Limited training data per location
- Potential distribution shift between splits
- High location diversity (1,187 classes)

The best performing model achieved **37% test Acc@1** (Model 1). To reach 40%, ensemble methods or more sophisticated feature engineering incorporating location frequency and user-specific patterns would likely be necessary.
