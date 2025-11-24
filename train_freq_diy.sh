#!/bin/bash

################################################################################
# Train Frequency-Aware Model on DIY Dataset
################################################################################

# Dataset configuration
DATASET="diy"
TRAIN_PATH="/content/lose_expr_2/data/diy/diy_h3_res8_transformer_7_train.pk"
VAL_PATH="/content/lose_expr_2/data/diy/diy_h3_res8_transformer_7_validation.pk"
TEST_PATH="/content/lose_expr_2/data/diy/diy_h3_res8_transformer_7_test.pk"
BATCH_SIZE=128

# Model architecture
D_MODEL=96            # Model dimension (64, 96, 128, 256)
NUM_HEADS=4           # Attention heads (4, 8, 16)
DROPOUT=0.25          # Dropout rate (0.1-0.4)

# Training hyperparameters
EPOCHS=150            # Number of epochs
LR=0.001              # Learning rate
WEIGHT_DECAY=0.0001   # Weight decay (L2 regularization)
GRAD_CLIP=1.0         # Gradient clipping
LABEL_SMOOTHING=0.1   # Label smoothing (0.0-0.2)

# Learning rate scheduler
SCHEDULER="onecycle"  # Options: onecycle, cosine, step
PCT_START=0.1         # Warmup percentage for OneCycleLR

# Early stopping
PATIENCE=30           # Early stopping patience
MONITOR="val_acc"     # Monitor metric: val_acc or val_loss

# Output
CHECKPOINT_DIR="./checkpoints_freq_diy"
NAME="freq_aware_diy"

# Other
SEED=42
DEVICE="cuda"         # cuda or cpu

# ============================================================================
# Run training
# ============================================================================

python src/train_freq_configurable.py \
    --dataset "$DATASET" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --batch-size $BATCH_SIZE \
    --d-model $D_MODEL \
    --num-heads $NUM_HEADS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --grad-clip $GRAD_CLIP \
    --label-smoothing $LABEL_SMOOTHING \
    --scheduler "$SCHEDULER" \
    --pct-start $PCT_START \
    --patience $PATIENCE \
    --monitor "$MONITOR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --name "$NAME" \
    --seed $SEED \
    --device "$DEVICE"
