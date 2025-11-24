#!/bin/bash

################################################################################
# Test Frequency-Aware Model on DIY Dataset - 2 Epochs
################################################################################

# Dataset configuration
DATASET="diy"
TRAIN_PATH="/content/lose_expr_2/data/diy/diy_h3_res8_transformer_7_train.pk"
VAL_PATH="/content/lose_expr_2/data/diy/diy_h3_res8_transformer_7_validation.pk"
TEST_PATH="/content/lose_expr_2/data/diy/diy_h3_res8_transformer_7_test.pk"
BATCH_SIZE=128

# Model architecture
D_MODEL=96
NUM_HEADS=4
DROPOUT=0.25

# Training hyperparameters
EPOCHS=2              # TEST: Only 2 epochs
LR=0.001
WEIGHT_DECAY=0.0001
GRAD_CLIP=1.0
LABEL_SMOOTHING=0.1

# Learning rate scheduler
SCHEDULER="onecycle"
PCT_START=0.1

# Early stopping
PATIENCE=30
MONITOR="val_acc"

# Output
CHECKPOINT_DIR="./checkpoints_freq_test_diy"
NAME="freq_aware_test_diy"

# Other
SEED=42
DEVICE="cuda"

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
