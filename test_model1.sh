#!/bin/bash

################################################################################
# Test Model 1 (Temporal Fusion) - 2 Epochs Test Run
################################################################################

# Dataset configuration
DATASET="geolife"
TRAIN_PATH="/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk"
VAL_PATH="/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk"
TEST_PATH="/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk"
BATCH_SIZE=128
MAX_SEQ_LEN=50

# Model architecture
D_MODEL=96
NUM_LAYERS=2
NUM_HEADS=4
KERNEL_SIZE=3
DROPOUT=0.2

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
PATIENCE=15
MONITOR="val_acc"

# Output
CHECKPOINT_DIR="./checkpoints_test"
NAME="model1_test"

# Other
SEED=42
DEVICE="cuda"

# ============================================================================
# Run training
# ============================================================================

python src/train_model1_configurable.py \
    --dataset "$DATASET" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --batch-size $BATCH_SIZE \
    --max-seq-len $MAX_SEQ_LEN \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --kernel-size $KERNEL_SIZE \
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
