#!/bin/bash

# Training script for Improved Temporal Fusion Model (Model 1 + 8 dim)
# Baseline: d_model=96, 412K params, 36.98% test acc
# Improved: d_model=104, 472K params, target >40%

DATASET="geolife"
TRAIN_PATH="data/${DATASET}/${DATASET}_transformer_7_train.pk"
VAL_PATH="data/${DATASET}/${DATASET}_transformer_7_validation.pk"
TEST_PATH="data/${DATASET}/${DATASET}_transformer_7_test.pk"

# Model hyperparameters (baseline + 8 dim)
D_MODEL=104
NUM_LAYERS=2
NUM_HEADS=4
KERNEL_SIZE=3
DROPOUT=0.2

# Training hyperparameters (same as baseline)
BATCH_SIZE=128
EPOCHS=100
LR=0.001
WEIGHT_DECAY=0.0001
GRAD_CLIP=1.0
LABEL_SMOOTHING=0.1
PATIENCE=15

# Output directory
CHECKPOINT_DIR="checkpoints_improved"

echo "=========================================="
echo "Training Improved Temporal Fusion (Model 1)"
echo "=========================================="
echo "Model: d_model=${D_MODEL}, layers=${NUM_LAYERS}, heads=${NUM_HEADS}"
echo "Training: batch=${BATCH_SIZE}, lr=${LR}, epochs=${EPOCHS}"
echo "Target: >40% test Acc@1 (baseline: 36.98%)"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

python src/train_model1_configurable.py \
    --train-path ${TRAIN_PATH} \
    --val-path ${VAL_PATH} \
    --test-path ${TEST_PATH} \
    --d-model ${D_MODEL} \
    --num-layers ${NUM_LAYERS} \
    --num-heads ${NUM_HEADS} \
    --kernel-size ${KERNEL_SIZE} \
    --dropout ${DROPOUT} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --grad-clip ${GRAD_CLIP} \
    --label-smoothing ${LABEL_SMOOTHING} \
    --patience ${PATIENCE} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --seed 42

echo ""
echo "Training complete! Check ${CHECKPOINT_DIR}/config.json for results."
