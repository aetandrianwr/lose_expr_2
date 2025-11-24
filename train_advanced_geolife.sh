#!/bin/bash

# Training script for Advanced Model on Geolife dataset
# Goal: Achieve >40% Acc@1 with <500K parameters

DATASET="geolife"
TRAIN_PATH="data/${DATASET}/${DATASET}_transformer_7_train.pk"
VAL_PATH="data/${DATASET}/${DATASET}_transformer_7_validation.pk"
TEST_PATH="data/${DATASET}/${DATASET}_transformer_7_test.pk"

# Model hyperparameters (optimized for <500K params)
D_MODEL=64
NUM_HEADS=4
NUM_LAYERS=2
NUM_CLUSTERS=40
DROPOUT=0.2

# Training hyperparameters
BATCH_SIZE=128
EPOCHS=100
LR=0.001
WEIGHT_DECAY=1e-4
PATIENCE=20

# Loss function
USE_FOCAL_LOSS="--use-focal-loss"
FOCAL_GAMMA=2.0

# Output directory
CHECKPOINT_DIR="checkpoints_advanced_geolife"

echo "=========================================="
echo "Training Advanced Model on Geolife"
echo "=========================================="
echo "Model: d_model=${D_MODEL}, heads=${NUM_HEADS}, layers=${NUM_LAYERS}"
echo "Clusters: ${NUM_CLUSTERS}"
echo "Training: batch=${BATCH_SIZE}, lr=${LR}, epochs=${EPOCHS}"
echo "Loss: Focal Loss (gamma=${FOCAL_GAMMA})"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

python src/train_advanced.py \
    --train-path ${TRAIN_PATH} \
    --val-path ${VAL_PATH} \
    --test-path ${TEST_PATH} \
    --d-model ${D_MODEL} \
    --num-heads ${NUM_HEADS} \
    --num-layers ${NUM_LAYERS} \
    --num-clusters ${NUM_CLUSTERS} \
    --dropout ${DROPOUT} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    ${USE_FOCAL_LOSS} \
    --focal-gamma ${FOCAL_GAMMA} \
    --patience ${PATIENCE} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --seed 42

echo ""
echo "Training complete! Check ${CHECKPOINT_DIR}/config.json for results."
