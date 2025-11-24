#!/bin/bash

# Final attempt: Retrain baseline with lower LR and longer patience
# Baseline: d_model=96, 36.98% test acc
# Target: >40% with better hyperparameters

DATASET="geolife"
TRAIN_PATH="data/${DATASET}/${DATASET}_transformer_7_train.pk"
VAL_PATH="data/${DATASET}/${DATASET}_transformer_7_validation.pk"
TEST_PATH="data/${DATASET}/${DATASET}_transformer_7_test.pk"

# Model hyperparameters (same as best baseline)
D_MODEL=96
NUM_LAYERS=2
NUM_HEADS=4
KERNEL_SIZE=3
DROPOUT=0.25  # Slightly more regularization

# Training hyperparameters (improved)
BATCH_SIZE=128
EPOCHS=150  # More epochs
LR=0.0005  # Lower LR for better convergence
WEIGHT_DECAY=0.0001
GRAD_CLIP=1.0
LABEL_SMOOTHING=0.1
PATIENCE=25  # More patience

# Output directory
CHECKPOINT_DIR="checkpoints_final"

echo "=========================================="
echo "Final Training Attempt - Lower LR + More Patience"
echo "=========================================="
echo "Model: d_model=${D_MODEL}, layers=${NUM_LAYERS} (same as 36.98% baseline)"
echo "Training: lr=${LR} (lower), patience=${PATIENCE} (higher), epochs=${EPOCHS}"
echo "Target: >40% test Acc@1"
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
    --seed 123  # Different seed

echo ""
echo "Training complete! Check ${CHECKPOINT_DIR}/config.json for results."
