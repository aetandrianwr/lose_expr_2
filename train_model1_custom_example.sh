#!/bin/bash

################################################################################
# Custom Training Example for Model 1
#
# This demonstrates how to easily experiment with different configurations.
# Copy this file and modify parameters to run experiments.
################################################################################

# Example 1: Larger model with more regularization
# Uncomment to use:
# python src/train_model1_configurable.py \
#     --d-model 128 \
#     --num-layers 3 \
#     --num-heads 8 \
#     --dropout 0.3 \
#     --label-smoothing 0.15 \
#     --checkpoint-dir ./checkpoints_large \
#     --name model1_large

# Example 2: Smaller model for faster training
# Uncomment to use:
# python src/train_model1_configurable.py \
#     --d-model 64 \
#     --num-layers 2 \
#     --num-heads 4 \
#     --batch-size 256 \
#     --epochs 50 \
#     --checkpoint-dir ./checkpoints_small \
#     --name model1_small_fast

# Example 3: Different learning rate and scheduler
# Uncomment to use:
# python src/train_model1_configurable.py \
#     --lr 0.0005 \
#     --scheduler cosine \
#     --epochs 150 \
#     --checkpoint-dir ./checkpoints_cosine \
#     --name model1_cosine_lr

# Example 4: Monitor val_loss instead of val_acc
# Uncomment to use:
# python src/train_model1_configurable.py \
#     --monitor val_loss \
#     --patience 20 \
#     --checkpoint-dir ./checkpoints_loss_monitor \
#     --name model1_loss_based

# Example 5: Different random seed for reproducibility test
# Uncomment to use:
# python src/train_model1_configurable.py \
#     --seed 123 \
#     --checkpoint-dir ./checkpoints_seed123 \
#     --name model1_seed123

# Example 6: Use DIY dataset instead of Geolife
# Uncomment to use:
# python src/train_model1_configurable.py \
#     --dataset diy \
#     --train-path /content/lose_expr_2/data/diy/diy_transformer_7_train.pk \
#     --val-path /content/lose_expr_2/data/diy/diy_transformer_7_validation.pk \
#     --test-path /content/lose_expr_2/data/diy/diy_transformer_7_test.pk \
#     --checkpoint-dir ./checkpoints_diy \
#     --name model1_diy

# Example 7: Heavy regularization to reduce overfitting
python src/train_model1_configurable.py \
    --dropout 0.4 \
    --weight-decay 0.0005 \
    --label-smoothing 0.2 \
    --patience 25 \
    --checkpoint-dir ./checkpoints_heavy_reg \
    --name model1_heavy_regularization

echo ""
echo "Training complete! Check ./checkpoints_heavy_reg/ for results."
echo ""
echo "To run other examples, edit this script and uncomment the desired configuration."
