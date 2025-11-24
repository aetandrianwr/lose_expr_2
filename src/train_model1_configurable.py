"""
Configurable training script for Model 1 (Temporal Fusion).
Supports command-line arguments to override default configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.temporal_fusion import TemporalFusionModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Model 1 (Temporal Fusion) for next-location prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument('--dataset', type=str, default='geolife',
                              help='Dataset name (geolife or diy)')
    dataset_group.add_argument('--train-path', type=str,
                              default='/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
                              help='Path to training data')
    dataset_group.add_argument('--val-path', type=str,
                              default='/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
                              help='Path to validation data')
    dataset_group.add_argument('--test-path', type=str,
                              default='/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
                              help='Path to test data')
    dataset_group.add_argument('--batch-size', type=int, default=128,
                              help='Batch size')

    # Model architecture arguments
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--d-model', type=int, default=96,
                            help='Model dimension')
    model_group.add_argument('--num-layers', type=int, default=2,
                            help='Number of temporal layers')
    model_group.add_argument('--num-heads', type=int, default=4,
                            help='Number of attention heads')
    model_group.add_argument('--kernel-size', type=int, default=3,
                            help='Convolution kernel size')
    model_group.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout rate')

    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_group.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=1e-4,
                            help='Weight decay')
    train_group.add_argument('--grad-clip', type=float, default=1.0,
                            help='Gradient clipping value')
    train_group.add_argument('--label-smoothing', type=float, default=0.1,
                            help='Label smoothing factor')

    # Scheduler arguments
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler')
    scheduler_group.add_argument('--scheduler', type=str, default='onecycle',
                                choices=['onecycle', 'cosine', 'step'],
                                help='Learning rate scheduler type')
    scheduler_group.add_argument('--pct-start', type=float, default=0.1,
                                help='Percentage of warmup for OneCycleLR')

    # Early stopping arguments
    early_stop_group = parser.add_argument_group('Early Stopping')
    early_stop_group.add_argument('--patience', type=int, default=15,
                                  help='Early stopping patience')
    early_stop_group.add_argument('--monitor', type=str, default='val_acc',
                                  choices=['val_acc', 'val_loss'],
                                  help='Metric to monitor for best model')

    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                             help='Directory to save checkpoints')
    output_group.add_argument('--name', type=str, default='model1',
                             help='Experiment name')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')

    return parser.parse_args()


def print_config(config):
    """Print configuration in a formatted way."""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print()

    print("Dataset:")
    print(f"  Dataset name:       {config['dataset']}")
    print(f"  Train path:         {config['train_path']}")
    print(f"  Val path:           {config['val_path']}")
    print(f"  Test path:          {config['test_path']}")
    print(f"  Batch size:         {config['batch_size']}")
    print()

    print("Model Architecture:")
    print(f"  Model type:         Temporal Fusion")
    print(f"  d_model:            {config['d_model']}")
    print(f"  num_layers:         {config['num_layers']}")
    print(f"  num_heads:          {config['num_heads']}")
    print(f"  kernel_size:        {config['kernel_size']}")
    print(f"  dropout:            {config['dropout']}")
    print()

    print("Training:")
    print(f"  Epochs:             {config['epochs']}")
    print(f"  Learning rate:      {config['lr']}")
    print(f"  Weight decay:       {config['weight_decay']}")
    print(f"  Grad clip:          {config['grad_clip']}")
    print(f"  Label smoothing:    {config['label_smoothing']}")
    print(f"  Scheduler:          {config['scheduler']}")
    if config['scheduler'] == 'onecycle':
        print(f"  PCT start:          {config['pct_start']}")
    print()

    print("Early Stopping:")
    print(f"  Patience:           {config['patience']}")
    print(f"  Monitor metric:     {config['monitor']}")
    print()

    print("Output:")
    print(f"  Checkpoint dir:     {config['checkpoint_dir']}")
    print(f"  Experiment name:    {config['name']}")
    print()

    print("Other:")
    print(f"  Random seed:        {config['seed']}")
    print(f"  Device:             {config['device']}")
    print()

    print("=" * 80)
    print()


def main():
    """Main training function."""
    args = parse_args()

    # Convert args to config dict
    config = {
        'dataset': args.dataset,
        'train_path': args.train_path,
        'val_path': args.val_path,
        'test_path': args.test_path,
        'batch_size': args.batch_size,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'kernel_size': args.kernel_size,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'label_smoothing': args.label_smoothing,
        'scheduler': args.scheduler,
        'pct_start': args.pct_start,
        'patience': args.patience,
        'monitor': args.monitor,
        'checkpoint_dir': args.checkpoint_dir,
        'name': args.name,
        'seed': args.seed,
        'device': args.device
    }

    # Print configuration
    print_config(config)

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        config['train_path'],
        config['val_path'],
        config['test_path'],
        batch_size=config['batch_size'],
        max_seq_len=None  # Infer from data
    )

    print(f"Loaded: {dataset_info['num_locations']} locations, {dataset_info['num_users']} users")
    print(f"Max sequence length: {dataset_info['max_seq_len']} (inferred from data)")
    print(f"Splits: train={dataset_info['train_size']}, val={dataset_info['val_size']}, test={dataset_info['test_size']}")
    print()

    # Create model
    print("Creating model...")
    model = TemporalFusionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        max_seq_len=dataset_info['max_seq_len']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    if num_params >= 500000:
        print(f"⚠ Warning: Model has {num_params:,} parameters (limit: 500K)")
    print()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Scheduler
    if config['scheduler'] == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=config['pct_start'],
            anneal_strategy='cos'
        )
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    scaler = GradScaler()

    # Tracking
    best_val_metric = 0.0 if config['monitor'] == 'val_acc' else float('inf')
    best_test_acc = 0.0
    patience_counter = 0

    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_save = config.copy()
    config_save['num_locations'] = int(dataset_info['num_locations'])
    config_save['num_users'] = int(dataset_info['num_users'])
    config_save['max_seq_len'] = int(dataset_info['max_seq_len'])
    config_save['num_parameters'] = int(num_params)

    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2)

    print(f"{'='*80}")
    print(f"Starting training - Monitoring: {config['monitor']}")
    print(f"{'='*80}")
    print()

    # Training loop
    for epoch in range(config['epochs']):
        # ===== TRAIN =====
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        for batch in pbar:
            loc_seq = batch['loc_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            weekday_seq = batch['weekday_seq'].to(device)
            start_min_seq = batch['start_min_seq'].to(device)
            dur_seq = batch['dur_seq'].to(device)
            diff_seq = batch['diff_seq'].to(device)
            seq_len = batch['seq_len'].to(device)
            target = batch['target'].to(device)

            with autocast():
                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq,
                             dur_seq, diff_seq, seq_len)
                loss = criterion(logits, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            if config['scheduler'] == 'onecycle':
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if config['scheduler'] != 'onecycle':
            scheduler.step()

        avg_loss = total_loss / num_batches

        # ===== VALIDATE =====
        model.eval()
        val_metrics = {
            'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
            'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0
        }
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='[Val]', leave=False):
                loc_seq = batch['loc_seq'].to(device)
                user_seq = batch['user_seq'].to(device)
                weekday_seq = batch['weekday_seq'].to(device)
                start_min_seq = batch['start_min_seq'].to(device)
                dur_seq = batch['dur_seq'].to(device)
                diff_seq = batch['diff_seq'].to(device)
                seq_len = batch['seq_len'].to(device)
                target = batch['target'].to(device)

                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq,
                             dur_seq, diff_seq, seq_len)
                loss = criterion(logits, target)
                val_loss += loss.item()
                val_batches += 1

                metrics, _, _ = calculate_correct_total_prediction(logits, target)
                for i, key in enumerate(['correct@1', 'correct@3', 'correct@5',
                                        'correct@10', 'rr', 'ndcg', 'f1', 'total']):
                    val_metrics[key] += metrics[i]

        val_perf = get_performance_dict(val_metrics)
        avg_val_loss = val_loss / val_batches

        # Print epoch results
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc@1:  {val_perf['acc@1']:.2f}%")
        print(f"  Val Acc@5:  {val_perf['acc@5']:.2f}%")
        print(f"  Val MRR:    {val_perf['mrr']:.2f}%")

        # Check for improvement
        current_metric = val_perf['acc@1'] if config['monitor'] == 'val_acc' else avg_val_loss
        is_best = (current_metric > best_val_metric) if config['monitor'] == 'val_acc' else (current_metric < best_val_metric)

        if is_best:
            best_val_metric = current_metric
            patience_counter = 0

            # Test on best validation
            test_metrics = {
                'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0
            }

            with torch.no_grad():
                for batch in tqdm(test_loader, desc='[Test]', leave=False):
                    loc_seq = batch['loc_seq'].to(device)
                    user_seq = batch['user_seq'].to(device)
                    weekday_seq = batch['weekday_seq'].to(device)
                    start_min_seq = batch['start_min_seq'].to(device)
                    dur_seq = batch['dur_seq'].to(device)
                    diff_seq = batch['diff_seq'].to(device)
                    seq_len = batch['seq_len'].to(device)
                    target = batch['target'].to(device)

                    logits = model(loc_seq, user_seq, weekday_seq, start_min_seq,
                                 dur_seq, diff_seq, seq_len)
                    metrics, _, _ = calculate_correct_total_prediction(logits, target)
                    for i, key in enumerate(['correct@1', 'correct@3', 'correct@5',
                                            'correct@10', 'rr', 'ndcg', 'f1', 'total']):
                        test_metrics[key] += metrics[i]

            test_perf = get_performance_dict(test_metrics)
            test_acc = test_perf['acc@1']

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            metric_str = f"{current_metric:.4f}" if config['monitor'] == 'val_loss' else f"{current_metric:.2f}%"
            print(f"  ✓ New best {config['monitor']}: {metric_str}")
            print(f"  → Test Acc@1: {test_acc:.2f}%")
            print(f"  → Test Acc@5: {test_perf['acc@5']:.2f}%")
            print(f"  → Test MRR:   {test_perf['mrr']:.2f}%")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_perf['acc@1'],
                'val_loss': avg_val_loss,
                'test_acc': test_acc,
                'config': config_save
            }
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        print()

    # Final summary
    print(f"{'='*80}")
    print(f"Training completed!")
    metric_str = f"{best_val_metric:.4f}" if config['monitor'] == 'val_loss' else f"{best_val_metric:.2f}%"
    print(f"Best Val {config['monitor']}: {metric_str}")
    print(f"Best Test Acc@1: {best_test_acc:.2f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
