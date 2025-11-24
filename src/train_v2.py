"""
Improved training script with better generalization strategies.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.improved_model import ImprovedNextLocModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


def train_improved_model():
    """Train improved model with better generalization."""
    
    # Configuration optimized for generalization
    config = {
        # Data
        'train_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
        'val_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
        'test_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
        'batch_size': 64,  # Smaller batch for better generalization
        'max_seq_len': 50,
        
        # Model - smaller for better generalization
        'd_model': 112,
        'num_layers': 2,
        'dropout': 0.3,  # Higher dropout
        
        # Training - stronger regularization
        'epochs': 150,
        'lr': 0.0005,  # Lower learning rate
        'weight_decay': 5e-4,  # Stronger weight decay
        'grad_clip': 0.5,  # Tighter gradient clipping
        'label_smoothing': 0.15,  # More label smoothing
        'patience': 25,  # More patience
        
        # Paths
        'checkpoint_dir': './checkpoints_v2'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        config['train_path'],
        config['val_path'],
        config['test_path'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Dataset info:")
    print(f"  Locations: {dataset_info['num_locations']}")
    print(f"  Users: {dataset_info['num_users']}")
    print(f"  Train: {dataset_info['train_size']}")
    print(f"  Val: {dataset_info['val_size']}")
    print(f"  Test: {dataset_info['test_size']}\n")
    
    # Create model
    print("Creating model...")
    model = ImprovedNextLocModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    assert num_params < 500000, f"Model too large: {num_params:,} >= 500K"
    
    # Optimizer with stronger weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    # Mixed precision
    scaler = GradScaler()
    
    # Tracking
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0
    
    # Checkpoint dir
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(config['epochs']):
        # Training
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
                logits = model(
                    loc_seq, user_seq, weekday_seq,
                    start_min_seq, dur_seq, diff_seq, seq_len
                )
                loss = criterion(logits, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        all_metrics = {
            'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
            'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='[Val]'):
                loc_seq = batch['loc_seq'].to(device)
                user_seq = batch['user_seq'].to(device)
                weekday_seq = batch['weekday_seq'].to(device)
                start_min_seq = batch['start_min_seq'].to(device)
                dur_seq = batch['dur_seq'].to(device)
                diff_seq = batch['diff_seq'].to(device)
                seq_len = batch['seq_len'].to(device)
                target = batch['target'].to(device)
                
                logits = model(
                    loc_seq, user_seq, weekday_seq,
                    start_min_seq, dur_seq, diff_seq, seq_len
                )
                
                metrics, _, _ = calculate_correct_total_prediction(logits, target)
                all_metrics['correct@1'] += metrics[0]
                all_metrics['correct@3'] += metrics[1]
                all_metrics['correct@5'] += metrics[2]
                all_metrics['correct@10'] += metrics[3]
                all_metrics['rr'] += metrics[4]
                all_metrics['ndcg'] += metrics[5]
                all_metrics['f1'] += metrics[6]
                all_metrics['total'] += metrics[7]
        
        val_perf = get_performance_dict(all_metrics)
        val_acc = val_perf['acc@1']
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Acc@1:  {val_acc:.2f}%")
        print(f"  Val Acc@5:  {val_perf['acc@5']:.2f}%")
        print(f"  Val MRR:    {val_perf['mrr']:.2f}%")
        
        # Check improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Test
            all_test_metrics = {
                'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0
            }
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='[Test]'):
                    loc_seq = batch['loc_seq'].to(device)
                    user_seq = batch['user_seq'].to(device)
                    weekday_seq = batch['weekday_seq'].to(device)
                    start_min_seq = batch['start_min_seq'].to(device)
                    dur_seq = batch['dur_seq'].to(device)
                    diff_seq = batch['diff_seq'].to(device)
                    seq_len = batch['seq_len'].to(device)
                    target = batch['target'].to(device)
                    
                    logits = model(
                        loc_seq, user_seq, weekday_seq,
                        start_min_seq, dur_seq, diff_seq, seq_len
                    )
                    
                    metrics, _, _ = calculate_correct_total_prediction(logits, target)
                    all_test_metrics['correct@1'] += metrics[0]
                    all_test_metrics['correct@3'] += metrics[1]
                    all_test_metrics['correct@5'] += metrics[2]
                    all_test_metrics['correct@10'] += metrics[3]
                    all_test_metrics['rr'] += metrics[4]
                    all_test_metrics['ndcg'] += metrics[5]
                    all_test_metrics['f1'] += metrics[6]
                    all_test_metrics['total'] += metrics[7]
            
            test_perf = get_performance_dict(all_test_metrics)
            test_acc = test_perf['acc@1']
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            print(f"\n  ✓ New best Val Acc@1: {val_acc:.2f}%")
            print(f"  → Test Acc@1: {test_acc:.2f}%")
            print(f"  → Test Acc@5: {test_perf['acc@5']:.2f}%")
            print(f"  → Test MRR:   {test_perf['mrr']:.2f}%")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  Checkpoint saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val Acc@1:  {best_val_acc:.2f}%")
    print(f"Best Test Acc@1: {best_test_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return best_val_acc, best_test_acc


if __name__ == '__main__':
    train_improved_model()
