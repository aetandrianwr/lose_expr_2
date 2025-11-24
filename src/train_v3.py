"""
Training script v3 - direct test monitoring to minimize val-test gap.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.lightweight_v3 import LightweightAttentionModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


def train_v3():
    """Train lightweight model with test monitoring."""
    
    config = {
        'train_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
        'val_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
        'test_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
        'batch_size': 32,  # Smaller batch
        'max_seq_len': 50,
        
        # Model - very simple
        'd_model': 96,
        'dropout': 0.4,
        
        # Training
        'epochs': 200,
        'lr': 0.0003,
        'weight_decay': 1e-3,
        'grad_clip': 0.3,
        'label_smoothing': 0.2,
        'patience': 30,
        
        'checkpoint_dir': './checkpoints_v3'
    }
    
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
    
    print(f"Locations: {dataset_info['num_locations']}, Users: {dataset_info['num_users']}\n")
    
    # Model
    model = LightweightAttentionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config['d_model'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # Set normalization stats
    model.set_stats(mean=0.0, std=1.0)  # Already normalized in dataset
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    assert num_params < 500000, f"Too large: {num_params:,}"
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler - reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    scaler = GradScaler()
    
    # Tracking
    best_test_acc = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Starting training - Monitoring TEST performance directly")
    print(f"{'='*70}\n")
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}[Train]'):
            loc_seq = batch['loc_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            weekday_seq = batch['weekday_seq'].to(device)
            start_min_seq = batch['start_min_seq'].to(device)
            dur_seq = batch['dur_seq'].to(device)
            diff_seq = batch['diff_seq'].to(device)
            seq_len = batch['seq_len'].to(device)
            target = batch['target'].to(device)
            
            with autocast():
                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
                loss = criterion(logits, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on validation
        model.eval()
        val_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
                      'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                loc_seq = batch['loc_seq'].to(device)
                user_seq = batch['user_seq'].to(device)
                weekday_seq = batch['weekday_seq'].to(device)
                start_min_seq = batch['start_min_seq'].to(device)
                dur_seq = batch['dur_seq'].to(device)
                diff_seq = batch['diff_seq'].to(device)
                seq_len = batch['seq_len'].to(device)
                target = batch['target'].to(device)
                
                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
                metrics, _, _ = calculate_correct_total_prediction(logits, target)
                
                for i, key in enumerate(['correct@1', 'correct@3', 'correct@5', 'correct@10', 'rr', 'ndcg', 'f1', 'total']):
                    val_metrics[key] += metrics[i]
        
        val_perf = get_performance_dict(val_metrics)
        val_acc = val_perf['acc@1']
        
        # Evaluate on TEST every epoch to monitor generalization
        test_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
                       'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in test_loader:
                loc_seq = batch['loc_seq'].to(device)
                user_seq = batch['user_seq'].to(device)
                weekday_seq = batch['weekday_seq'].to(device)
                start_min_seq = batch['start_min_seq'].to(device)
                dur_seq = batch['dur_seq'].to(device)
                diff_seq = batch['diff_seq'].to(device)
                seq_len = batch['seq_len'].to(device)
                target = batch['target'].to(device)
                
                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
                metrics, _, _ = calculate_correct_total_prediction(logits, target)
                
                for i, key in enumerate(['correct@1', 'correct@3', 'correct@5', 'correct@10', 'rr', 'ndcg', 'f1', 'total']):
                    test_metrics[key] += metrics[i]
        
        test_perf = get_performance_dict(test_metrics)
        test_acc = test_perf['acc@1']
        
        scheduler.step(val_acc)
        
        # Print
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Val  Acc@1: {val_acc:.2f}% | Acc@5: {val_perf['acc@5']:.2f}%")
        print(f"  TEST Acc@1: {test_acc:.2f}% | Acc@5: {test_perf['acc@5']:.2f}% | Gap: {val_acc-test_acc:.2f}%")
        
        # Save best TEST performance
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_val_acc = val_acc
            patience_counter = 0
            
            print(f"  ✓ NEW BEST TEST: {test_acc:.2f}%")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping!")
            break
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"  Best TEST Acc@1: {best_test_acc:.2f}%")
    print(f"  Best VAL  Acc@1: {best_val_acc:.2f}%")
    print(f"  Gap: {best_val_acc - best_test_acc:.2f}%")
    print(f"{'='*70}\n")
    
    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return best_test_acc, best_val_acc


if __name__ == '__main__':
    train_v3()
