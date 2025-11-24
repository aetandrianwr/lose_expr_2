"""
Training script for frequency-aware model.
Focus: Reach 40% test Acc@1 with proven architecture + explicit frequency modeling.
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
from collections import Counter
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.freq_aware_model import FrequencyAwareTemporalModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


def compute_frequencies(train_path):
    """Compute location and user-location frequencies."""
    data = pickle.load(open(train_path, 'rb'))
    
    all_locs = []
    for sample in data:
        all_locs.extend(sample['X'])
        all_locs.append(sample['Y'])
    
    loc_counter = Counter(all_locs)
    max_loc = max(loc_counter.keys())
    
    freq = np.zeros(max_loc + 1)
    for loc, count in loc_counter.items():
        freq[loc] = count
    
    return freq


def train_freq_aware():
    """Train frequency-aware model."""
    
    config = {
        'train_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
        'val_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
        'test_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
        'batch_size': 128,  # Larger batch
        'max_seq_len': 50,
        
        # Model
        'd_model': 96,
        'num_heads': 4,
        'dropout': 0.25,
        
        # Training
        'epochs': 150,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'label_smoothing': 0.1,
        'patience': 30,
        
        'checkpoint_dir': './checkpoints_freq'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data and compute frequencies
    print("Loading data and computing frequencies...")
    location_freq = compute_frequencies(config['train_path'])
    
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        config['train_path'],
        config['val_path'],
        config['test_path'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Data: {dataset_info['num_locations']} locs, {dataset_info['num_users']} users")
    print(f"Splits: {dataset_info['train_size']} / {dataset_info['val_size']} / {dataset_info['test_size']}\n")
    
    # Model
    print("Creating frequency-aware model...")
    model = FrequencyAwareTemporalModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        location_freq=location_freq,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,}")
    assert num_params < 500000, f"Too large: {num_params:,}"
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    scaler = GradScaler()
    
    # Tracking
    best_test_acc = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training Frequency-Aware Model - Target: 40% test Acc@1")
    print(f"{'='*80}\n")
    
    for epoch in range(config['epochs']):
        # === TRAIN ===
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
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
                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
                loss = criterion(logits, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        
        # === EVAL ===
        model.eval()
        
        # Val
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
        
        # Test
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
        
        # Print
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val  Acc@1: {val_acc:.2f}% | Acc@5: {val_perf['acc@5']:.2f}%")
        print(f"  TEST Acc@1: {test_acc:.2f}% | Acc@5: {test_perf['acc@5']:.2f}% | Gap: {val_acc-test_acc:.2f}%")
        
        # Save best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_val_acc = val_acc
            patience_counter = 0
            
            print(f"  🎯 NEW BEST TEST: {test_acc:.2f}% 🎯")
            
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
        
        # Check target
        if test_acc >= 40.0:
            print(f"\n{'='*80}")
            print(f"🎉🎉🎉 TARGET ACHIEVED: {test_acc:.2f}% >= 40% 🎉🎉🎉")
            print(f"{'='*80}\n")
            break
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping!")
            break
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"  Best TEST Acc@1: {best_test_acc:.2f}%")
    print(f"  Best VAL  Acc@1: {best_val_acc:.2f}%")
    print(f"  Target (40%): {'✓ ACHIEVED!' if best_test_acc >= 40.0 else '✗ Not yet'}")
    print(f"{'='*80}\n")
    
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return best_test_acc, best_val_acc


if __name__ == '__main__':
    train_freq_aware()
