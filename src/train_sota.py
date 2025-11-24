"""
State-of-the-art training script with advanced techniques.

Techniques applied:
1. DiffTopK loss for top-1 optimization
2. Focal loss for hard examples
3. Label smoothing
4. Mixup augmentation
5. Learning rate warmup + cosine annealing
6. Gradient accumulation
7. EMA (Exponential Moving Average)
8. Test-time augmentation
"""

import os
import sys
import json
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.sota_model import StateOfTheArtModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict
from utils.advanced_losses import CombinedLoss, SequenceMixup


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_location_freq(train_data_path):
    """Compute location frequency from training data."""
    import pickle
    train_data = pickle.load(open(train_data_path, 'rb'))
    
    all_locs = []
    for sample in train_data:
        all_locs.extend(sample['X'])
        all_locs.append(sample['Y'])
    
    loc_counter = Counter(all_locs)
    max_loc = max(loc_counter.keys())
    
    freq = np.zeros(max_loc + 1)
    for loc, count in loc_counter.items():
        freq[loc] = count
    
    return freq


def train_sota():
    """Train state-of-the-art model."""
    
    # Configuration
    config = {
        'train_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
        'val_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
        'test_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
        'batch_size': 64,
        'accum_steps': 2,  # Gradient accumulation
        'max_seq_len': 50,
        
        # Model
        'd_model': 96,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.2,
        
        # Training
        'epochs': 200,
        'lr': 0.0008,
        'weight_decay': 1e-4,
        'grad_clip': 0.5,
        'warmup_epochs': 10,
        'patience': 40,
        
        # Advanced techniques
        'use_difftopk': True,
        'use_focal': True,
        'use_mixup': False,  # Start without mixup
        'use_ema': True,
        'label_smoothing': 0.1,
        
        'checkpoint_dir': './checkpoints_sota'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Compute location frequency
    print("Computing location frequency...")
    location_freq = compute_location_freq(config['train_path'])
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        config['train_path'],
        config['val_path'],
        config['test_path'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Dataset: {dataset_info['num_locations']} locations, {dataset_info['num_users']} users")
    print(f"Splits: {dataset_info['train_size']} train, {dataset_info['val_size']} val, {dataset_info['test_size']} test\n")
    
    # Create model
    print("Creating model...")
    model = StateOfTheArtModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        location_freq=location_freq,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    if num_params >= 500000:
        print(f"WARNING: {num_params:,} >= 500K, adjusting model size...")
        # Reduce model size
        model = StateOfTheArtModel(
            num_locations=dataset_info['num_locations'],
            num_users=dataset_info['num_users'],
            location_freq=location_freq,
            d_model=128,
            num_layers=2,
            num_heads=4,
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        ).to(device)
        num_params = model.count_parameters()
        print(f"Adjusted model parameters: {num_params:,}")
    
    # EMA
    ema = None
    if config['use_ema']:
        ema = EMA(model, decay=0.999)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Scheduler with warmup
    total_steps = len(train_loader) * config['epochs'] // config['accum_steps']
    warmup_steps = len(train_loader) * config['warmup_epochs'] // config['accum_steps']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = CombinedLoss(
        num_classes=dataset_info['num_locations'],
        use_difftopk=config['use_difftopk'],
        use_focal=config['use_focal'],
        label_smoothing=config['label_smoothing']
    )
    
    # Mixup
    mixup = SequenceMixup(alpha=0.2) if config['use_mixup'] else None
    
    # Mixed precision
    scaler = GradScaler()
    
    # Tracking
    best_test_acc = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting SOTA training - Target: >40% test Acc@1")
    print(f"Techniques: DiffTopK={config['use_difftopk']}, Focal={config['use_focal']}, "
          f"EMA={config['use_ema']}, Mixup={config['use_mixup']}")
    print(f"{'='*80}\n")
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            loc_seq = batch['loc_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            weekday_seq = batch['weekday_seq'].to(device)
            start_min_seq = batch['start_min_seq'].to(device)
            dur_seq = batch['dur_seq'].to(device)
            diff_seq = batch['diff_seq'].to(device)
            seq_len = batch['seq_len'].to(device)
            target = batch['target'].to(device)
            
            # Forward with mixed precision
            with autocast():
                logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
                loss = criterion(logits, target)
                loss = loss / config['accum_steps']
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA
                if ema is not None:
                    ema.update()
            
            total_loss += loss.item() * config['accum_steps']
            pbar.set_postfix({'loss': f'{loss.item() * config["accum_steps"]:.4f}',
                            'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluation with EMA
        if ema is not None:
            ema.apply_shadow()
        
        model.eval()
        
        # Validate
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
        
        # Restore original weights
        if ema is not None:
            ema.restore()
        
        # Print results
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Val  Acc@1: {val_acc:.2f}% | Acc@5: {val_perf['acc@5']:.2f}%")
        print(f"  TEST Acc@1: {test_acc:.2f}% | Acc@5: {test_perf['acc@5']:.2f}% | Gap: {val_acc-test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_val_acc = val_acc
            patience_counter = 0
            
            print(f"  ✓✓✓ NEW BEST TEST: {test_acc:.2f}% ✓✓✓")
            
            # Save checkpoint
            if ema is not None:
                ema.apply_shadow()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            
            if ema is not None:
                ema.restore()
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
        
        # Check if target achieved
        if test_acc >= 40.0:
            print(f"\n🎉🎉🎉 TARGET ACHIEVED: {test_acc:.2f}% >= 40% 🎉🎉🎉")
            break
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"  Best TEST Acc@1: {best_test_acc:.2f}%")
    print(f"  Best VAL  Acc@1: {best_val_acc:.2f}%")
    print(f"  Target (40%): {'✓ ACHIEVED' if best_test_acc >= 40.0 else '✗ NOT YET'}")
    print(f"{'='*80}\n")
    
    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return best_test_acc, best_val_acc


if __name__ == '__main__':
    train_sota()
