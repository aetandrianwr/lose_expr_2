"""
Training script for next-location prediction.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.temporal_fusion import TemporalFusionModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


class Trainer:
    """Trainer for next-location prediction."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - OneCycleLR for better convergence
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_accs = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for batch in pbar:
            # Move to device
            loc_seq = batch['loc_seq'].to(self.device)
            user_seq = batch['user_seq'].to(self.device)
            weekday_seq = batch['weekday_seq'].to(self.device)
            start_min_seq = batch['start_min_seq'].to(self.device)
            dur_seq = batch['dur_seq'].to(self.device)
            diff_seq = batch['diff_seq'].to(self.device)
            seq_len = batch['seq_len'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                logits = self.model(
                    loc_seq, user_seq, weekday_seq,
                    start_min_seq, dur_seq, diff_seq, seq_len
                )
                loss = self.criterion(logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, data_loader, split='Val'):
        """Evaluate on validation or test set."""
        self.model.eval()
        
        all_metrics = {
            'correct@1': 0,
            'correct@3': 0,
            'correct@5': 0,
            'correct@10': 0,
            'rr': 0,
            'ndcg': 0,
            'f1': 0,
            'total': 0
        }
        
        pbar = tqdm(data_loader, desc=f'[{split}]')
        
        for batch in pbar:
            # Move to device
            loc_seq = batch['loc_seq'].to(self.device)
            user_seq = batch['user_seq'].to(self.device)
            weekday_seq = batch['weekday_seq'].to(self.device)
            start_min_seq = batch['start_min_seq'].to(self.device)
            dur_seq = batch['dur_seq'].to(self.device)
            diff_seq = batch['diff_seq'].to(self.device)
            seq_len = batch['seq_len'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            logits = self.model(
                loc_seq, user_seq, weekday_seq,
                start_min_seq, dur_seq, diff_seq, seq_len
            )
            
            # Calculate metrics
            metrics, _, _ = calculate_correct_total_prediction(logits, target)
            
            all_metrics['correct@1'] += metrics[0]
            all_metrics['correct@3'] += metrics[1]
            all_metrics['correct@5'] += metrics[2]
            all_metrics['correct@10'] += metrics[3]
            all_metrics['rr'] += metrics[4]
            all_metrics['ndcg'] += metrics[5]
            all_metrics['f1'] += metrics[6]
            all_metrics['total'] += metrics[7]
        
        # Compute performance
        perf = get_performance_dict(all_metrics)
        
        return perf
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_perf = self.evaluate(self.val_loader, 'Val')
            val_acc = val_perf['acc@1']
            
            self.val_accs.append(val_acc)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Acc@1:  {val_acc:.2f}%")
            print(f"  Val Acc@5:  {val_perf['acc@5']:.2f}%")
            print(f"  Val MRR:    {val_perf['mrr']:.2f}%")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                # Test on best validation
                test_perf = self.evaluate(self.test_loader, 'Test')
                test_acc = test_perf['acc@1']
                
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc
                
                print(f"\n  ✓ New best Val Acc@1: {val_acc:.2f}%")
                print(f"  → Test Acc@1: {test_acc:.2f}%")
                print(f"  → Test Acc@5: {test_perf['acc@5']:.2f}%")
                print(f"  → Test MRR:   {test_perf['mrr']:.2f}%")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_acc, test_acc)
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config['patience']})")
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val Acc@1:  {self.best_val_acc:.2f}%")
        print(f"Best Test Acc@1: {self.best_test_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.best_val_acc, self.best_test_acc
    
    def save_checkpoint(self, epoch, val_acc, test_acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")


def main():
    """Main function."""
    # Configuration
    config = {
        # Data
        'train_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
        'val_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
        'test_path': '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
        'batch_size': 128,
        'max_seq_len': 50,
        
        # Model
        'd_model': 96,
        'num_layers': 2,
        'num_heads': 4,
        'kernel_size': 3,
        'dropout': 0.2,
        
        # Training
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'label_smoothing': 0.1,
        'patience': 15,
        
        # Paths
        'checkpoint_dir': './checkpoints'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    print(f"  Test: {dataset_info['test_size']}")
    
    # Create model
    print("\nCreating model...")
    model = TemporalFusionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    )
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    if num_params >= 500000:
        print(f"WARNING: Model has {num_params:,} parameters (>= 500K limit)")
        return
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
    best_val_acc, best_test_acc = trainer.train()
    
    # Save config
    with open('checkpoints/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"  Best Val Acc@1:  {best_val_acc:.2f}%")
    print(f"  Best Test Acc@1: {best_test_acc:.2f}%")


if __name__ == '__main__':
    main()
