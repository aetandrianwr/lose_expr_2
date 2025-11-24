"""
Production-level training script for next-location prediction.

Features:
- Automatic parameter inference from dataset
- Reproducible experiments with seed setting
- Configuration-based training
- Comprehensive logging and experiment tracking
- Model checkpointing with best model saving
- Early stopping
- Mixed precision training
- Gradient clipping
- TensorBoard integration
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_v2 import create_dataloaders
from models.temporal_fusion import TemporalFusionModel
from utils.metrics_v2 import calculate_metrics
from utils.config import load_config, validate_config, save_config
from utils.reproducibility import set_seed, get_device, log_system_info
from utils.experiment_tracker import ExperimentTracker, setup_logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Comprehensive trainer for next-location prediction.
    
    Handles training loop, validation, testing, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_tracker: ExperimentTracker
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: PyTorch device
            experiment_tracker: Experiment tracking utility
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.tracker = experiment_tracker
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Mixed precision training
        self.use_amp = config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.best_test_metric = 0.0
        self.patience_counter = 0
        
        # Gradient clipping
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        
        logger.info(f"Trainer initialized")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from configuration."""
        opt_config = self.config['training']
        optimizer_name = opt_config.get('optimizer', 'adamw').lower()
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0.0),
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                eps=opt_config.get('eps', 1e-8)
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0.0),
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                eps=opt_config.get('eps', 1e-8)
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer: {optimizer_name}")
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler from configuration."""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config.get('type', 'onecycle').lower()
        
        if sched_type == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                epochs=self.config['training']['epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=sched_config.get('pct_start', 0.1),
                anneal_strategy=sched_config.get('anneal_strategy', 'cos'),
                div_factor=sched_config.get('div_factor', 25.0),
                final_div_factor=sched_config.get('final_div_factor', 10000.0)
            )
        elif sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('eta_min', 0)
            )
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                verbose=True
            )
        else:
            scheduler = None
            logger.warning(f"Unknown scheduler type: {sched_type}. Not using scheduler.")
        
        if scheduler:
            logger.info(f"LR Scheduler: {sched_type}")
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function from configuration."""
        loss_config = self.config['training']['loss']
        loss_type = loss_config.get('type', 'cross_entropy').lower()
        
        if loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        logger.info(f"Loss function: {loss_type} (label_smoothing={loss_config.get('label_smoothing', 0.0)})")
        return criterion
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.config["training"]["epochs"]} [Train]',
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            loc_seq = batch['loc_seq'].to(self.device)
            user_seq = batch['user_seq'].to(self.device)
            weekday_seq = batch['weekday_seq'].to(self.device)
            start_min_seq = batch['start_min_seq'].to(self.device)
            dur_seq = batch['dur_seq'].to(self.device)
            diff_seq = batch['diff_seq'].to(self.device)
            seq_len = batch['seq_len'].to(self.device)
            target = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(
                        loc_seq, user_seq, weekday_seq,
                        start_min_seq, dur_seq, diff_seq, seq_len
                    )
                    loss = self.criterion(logits, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    loc_seq, user_seq, weekday_seq,
                    start_min_seq, dur_seq, diff_seq, seq_len
                )
                loss = self.criterion(logits, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Update learning rate (for OneCycleLR)
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Calculate accuracy
            pred = logits.argmax(dim=1)
            correct = (pred == target).sum().item()
            
            # Update metrics
            total_loss += loss.item() * target.size(0)
            total_correct += correct
            total_samples += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / target.size(0)
            })
            
            # Log to TensorBoard
            if batch_idx % self.config['logging'].get('log_interval', 10) == 0:
                self.tracker.log_metric('train/batch_loss', loss.item(), self.global_step)
                self.tracker.log_metric('train/batch_acc', 100.0 * correct / target.size(0), self.global_step)
                self.tracker.log_metric('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Epoch metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader, split_name: str = 'val') -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of the split (for logging)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(data_loader, desc=f'Evaluating [{split_name}]', leave=False)
        
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
            loss = self.criterion(logits, target)
            
            # Collect predictions
            total_loss += loss.item() * target.size(0)
            all_preds.append(logits.cpu())
            all_targets.append(target.cpu())
        
        # Concatenate all predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(all_targets)
        
        return metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        if is_best:
            best_path = Path(filepath).parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing final results
        """
        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info("=" * 50)
        
        early_stop_config = self.config['training']['early_stopping']
        patience = early_stop_config.get('patience', 15)
        min_delta = early_stop_config.get('min_delta', 0.0001)
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, 'val')
            
            # Update scheduler (for non-OneCycle schedulers)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.tracker.log_metrics({
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/top5_accuracy': val_metrics.get('top5_accuracy', 0.0),
            }, step=epoch)
            
            # Print epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_metric + min_delta
            if is_best:
                self.best_val_metric = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Test on best validation model
                test_metrics = self.evaluate(self.test_loader, 'test')
                self.best_test_metric = test_metrics['accuracy']
                
                logger.info(
                    f"*** New best model! Val Acc: {self.best_val_metric:.2f}%, "
                    f"Test Acc: {self.best_test_metric:.2f}% ***"
                )
                
                # Save best model
                checkpoint_path = self.tracker.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(str(checkpoint_path), is_best=True)
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training'].get('save_frequency', 5) == 0:
                checkpoint_path = self.tracker.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(str(checkpoint_path), is_best=False)
            
            # Early stopping
            if early_stop_config.get('enabled', True) and self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Final evaluation on test set
        logger.info("=" * 50)
        logger.info("Training completed. Running final test evaluation...")
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"Best Val Accuracy: {self.best_val_metric:.2f}%")
        logger.info(f"Best Test Accuracy (at best val): {self.best_test_metric:.2f}%")
        logger.info("=" * 50)
        
        return {
            'best_val_accuracy': self.best_val_metric,
            'best_test_accuracy': self.best_test_metric,
            'final_test_accuracy': test_metrics['accuracy'],
            'final_test_metrics': test_metrics
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train next-location prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['geolife', 'diy'],
                       help='Dataset to use (overrides config)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.dataset:
        config['data']['dataset_name'] = args.dataset
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device:
        config['hardware']['device'] = args.device
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Validate configuration
    validate_config(config)
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Log system information
    log_system_info()
    
    # Set random seed for reproducibility
    set_seed(
        config['seed'],
        cuda_deterministic=config['hardware'].get('cuda_deterministic', True)
    )
    
    # Get device
    device = get_device(config['hardware']['device'])
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_name=config['experiment']['name'],
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging'].get('tensorboard', True),
        config=config
    )
    
    try:
        # Load data
        logger.info("Loading dataset...")
        train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
            dataset_name=config['data']['dataset_name'],
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['batch_size'],
            max_seq_len=config['data'].get('max_seq_len'),
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True)
        )
        
        # Update config with inferred parameters
        config['data']['num_locations'] = dataset_info['num_locations']
        config['data']['num_users'] = dataset_info['num_users']
        config['data']['max_seq_len'] = dataset_info['max_seq_len']
        
        # Create model
        logger.info("Creating model...")
        model = TemporalFusionModel(
            num_locations=dataset_info['num_locations'],
            num_users=dataset_info['num_users'],
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            experiment_tracker=tracker
        )
        
        # Train model
        results = trainer.train()
        
        # Log final results
        tracker.log_metrics(results)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        tracker.close()


if __name__ == '__main__':
    main()
