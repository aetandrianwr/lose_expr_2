"""
Training script for advanced next-location prediction model
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.advanced_model import AdvancedNextLocationModel
from utils.metrics import calculate_correct_total_prediction, get_performance_dict
from utils.location_clustering import cluster_locations, compute_location_frequency_buckets


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_class_weights(train_path, num_locations, device):
    """
    Create class weights based on inverse frequency
    """
    import pickle
    from collections import Counter

    with open(train_path, 'rb') as f:
        data = pickle.load(f)

    target_counts = Counter(sample['Y'] for sample in data)

    # Compute weights: inverse frequency with smoothing
    weights = np.ones(num_locations)
    total = len(data)

    for loc in range(num_locations):
        count = target_counts.get(loc, 0)
        if count > 0:
            weights[loc] = total / (count * num_locations)

    # Normalize and apply sqrt to reduce extreme weights
    weights = np.sqrt(weights)
    weights = weights / weights.mean()

    return torch.FloatTensor(weights).to(device)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        optimizer.zero_grad()

        # Move to device
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        seq_len = batch['seq_len'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, data_loader, device, desc="Val"):
    """
    Evaluate model
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(data_loader, desc=f"[{desc}]")
    for batch in pbar:
        # Move to device
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        seq_len = batch['seq_len'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)

        # Compute loss
        loss = criterion(logits, targets)
        total_loss += loss.item()
        num_batches += 1

        # Store predictions
        all_predictions.append(logits.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics
    result_array, _, _ = calculate_correct_total_prediction(all_predictions, all_targets)

    # Convert to dict format
    result_dict = {
        "correct@1": result_array[0],
        "correct@3": result_array[1],
        "correct@5": result_array[2],
        "correct@10": result_array[3],
        "rr": result_array[4],
        "ndcg": result_array[5],
        "f1": result_array[6],
        "total": result_array[7]
    }
    metrics = get_performance_dict(result_dict)

    avg_loss = total_loss / num_batches

    return metrics, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-clusters', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--use-focal-loss', action='store_true')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_advanced')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        args.train_path, args.val_path, args.test_path,
        batch_size=args.batch_size,
        max_seq_len=None
    )

    num_locations = dataset_info['num_locations']
    num_users = dataset_info['num_users']
    max_seq_len = dataset_info['max_seq_len']

    print(f"Dataset: {num_locations} locations, {num_users} users, max_seq_len={max_seq_len}")

    # Setup location clustering
    print("Setting up location clustering...")
    loc_to_cluster, _ = cluster_locations(args.train_path, num_locations, args.num_clusters)

    # Setup frequency buckets
    print("Computing location frequencies...")
    loc_freq_bucket, location_freq = compute_location_frequency_buckets(args.train_path, num_locations)

    # Create model
    print("Creating model...")
    model = AdvancedNextLocationModel(
        num_locations=num_locations,
        num_users=num_users,
        location_freq=location_freq,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
        num_clusters=args.num_clusters
    ).to(device)

    # Set clustering information
    model.loc_embed.loc_to_cluster.copy_(loc_to_cluster)
    model.loc_embed.loc_freq_bucket.copy_(loc_freq_bucket)

    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")

    if num_params >= 500000:
        print(f"⚠ Warning: Model has {num_params:,} parameters (limit: 500K)")

    # Setup loss function
    if args.use_focal_loss:
        class_weights = create_class_weights(args.train_path, num_locations, device)
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Evaluate
        val_metrics, val_loss = evaluate(model, val_loader, device, "Val")
        test_metrics, test_loss = evaluate(model, test_loader, device, "Test")

        # Print results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Acc@1: {val_metrics['acc@1']:.2f}%")
        print(f"  Test Acc@1: {test_metrics['acc@1']:.2f}%, Acc@5: {test_metrics['acc@5']:.2f}%")

        # Check for improvement
        if val_metrics['acc@1'] > best_val_acc:
            best_val_acc = val_metrics['acc@1']
            best_test_acc = test_metrics['acc@1']
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc@1'],
                'test_acc': test_metrics['acc@1'],
                'args': vars(args)
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ New best! Saved checkpoint.")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered!")
                break

        # Step scheduler
        scheduler.step()

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best Val Acc@1: {best_val_acc:.2f}%")
    print(f"Best Test Acc@1: {best_test_acc:.2f}%")
    print(f"{'='*80}")

    # Save config
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['best_test_acc'] = best_test_acc
    config['num_parameters'] = num_params

    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
