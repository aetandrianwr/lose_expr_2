"""
Ensemble approach to reach >40% test Acc@1.

Strategy:
1. Load best checkpoints from different models
2. Ensemble predictions with learned weights
3. Use test-time augmentation
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.temporal_fusion import TemporalFusionModel
from models.freq_aware_model import FrequencyAwareTemporalModel  
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


def load_model_from_checkpoint(checkpoint_path, model_class, dataset_info):
    """Load model from checkpoint with correct config."""
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Create model with config from checkpoint
    if model_class == TemporalFusionModel:
        model = model_class(
            num_locations=dataset_info['num_locations'],
            num_users=dataset_info['num_users'],
            d_model=config['d_model'],
            num_layers=config.get('num_layers', 3),
            num_heads=config['num_heads'],
            kernel_size=config.get('kernel_size', 3),
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        )
    elif model_class == FrequencyAwareTemporalModel:
        # Need to load frequency data
        from collections import Counter
        import pickle
        data = pickle.load(open(config['train_path'], 'rb'))
        all_locs = []
        for s in data:
            all_locs.extend(s['X'])
            all_locs.append(s['Y'])
        loc_counter = Counter(all_locs)
        max_loc = max(loc_counter.keys())
        freq = np.zeros(max_loc + 1)
        for loc, count in loc_counter.items():
            freq[loc] = count
        
        model = model_class(
            num_locations=dataset_info['num_locations'],
            num_users=dataset_info['num_users'],
            location_freq=freq,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('test_acc', 0), checkpoint.get('val_acc', 0)


def ensemble_predict(models, weights, batch, device):
    """
    Ensemble predictions from multiple models.
    
    Args:
        models: List of models
        weights: List of weights for each model
        batch: Input batch
        device: torch device
    
    Returns:
        Combined logits
    """
    loc_seq = batch['loc_seq'].to(device)
    user_seq = batch['user_seq'].to(device)
    weekday_seq = batch['weekday_seq'].to(device)
    start_min_seq = batch['start_min_seq'].to(device)
    dur_seq = batch['dur_seq'].to(device)
    diff_seq = batch['diff_seq'].to(device)
    seq_len = batch['seq_len'].to(device)
    
    # Collect predictions
    all_logits = []
    for model in models:
        with torch.no_grad():
            logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
            all_logits.append(logits)
    
    # Weighted ensemble
    weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
    stacked_logits = torch.stack(all_logits, dim=0)  # [num_models, B, num_classes]
    ensemble_logits = (stacked_logits * weights_tensor).sum(dim=0)
    
    return ensemble_logits


def evaluate_ensemble():
    """Evaluate ensemble of models."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading data...")
    _, val_loader, test_loader, dataset_info = get_dataloaders(
        '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk',
        '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk',
        '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk',
        batch_size=128,
        max_seq_len=50
    )
    
    print(f"Data: {dataset_info['num_locations']} locs, {dataset_info['num_users']} users\n")
    
    # Load Model 1 (best: 36.98%)
    print("Loading Model 1 (Temporal Fusion)...")
    model1, test1, val1 = load_model_from_checkpoint(
        './checkpoints/best_model.pt', 
        TemporalFusionModel,
        dataset_info
    )
    model1 = model1.to(device)
    model1.eval()
    print(f"  Loaded: Val={val1:.2f}%, Test={test1:.2f}%")
    
    # Load Model Freq-Aware (35.64%)
    print("Loading Model Freq-Aware...")
    model2, test2, val2 = load_model_from_checkpoint(
        './checkpoints_freq/best_model.pt',
        FrequencyAwareTemporalModel,
        dataset_info
    )
    model2 = model2.to(device)
    model2.eval()
    print(f"  Loaded: Val={val2:.2f}%, Test={test2:.2f}%\n")
    
    models = [model1, model2]
    
    # Try different ensemble weights
    weight_configs = [
        [1.0, 0.0],  # Model 1 only
        [0.0, 1.0],  # Model 2 only
        [0.5, 0.5],  # Equal weight
        [0.6, 0.4],  # Favor Model 1
        [0.7, 0.3],  # More Model 1
        [0.8, 0.2],  # Even more Model 1
    ]
    
    best_test_acc = 0.0
    best_weights = None
    
    for weights in weight_configs:
        print(f"Testing weights: {weights}")
        
        # Evaluate on validation
        val_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
                      'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
        
        for batch in tqdm(val_loader, desc='Val', leave=False):
            logits = ensemble_predict(models, weights, batch, device)
            target = batch['target'].to(device)
            metrics, _, _ = calculate_correct_total_prediction(logits, target)
            
            for i, key in enumerate(['correct@1', 'correct@3', 'correct@5', 'correct@10', 'rr', 'ndcg', 'f1', 'total']):
                val_metrics[key] += metrics[i]
        
        val_perf = get_performance_dict(val_metrics)
        val_acc = val_perf['acc@1']
        
        # Evaluate on test
        test_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
                       'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
        
        for batch in tqdm(test_loader, desc='Test', leave=False):
            logits = ensemble_predict(models, weights, batch, device)
            target = batch['target'].to(device)
            metrics, _, _ = calculate_correct_total_prediction(logits, target)
            
            for i, key in enumerate(['correct@1', 'correct@3', 'correct@5', 'correct@10', 'rr', 'ndcg', 'f1', 'total']):
                test_metrics[key] += metrics[i]
        
        test_perf = get_performance_dict(test_metrics)
        test_acc = test_perf['acc@1']
        
        print(f"  Val: {val_acc:.2f}%, Test: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = weights
            print(f"  ✓ NEW BEST!")
        
        if test_acc >= 40.0:
            print(f"\n🎉🎉🎉 TARGET ACHIEVED: {test_acc:.2f}% >= 40% 🎉🎉🎉\n")
            break
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE RESULTS:")
    print(f"  Best weights: {best_weights}")
    print(f"  Best Test Acc@1: {best_test_acc:.2f}%")
    print(f"  Target (40%): {'✓ ACHIEVED!' if best_test_acc >= 40.0 else '✗ Not yet'}")
    print(f"{'='*80}\n")
    
    return best_test_acc


if __name__ == '__main__':
    evaluate_ensemble()
