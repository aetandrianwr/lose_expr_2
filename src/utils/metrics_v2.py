"""
Comprehensive metrics calculation for next-location prediction.

Provides standard evaluation metrics with proper implementation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        k: Top-k value
        
    Returns:
        Top-k accuracy as percentage
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        num_classes = logits.size(1)
        
        # Handle case where k > num_classes
        k_actual = min(k, num_classes)
        
        _, pred = logits.topk(k_actual, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k_actual].reshape(-1).float().sum(0, keepdim=True)
        accuracy = correct_k.item() * 100.0 / batch_size
    return accuracy


def calculate_mrr(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR is the average of reciprocal ranks of the correct prediction.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        
    Returns:
        Mean Reciprocal Rank
    """
    with torch.no_grad():
        # Get sorted indices
        _, sorted_indices = torch.sort(logits, dim=1, descending=True)
        
        # Find rank of correct answer
        ranks = []
        for i, target in enumerate(targets):
            rank = (sorted_indices[i] == target).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
        
        mrr = np.mean(ranks)
    return mrr


def calculate_ndcg(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @ k.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        k: Cutoff for NDCG calculation
        
    Returns:
        NDCG@k score
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        num_classes = logits.size(1)
        
        # Handle case where k > num_classes
        k_actual = min(k, num_classes)
        
        _, top_k_indices = torch.topk(logits, k_actual, dim=1)
        
        dcg_scores = []
        for i, target in enumerate(targets):
            # Check if target is in top-k
            if target in top_k_indices[i]:
                rank = (top_k_indices[i] == target).nonzero(as_tuple=True)[0].item() + 1
                dcg = 1.0 / np.log2(rank + 1)
            else:
                dcg = 0.0
            dcg_scores.append(dcg)
        
        # IDCG is 1/log2(2) = 1.0 for single relevant item
        idcg = 1.0
        ndcg = np.mean(dcg_scores) / idcg
    
    return ndcg


def calculate_precision_recall_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'macro'
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        average: Averaging strategy ('macro', 'micro', 'weighted')
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average=average, zero_division=0
        )
    
    return float(precision), float(recall), float(f1)


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Accuracy metrics
    metrics['accuracy'] = calculate_accuracy(logits, targets, k=1)
    metrics['top5_accuracy'] = calculate_accuracy(logits, targets, k=5)
    metrics['top10_accuracy'] = calculate_accuracy(logits, targets, k=10)
    
    # Ranking metrics
    metrics['mrr'] = calculate_mrr(logits, targets)
    metrics['ndcg@5'] = calculate_ndcg(logits, targets, k=5)
    metrics['ndcg@10'] = calculate_ndcg(logits, targets, k=10)
    
    # Classification metrics
    precision, recall, f1 = calculate_precision_recall_f1(logits, targets, average='macro')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    return metrics


def get_performance_dict(correct: int, total: int) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    
    Args:
        correct: Number of correct predictions
        total: Total number of predictions
        
    Returns:
        Dictionary with accuracy
    """
    return {
        'accuracy': 100.0 * correct / total if total > 0 else 0.0
    }


def calculate_correct_total_prediction(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """
    Legacy function for backward compatibility.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        
    Returns:
        Tuple of (correct_count, total_count)
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
    return correct, total
