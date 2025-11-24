"""
Unit tests for metrics calculation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from utils.metrics_v2 import (
    calculate_accuracy,
    calculate_mrr,
    calculate_metrics
)


def test_accuracy():
    """Test accuracy calculation."""
    # Perfect predictions
    logits = torch.tensor([
        [0.1, 0.2, 0.7],  # Pred: 2
        [0.8, 0.1, 0.1],  # Pred: 0
        [0.1, 0.7, 0.2],  # Pred: 1
    ])
    targets = torch.tensor([2, 0, 1])
    
    acc = calculate_accuracy(logits, targets, k=1)
    assert acc == 100.0, f"Expected 100% accuracy, got {acc}%"
    
    # Partial predictions
    logits = torch.tensor([
        [0.1, 0.2, 0.7],  # Pred: 2, Target: 2 ✓
        [0.8, 0.1, 0.1],  # Pred: 0, Target: 1 ✗
        [0.1, 0.7, 0.2],  # Pred: 1, Target: 1 ✓
    ])
    targets = torch.tensor([2, 1, 1])
    
    acc = calculate_accuracy(logits, targets, k=1)
    expected = 200.0 / 3.0
    assert abs(acc - expected) < 0.01, f"Expected {expected}% accuracy, got {acc}%"
    
    print("✓ Accuracy test passed")


def test_top_k_accuracy():
    """Test top-k accuracy calculation."""
    logits = torch.tensor([
        [0.1, 0.3, 0.6],  # Top-2: [2, 1], Target: 1 ✓ (in top-2)
        [0.5, 0.3, 0.2],  # Top-2: [0, 1], Target: 0 ✓ (rank 1)
        [0.2, 0.1, 0.7],  # Top-2: [2, 0], Target: 1 ✗ (not in top-2)
    ])
    targets = torch.tensor([1, 0, 1])
    
    # Top-1 accuracy: only targets[1] is rank 1
    acc1 = calculate_accuracy(logits, targets, k=1)
    expected_top1 = 100.0 * 1 / 3  # Only second prediction is top-1
    assert abs(acc1 - expected_top1) < 0.01, f"Expected {expected_top1}%, got {acc1}%"
    
    # Top-2 accuracy: targets[0] and targets[1] in top-2
    acc2 = calculate_accuracy(logits, targets, k=2)
    expected_top2 = 100.0 * 2 / 3  # Two predictions in top-2
    assert abs(acc2 - expected_top2) < 0.01, f"Expected {expected_top2}%, got {acc2}%"
    
    print("✓ Top-k accuracy test passed")


def test_mrr():
    """Test Mean Reciprocal Rank calculation."""
    # All predictions rank 1
    logits = torch.tensor([
        [0.1, 0.2, 0.7],  # Target 2 is rank 1
        [0.8, 0.1, 0.1],  # Target 0 is rank 1
    ])
    targets = torch.tensor([2, 0])
    
    mrr = calculate_mrr(logits, targets)
    assert mrr == 1.0, f"Expected MRR=1.0, got {mrr}"
    
    # Mixed ranks
    logits = torch.tensor([
        [0.7, 0.2, 0.1],  # Target 2 is rank 3, RR = 1/3
        [0.2, 0.7, 0.1],  # Target 1 is rank 1, RR = 1/1
        [0.1, 0.7, 0.2],  # Target 2 is rank 2, RR = 1/2
    ])
    targets = torch.tensor([2, 1, 2])
    
    mrr = calculate_mrr(logits, targets)
    expected_mrr = (1/3 + 1/1 + 1/2) / 3
    assert abs(mrr - expected_mrr) < 0.01, f"Expected MRR={expected_mrr}, got {mrr}"
    
    print("✓ MRR test passed")


def test_calculate_metrics():
    """Test comprehensive metrics calculation."""
    # Create test data
    batch_size = 10
    num_classes = 5
    
    # Random logits
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Calculate metrics
    metrics = calculate_metrics(logits, targets)
    
    # Check that all expected metrics are present
    expected_metrics = [
        'accuracy', 'top5_accuracy', 'top10_accuracy',
        'mrr', 'ndcg@5', 'ndcg@10',
        'precision', 'recall', 'f1_score'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], (int, float)), f"Metric {metric} is not numeric"
        assert 0 <= metrics[metric] <= 100 or 0 <= metrics[metric] <= 1, \
            f"Metric {metric} out of range: {metrics[metric]}"
    
    print("✓ Comprehensive metrics test passed")


def test_numerical_stability():
    """Test metrics calculation with edge cases."""
    # All predictions are the same
    logits = torch.ones(5, 3) * 0.5
    targets = torch.tensor([0, 1, 2, 0, 1])
    
    metrics = calculate_metrics(logits, targets)
    
    # Should not crash and should return valid values
    assert not np.isnan(metrics['accuracy'])
    assert not np.isnan(metrics['mrr'])
    
    print("✓ Numerical stability test passed")


if __name__ == '__main__':
    print("Running metrics tests...")
    test_accuracy()
    test_top_k_accuracy()
    test_mrr()
    test_calculate_metrics()
    test_numerical_stability()
    print("\n✓ All tests passed!")
