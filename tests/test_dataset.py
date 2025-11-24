"""
Unit tests for dataset loading and statistics computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
import pickle
import tempfile
from data.dataset_v2 import TrajectoryDataset, create_dataloaders, collate_fn


def create_dummy_dataset(num_samples=10):
    """Create a dummy dataset for testing."""
    data = []
    for i in range(num_samples):
        sample = {
            'X': list(range(i + 1, i + 6)),  # 5 locations
            'user_X': [0] * 5,
            'weekday_X': [1, 2, 3, 4, 5],
            'start_min_X': [60.0 * j for j in range(5)],
            'dur_X': [15.0, 20.0, 30.0, 25.0, 10.0],
            'diff': [1, 2, 3, 4, 5],
            'Y': i + 10
        }
        data.append(sample)
    return data


def test_dataset_statistics():
    """Test that dataset statistics are computed correctly."""
    # Create dummy data
    data = create_dummy_dataset(10)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pk') as f:
        pickle.dump(data, f)
        temp_path = f.name
    
    try:
        # Load dataset
        dataset = TrajectoryDataset(temp_path)
        
        # Check statistics
        stats = dataset.get_statistics()
        assert stats['num_samples'] == 10
        assert stats['max_seq_len'] == 5
        assert stats['min_seq_len'] == 5
        
        # Check normalization stats
        norm_stats = dataset.get_normalization_stats()
        assert 'dur_mean' in norm_stats
        assert 'dur_std' in norm_stats
        assert 'start_min_mean' in norm_stats
        assert 'start_min_std' in norm_stats
        
        print("✓ Dataset statistics test passed")
    finally:
        Path(temp_path).unlink()


def test_dataset_getitem():
    """Test that __getitem__ returns correct format."""
    data = create_dummy_dataset(10)
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pk') as f:
        pickle.dump(data, f)
        temp_path = f.name
    
    try:
        dataset = TrajectoryDataset(temp_path)
        
        # Get first sample
        sample = dataset[0]
        
        # Check keys
        required_keys = ['loc_seq', 'user_seq', 'weekday_seq', 'start_min_seq',
                        'dur_seq', 'diff_seq', 'seq_len', 'target']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        
        # Check types
        assert isinstance(sample['loc_seq'], torch.Tensor)
        assert isinstance(sample['target'], torch.Tensor)
        assert sample['loc_seq'].dtype == torch.long
        
        # Check sequence length
        assert sample['seq_len'] == len(sample['loc_seq'])
        
        print("✓ Dataset __getitem__ test passed")
    finally:
        Path(temp_path).unlink()


def test_collate_function():
    """Test that collate function properly pads sequences."""
    data = []
    # Create samples with different lengths
    for length in [3, 5, 7]:
        sample = {
            'X': list(range(length)),
            'user_X': [0] * length,
            'weekday_X': [1] * length,
            'start_min_X': [60.0] * length,
            'dur_X': [15.0] * length,
            'diff': [1] * length,
            'Y': 10
        }
        data.append(sample)
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pk') as f:
        pickle.dump(data, f)
        temp_path = f.name
    
    try:
        dataset = TrajectoryDataset(temp_path)
        
        # Get samples
        samples = [dataset[i] for i in range(3)]
        
        # Collate
        batch = collate_fn(samples)
        
        # Check batch format
        assert batch['loc_seq'].shape[0] == 3  # batch size
        assert batch['loc_seq'].shape[1] == 7  # max length
        
        # Check padding
        assert batch['seq_len'][0] == 3
        assert batch['seq_len'][1] == 5
        assert batch['seq_len'][2] == 7
        
        print("✓ Collate function test passed")
    finally:
        Path(temp_path).unlink()


def test_max_seq_len_truncation():
    """Test that max_seq_len properly truncates sequences."""
    data = [{
        'X': list(range(20)),  # Long sequence
        'user_X': [0] * 20,
        'weekday_X': [1] * 20,
        'start_min_X': [60.0] * 20,
        'dur_X': [15.0] * 20,
        'diff': [1] * 20,
        'Y': 10
    }]
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pk') as f:
        pickle.dump(data, f)
        temp_path = f.name
    
    try:
        # Load with max_seq_len
        dataset = TrajectoryDataset(temp_path, max_seq_len=10)
        
        # Get sample
        sample = dataset[0]
        
        # Check truncation
        assert sample['seq_len'] == 10
        assert len(sample['loc_seq']) == 10
        
        # Check that most recent items are kept (last 10)
        assert sample['loc_seq'][-1].item() == 19
        
        print("✓ Max sequence length truncation test passed")
    finally:
        Path(temp_path).unlink()


if __name__ == '__main__':
    print("Running dataset tests...")
    test_dataset_statistics()
    test_dataset_getitem()
    test_collate_function()
    test_max_seq_len_truncation()
    print("\n✓ All tests passed!")
