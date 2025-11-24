"""
Improved dataset loader with automatic parameter inference and multi-dataset support.

Supports:
- Geolife dataset
- DIY dataset
- Automatic vocabulary size inference
- Dynamic max sequence length computation
- Proper normalization statistics
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    Generic trajectory dataset for next-location prediction.
    
    Automatically infers dataset statistics and vocabulary sizes.
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: Optional[int] = None,
        normalization_stats: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            data_path: Path to pickle file
            max_seq_len: Maximum sequence length (None = auto-infer from data)
            normalization_stats: Pre-computed normalization statistics
        """
        self.data_path = data_path
        
        # Load data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        # Compute dataset statistics
        self.stats = self._compute_statistics()
        
        # Use provided normalization stats or compute new ones
        if normalization_stats:
            self.normalization_stats = normalization_stats
            logger.info("Using provided normalization statistics")
        else:
            self.normalization_stats = self._compute_normalization_stats()
            logger.info("Computed normalization statistics from data")
        
        # Set max sequence length
        if max_seq_len is None:
            self.max_seq_len = self.stats['max_seq_len']
            logger.info(f"Auto-inferred max_seq_len: {self.max_seq_len}")
        else:
            self.max_seq_len = max_seq_len
            logger.info(f"Using provided max_seq_len: {self.max_seq_len}")
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        all_locs = set()
        all_users = set()
        seq_lengths = []
        
        for sample in self.data:
            # Collect location vocabulary
            all_locs.update(sample['X'])
            all_locs.add(sample['Y'])
            
            # Collect user vocabulary
            all_users.update(sample['user_X'])
            
            # Collect sequence lengths
            seq_lengths.append(len(sample['X']))
        
        stats = {
            'num_samples': len(self.data),
            'num_locations': max(all_locs) + 1,  # +1 because indices start at 0
            'num_users': max(all_users) + 1,
            'max_seq_len': max(seq_lengths),
            'min_seq_len': min(seq_lengths),
            'avg_seq_len': np.mean(seq_lengths),
            'median_seq_len': np.median(seq_lengths),
            'unique_locations': len(all_locs),
            'unique_users': len(all_users)
        }
        
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats
    
    def _compute_normalization_stats(self) -> Dict[str, float]:
        """
        Compute normalization statistics for continuous features.
        
        Returns:
            Dictionary containing normalization parameters
        """
        all_durs = []
        all_start_mins = []
        all_diffs = []
        
        for sample in self.data:
            all_durs.extend(sample['dur_X'])
            all_start_mins.extend(sample['start_min_X'])
            if 'diff' in sample:
                all_diffs.extend(sample['diff'])
        
        # Compute statistics with numerical stability
        stats = {
            'dur_mean': float(np.mean(all_durs)),
            'dur_std': float(np.std(all_durs) + 1e-8),
            'dur_min': float(np.min(all_durs)),
            'dur_max': float(np.max(all_durs)),
            'start_min_mean': float(np.mean(all_start_mins)),
            'start_min_std': float(np.std(all_start_mins) + 1e-8),
            'start_min_min': float(np.min(all_start_mins)),
            'start_min_max': float(np.max(all_start_mins)),
        }
        
        if all_diffs:
            stats.update({
                'diff_mean': float(np.mean(all_diffs)),
                'diff_std': float(np.std(all_diffs) + 1e-8),
            })
        
        logger.info("Normalization Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats
    
    def get_normalization_stats(self) -> Dict[str, float]:
        """Get normalization statistics."""
        return self.normalization_stats
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tensors for all features
        """
        sample = self.data[idx]
        
        # Get sequence data
        loc_seq = torch.LongTensor(sample['X'])
        user_seq = torch.LongTensor(sample['user_X'])
        weekday_seq = torch.LongTensor(sample['weekday_X'])
        start_min_seq = torch.FloatTensor(sample['start_min_X'])
        dur_seq = torch.FloatTensor(sample['dur_X'])
        diff_seq = torch.LongTensor(sample['diff'])
        
        # Normalize continuous features
        norm_stats = self.normalization_stats
        start_min_seq = (start_min_seq - norm_stats['start_min_mean']) / norm_stats['start_min_std']
        dur_seq = (dur_seq - norm_stats['dur_mean']) / norm_stats['dur_std']
        
        # Target
        target = torch.LongTensor([sample['Y']])[0]
        
        # Truncate if needed (keep most recent)
        if self.max_seq_len is not None and len(loc_seq) > self.max_seq_len:
            loc_seq = loc_seq[-self.max_seq_len:]
            user_seq = user_seq[-self.max_seq_len:]
            weekday_seq = weekday_seq[-self.max_seq_len:]
            start_min_seq = start_min_seq[-self.max_seq_len:]
            dur_seq = dur_seq[-self.max_seq_len:]
            diff_seq = diff_seq[-self.max_seq_len:]
        
        seq_len = len(loc_seq)
        
        return {
            'loc_seq': loc_seq,
            'user_seq': user_seq,
            'weekday_seq': weekday_seq,
            'start_min_seq': start_min_seq,
            'dur_seq': dur_seq,
            'diff_seq': diff_seq,
            'seq_len': seq_len,
            'target': target
        }


def collate_fn(batch):
    """
    Collate function for batching variable-length sequences.
    
    Pads sequences to the maximum length in the batch.
    """
    # Find max length in batch
    max_len = max([item['seq_len'] for item in batch])
    batch_size = len(batch)
    
    # Initialize tensors
    loc_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    user_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    weekday_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    start_min_seqs = torch.zeros(batch_size, max_len, dtype=torch.float)
    dur_seqs = torch.zeros(batch_size, max_len, dtype=torch.float)
    diff_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors (pad sequences)
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        loc_seqs[i, :seq_len] = item['loc_seq']
        user_seqs[i, :seq_len] = item['user_seq']
        weekday_seqs[i, :seq_len] = item['weekday_seq']
        start_min_seqs[i, :seq_len] = item['start_min_seq']
        dur_seqs[i, :seq_len] = item['dur_seq']
        diff_seqs[i, :seq_len] = item['diff_seq']
        seq_lens[i] = seq_len
        targets[i] = item['target']
    
    return {
        'loc_seq': loc_seqs,
        'user_seq': user_seqs,
        'weekday_seq': weekday_seqs,
        'start_min_seq': start_min_seqs,
        'dur_seq': dur_seqs,
        'diff_seq': diff_seqs,
        'seq_len': seq_lens,
        'target': targets
    }


def get_dataset_paths(dataset_name: str, data_dir: str) -> Tuple[str, str, str]:
    """
    Get dataset paths for train/val/test splits.
    
    Args:
        dataset_name: Name of dataset ("geolife" or "diy")
        data_dir: Root data directory
        
    Returns:
        Tuple of (train_path, val_path, test_path)
        
    Raises:
        ValueError: If dataset not found
    """
    data_dir = Path(data_dir)
    
    if dataset_name == "geolife":
        dataset_dir = data_dir / "geolife"
        train_path = dataset_dir / "geolife_transformer_7_train.pk"
        val_path = dataset_dir / "geolife_transformer_7_validation.pk"
        test_path = dataset_dir / "geolife_transformer_7_test.pk"
    elif dataset_name == "diy":
        dataset_dir = data_dir / "diy"
        train_path = dataset_dir / "diy_h3_res8_transformer_7_train.pk"
        val_path = dataset_dir / "diy_h3_res8_transformer_7_validation.pk"
        test_path = dataset_dir / "diy_h3_res8_transformer_7_test.pk"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'geolife', 'diy'")
    
    # Verify files exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
    
    logger.info(f"Dataset paths for '{dataset_name}':")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Val: {val_path}")
    logger.info(f"  Test: {test_path}")
    
    return str(train_path), str(val_path), str(test_path)


def create_dataloaders(
    dataset_name: str,
    data_dir: str = "data",
    batch_size: int = 64,
    max_seq_len: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create dataloaders with automatic parameter inference.
    
    Args:
        dataset_name: Name of dataset ("geolife" or "diy")
        data_dir: Root data directory
        batch_size: Batch size
        max_seq_len: Maximum sequence length (None = auto-infer)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info)
    """
    # Get dataset paths
    train_path, val_path, test_path = get_dataset_paths(dataset_name, data_dir)
    
    # Create train dataset first to get statistics
    train_dataset = TrajectoryDataset(train_path, max_seq_len=max_seq_len)
    
    # Get normalization stats from train set
    norm_stats = train_dataset.get_normalization_stats()
    
    # Create val and test datasets with train normalization stats
    val_dataset = TrajectoryDataset(
        val_path,
        max_seq_len=train_dataset.max_seq_len,
        normalization_stats=norm_stats
    )
    test_dataset = TrajectoryDataset(
        test_path,
        max_seq_len=train_dataset.max_seq_len,
        normalization_stats=norm_stats
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Compile dataset information
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    test_stats = test_dataset.get_statistics()
    
    dataset_info = {
        'dataset_name': dataset_name,
        'num_locations': train_stats['num_locations'],
        'num_users': train_stats['num_users'],
        'unique_locations': train_stats['unique_locations'],
        'unique_users': train_stats['unique_users'],
        'max_seq_len': train_dataset.max_seq_len,
        'train_size': train_stats['num_samples'],
        'val_size': val_stats['num_samples'],
        'test_size': test_stats['num_samples'],
        'train_avg_seq_len': train_stats['avg_seq_len'],
        'normalization_stats': norm_stats
    }
    
    logger.info("=" * 50)
    logger.info("Dataset Information:")
    for key, value in dataset_info.items():
        if key != 'normalization_stats':
            logger.info(f"  {key}: {value}")
    logger.info("=" * 50)
    
    return train_loader, val_loader, test_loader, dataset_info
