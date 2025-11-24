"""
Geolife dataset loader.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GeolifeDataset(Dataset):
    """
    Geolife trajectory dataset for next-location prediction.
    
    Each sample contains:
        - X: sequence of location IDs
        - user_X: user IDs for each location
        - weekday_X: weekday (0-6) for each visit
        - start_min_X: start time in minutes of day
        - dur_X: duration of visit
        - diff: time difference features
        - Y: next location (target)
    """
    
    def __init__(self, data_path, max_seq_len=None):
        """
        Args:
            data_path: Path to pickle file
            max_seq_len: Maximum sequence length (None = no truncation)
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_seq_len = max_seq_len
        
        # Compute dataset statistics
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute dataset statistics for normalization."""
        all_durs = []
        all_start_mins = []
        
        for sample in self.data:
            all_durs.extend(sample['dur_X'])
            all_start_mins.extend(sample['start_min_X'])
        
        self.dur_mean = np.mean(all_durs)
        self.dur_std = np.std(all_durs) + 1e-8
        self.start_min_mean = np.mean(all_start_mins)
        self.start_min_std = np.std(all_start_mins) + 1e-8
        
        # Get vocabulary sizes
        all_locs = set()
        all_users = set()
        for sample in self.data:
            all_locs.update(sample['X'])
            all_locs.add(sample['Y'])
            all_users.update(sample['user_X'])
        
        self.num_locations = max(all_locs) + 1
        self.num_users = max(all_users) + 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get sequence data
        loc_seq = torch.LongTensor(sample['X'])
        user_seq = torch.LongTensor(sample['user_X'])
        weekday_seq = torch.LongTensor(sample['weekday_X'])
        start_min_seq = torch.FloatTensor(sample['start_min_X'])
        dur_seq = torch.FloatTensor(sample['dur_X'])
        diff_seq = torch.LongTensor(sample['diff'])
        
        # Normalize continuous features
        start_min_seq = (start_min_seq - self.start_min_mean) / self.start_min_std
        dur_seq = (dur_seq - self.dur_mean) / self.dur_std
        
        # Target
        target = torch.LongTensor([sample['Y']])[0]
        
        # Truncate if needed
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


def get_dataloaders(train_path, val_path, test_path, batch_size=64, max_seq_len=None):
    """
    Create dataloaders for training, validation, and test sets.
    
    Args:
        train_path: Path to training pickle file
        val_path: Path to validation pickle file
        test_path: Path to test pickle file
        batch_size: Batch size
        max_seq_len: Maximum sequence length
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    train_dataset = GeolifeDataset(train_path, max_seq_len=max_seq_len)
    val_dataset = GeolifeDataset(val_path, max_seq_len=max_seq_len)
    test_dataset = GeolifeDataset(test_path, max_seq_len=max_seq_len)
    
    # Use train dataset stats for all
    val_dataset.dur_mean = train_dataset.dur_mean
    val_dataset.dur_std = train_dataset.dur_std
    val_dataset.start_min_mean = train_dataset.start_min_mean
    val_dataset.start_min_std = train_dataset.start_min_std
    
    test_dataset.dur_mean = train_dataset.dur_mean
    test_dataset.dur_std = train_dataset.dur_std
    test_dataset.start_min_mean = train_dataset.start_min_mean
    test_dataset.start_min_std = train_dataset.start_min_std
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Compute actual max sequence length from all datasets if not provided
    if max_seq_len is None:
        max_seq_lens = []
        for sample in train_dataset.data:
            max_seq_lens.append(len(sample['X']))
        for sample in val_dataset.data:
            max_seq_lens.append(len(sample['X']))
        for sample in test_dataset.data:
            max_seq_lens.append(len(sample['X']))
        inferred_max_seq_len = max(max_seq_lens)
    else:
        inferred_max_seq_len = max_seq_len

    dataset_info = {
        'num_locations': train_dataset.num_locations,
        'num_users': train_dataset.num_users,
        'max_seq_len': inferred_max_seq_len,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }

    return train_loader, val_loader, test_loader, dataset_info
