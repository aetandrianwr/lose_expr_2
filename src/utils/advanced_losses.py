"""
Advanced loss functions for next-location prediction.
Combining multiple state-of-the-art techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import difftopk


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes]
            targets: [B]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss using multiple objectives for better top-1 accuracy.
    """
    
    def __init__(self, num_classes, use_difftopk=True, use_focal=True, 
                 label_smoothing=0.1, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        
        self.use_difftopk = use_difftopk
        self.use_focal = use_focal
        
        # Standard cross-entropy
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Focal loss for hard examples
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, 
                                       label_smoothing=label_smoothing)
        
        # DiffTopK loss for top-1 optimization
        if use_difftopk:
            try:
                # Use odd_even which doesn't require sparse operations
                self.difftopk_loss = difftopk.TopKCrossEntropyLoss(
                    diffsort_method='odd_even',
                    inverse_temperature=2.0,
                    p_k=[1.0, 0., 0., 0., 0.],  # Focus on top-1
                    n=num_classes,
                    m=min(32, num_classes),  # Use subset for efficiency
                    top1_mode='sm'
                )
            except Exception as e:
                print(f"DiffTopK initialization failed: {e}, using standard CE")
                self.use_difftopk = False
    
    def forward(self, logits, targets):
        """
        Compute combined loss.
        
        Args:
            logits: [B, num_classes]
            targets: [B]
        """
        # Standard CE loss (weighted lower)
        loss = 0.3 * self.ce_loss(logits, targets)
        
        # Focal loss (for hard examples)
        if self.use_focal:
            loss += 0.3 * self.focal_loss(logits, targets)
        
        # DiffTopK loss (for top-1 optimization)
        if self.use_difftopk:
            try:
                difftopk_loss = self.difftopk_loss(logits, targets)
                loss += 0.4 * difftopk_loss
            except Exception as e:
                # Fallback if difftopk fails
                pass
        
        return loss


class SequenceMixup:
    """
    Mixup augmentation for sequences.
    From: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Apply mixup to a batch.
        
        Args:
            batch: dict with 'loc_seq', 'user_seq', etc., 'target'
        
        Returns:
            Mixed batch and lambda value
        """
        if self.alpha <= 0:
            return batch, 1.0
        
        batch_size = batch['target'].shape[0]
        lam = torch.from_numpy(np.random.beta(self.alpha, self.alpha, size=batch_size)).float()
        lam = lam.to(batch['target'].device)
        
        # Get random permutation
        indices = torch.randperm(batch_size)
        
        # Mix sequences (for embedding inputs, we keep discrete)
        # But we can mix the targets
        mixed_batch = {}
        for key in batch:
            if key == 'target':
                # Don't mix targets, use for loss computation
                mixed_batch[key] = batch[key]
                mixed_batch['target_b'] = batch[key][indices]
                mixed_batch['lam'] = lam
            else:
                # Keep sequences discrete (no mixing for embeddings)
                mixed_batch[key] = batch[key]
        
        return mixed_batch
    
    def loss(self, criterion, logits, targets, targets_b, lam):
        """
        Compute mixup loss.
        """
        lam = lam.view(-1, 1)
        return (lam * criterion(logits, targets) + (1 - lam) * criterion(logits, targets_b)).mean()
