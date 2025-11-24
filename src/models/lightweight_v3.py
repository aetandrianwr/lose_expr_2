"""
Highly optimized model v3 with focus on generalization.

Key strategies:
1. Very simple architecture - minimal parameters
2. Strong dropout and regularization
3. Direct location prediction focus
4. Ensemble-ready design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LightweightAttentionModel(nn.Module):
    """
    Ultra-lightweight model focused on generalization.
    
    Design philosophy:
    - Simplicity over complexity
    - Strong regularization
    - Focus on core features (location, user, time)
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=96,
        dropout=0.4,
        max_seq_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Core embeddings - location is most important
        self.loc_embedding = nn.Embedding(num_locations, d_model // 2, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 4)
        
        # Temporal features - simplified
        self.weekday_embedding = nn.Embedding(7, d_model // 8)
        self.hour_proj = nn.Linear(1, d_model // 8)
        
        # Input projection
        input_dim = d_model // 2 + d_model // 4 + d_model // 8 + d_model // 8
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Simple positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Single attention layer - keep it simple
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization."""
        for name, p in self.named_parameters():
            if 'embedding' in name and 'pos' not in name:
                nn.init.normal_(p, mean=0.0, std=0.01)
            elif p.dim() > 1 and 'pos' not in name:
                nn.init.xavier_normal_(p, gain=0.3)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len):
        """
        Forward pass.
        
        Args:
            loc_seq: [B, L]
            user_seq: [B, L]
            weekday_seq: [B, L]
            start_min_seq: [B, L] - normalized
            dur_seq: [B, L] - normalized (not used)
            diff_seq: [B, L] - not used
            seq_len: [B]
        
        Returns:
            logits: [B, num_locations]
        """
        B, L = loc_seq.shape
        device = loc_seq.device
        
        # Padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        # Hour from start_min (extract hour info)
        hour = (start_min_seq * self.train_std + self.train_mean) / 60.0
        hour = hour.unsqueeze(-1) / 24.0  # Normalize to [0, 1]
        hour_emb = self.hour_proj(hour)
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb], dim=-1)
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :L, :]
        
        # Self-attention
        residual = x  # [B, L, d_model]
        x = self.attn_norm(x)
        
        qkv = self.attn_qkv(x).reshape(B, L, 3, self.d_model).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, L, d_model]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)  # [B, L, L]
        
        # Apply mask
        mask_expanded = mask.unsqueeze(1)  # [B, 1, L]
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)  # [B, L, L]
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, L, d_model]
        out = self.attn_out(out)
        x = residual + out  # [B, L, d_model]
        
        # Feed-forward
        x = x + self.ff(self.ff_norm(x))  # [B, L, d_model]
        
        # Extract last valid position for each sequence
        idx = (seq_len - 1).long()  # [B]
        idx = idx.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        idx = idx.expand(-1, -1, self.d_model)  # [B, 1, d_model]
        x = x.gather(dim=1, index=idx).squeeze(1)  # [B, d_model]
        
        x = self.output_norm(x)
        x = self.output_dropout(x)
        
        logits = self.classifier(x)
        
        return logits
    
    def set_stats(self, mean, std):
        """Set normalization stats for temporal features."""
        self.register_buffer('train_mean', torch.tensor(mean))
        self.register_buffer('train_std', torch.tensor(std))
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
