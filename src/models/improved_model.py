"""
Improved model with better generalization:
- Simpler architecture to reduce overfitting
- Stronger regularization (dropout, weight decay)
- Lightweight attention mechanism
- Focus on location sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Efficient attention without separate projections."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: [batch, seq_len, d_model]
            mask: [batch, seq_len]
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, L]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        
        return out


class SelfAttentionBlock(nn.Module):
    """Self-attention with residual connection and layer norm."""
    
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class ImprovedNextLocModel(nn.Module):
    """
    Simplified next-location prediction model with better generalization.
    
    Key features:
    - Location + User embeddings as core features
    - Simple temporal encoding
    - 2-layer self-attention
    - Strong regularization (dropout 0.3)
    - Under 500K parameters
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=112,
        num_layers=2,
        dropout=0.3,
        max_seq_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Primary embeddings - focus on location and user
        self.loc_embedding = nn.Embedding(num_locations, d_model // 2)
        self.user_embedding = nn.Embedding(num_users, d_model // 4)
        
        # Simple temporal embeddings
        self.weekday_embedding = nn.Embedding(7, d_model // 8)
        
        # Temporal features projection (time and duration)
        self.temporal_proj = nn.Linear(2, d_model // 8)
        
        # Input projection
        input_dim = d_model // 2 + d_model // 4 + d_model // 8 + d_model // 8
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Self-attention layers
        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        # Prediction head - simplified
        self.classifier = nn.Linear(d_model, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for better generalization."""
        for name, p in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len):
        """
        Args:
            loc_seq: [batch_size, seq_len]
            user_seq: [batch_size, seq_len]
            weekday_seq: [batch_size, seq_len]
            start_min_seq: [batch_size, seq_len]
            dur_seq: [batch_size, seq_len]
            diff_seq: [batch_size, seq_len]  (not used to simplify)
            seq_len: [batch_size]
        
        Returns:
            logits: [batch_size, num_locations]
        """
        batch_size, max_len = loc_seq.shape
        device = loc_seq.device
        
        # Create padding mask
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        # Temporal features
        temporal_features = torch.stack([start_min_seq, dur_seq], dim=-1)
        temporal_emb = self.temporal_proj(temporal_features)
        
        # Concatenate features
        x = torch.cat([loc_emb, user_emb, weekday_emb, temporal_emb], dim=-1)
        
        # Project to d_model
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Add positional encoding
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        
        # Apply self-attention layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.output_norm(x)
        x = self.output_dropout(x)
        
        # Extract last valid position
        idx = (seq_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
        x = x.gather(1, idx).squeeze(1)
        
        # Classify
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
