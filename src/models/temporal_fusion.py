"""
Modern architecture for next-location prediction.

Combines:
1. Multi-scale temporal convolutions (inspired by TCN/WaveNet)
2. Multi-head self-attention (Transformer-style)
3. Gated residual connections
4. Feature fusion mechanisms

Avoids all RNN-style architectures (no LSTM, GRU, or recurrence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class CausalConv1d(nn.Module):
    """Causal 1D convolution for temporal modeling."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, seq_len]
        """
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    """Temporal convolutional block with gating and residual connections."""
    
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism
        self.gate = CausalConv1d(channels, channels, kernel_size, dilation)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, channels]
        """
        residual = x
        
        # Transpose for conv1d
        x = x.transpose(1, 2)  # [B, C, L]
        
        # First conv + gate
        out = self.conv1(x)
        gate = torch.sigmoid(self.gate(x))
        out = out * gate
        
        out = out.transpose(1, 2)  # [B, L, C]
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Second conv
        out = out.transpose(1, 2)
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        
        return F.gelu(residual + out)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - padding mask
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class FusionLayer(nn.Module):
    """Combine temporal convolution and attention outputs."""
    
    def __init__(self, d_model, num_heads, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.temporal_block = TemporalBlock(d_model, kernel_size, dilation, dropout)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable fusion weights
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len]
        """
        # Temporal convolution path
        temp_out = self.temporal_block(x)
        
        # Attention path
        attn_out = self.attention(x, mask)
        
        # Fusion with gating
        combined = torch.cat([temp_out, attn_out], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        
        fused = gate * temp_out + (1 - gate) * attn_out
        fused = self.norm(x + self.dropout(fused))
        
        return fused


class TemporalFusionModel(nn.Module):
    """
    Temporal Fusion model for next-location prediction.
    
    Architecture:
    - Multi-feature embedding (location, user, temporal features)
    - Stacked fusion layers (temporal conv + attention)
    - Final prediction head
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        num_layers=3,
        num_heads=4,
        kernel_size=3,
        dropout=0.2,
        max_seq_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.loc_embedding = nn.Embedding(num_locations, d_model // 2)
        self.user_embedding = nn.Embedding(num_users, d_model // 4)
        self.weekday_embedding = nn.Embedding(7, d_model // 8)
        self.diff_embedding = nn.Embedding(100, d_model // 8)  # Assuming diff < 100
        
        # Temporal feature projection
        self.temporal_proj = nn.Linear(2, d_model // 8)  # start_min, duration
        
        # Input projection to d_model
        input_dim = d_model // 2 + d_model // 4 + d_model // 8 + d_model // 8 + d_model // 8
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Fusion layers with increasing dilation
        self.layers = nn.ModuleList([
            FusionLayer(
                d_model,
                num_heads,
                kernel_size,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Prediction head
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_locations)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len):
        """
        Args:
            loc_seq: [batch_size, seq_len]
            user_seq: [batch_size, seq_len]
            weekday_seq: [batch_size, seq_len]
            start_min_seq: [batch_size, seq_len]
            dur_seq: [batch_size, seq_len]
            diff_seq: [batch_size, seq_len]
            seq_len: [batch_size]
        
        Returns:
            logits: [batch_size, num_locations]
        """
        batch_size, max_len = loc_seq.shape
        
        # Create padding mask
        mask = torch.arange(max_len, device=loc_seq.device).unsqueeze(0) < seq_len.unsqueeze(1)
        
        # Clip diff_seq to valid range
        diff_seq = torch.clamp(diff_seq, 0, 99)
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        diff_emb = self.diff_embedding(diff_seq)
        
        # Temporal features
        temporal_features = torch.stack([start_min_seq, dur_seq], dim=-1)
        temporal_emb = self.temporal_proj(temporal_features)
        
        # Concatenate all features
        x = torch.cat([loc_emb, user_emb, weekday_emb, diff_emb, temporal_emb], dim=-1)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply fusion layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Extract last valid position for each sequence
        idx = (seq_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
        x = x.gather(1, idx).squeeze(1)
        
        # Prediction
        logits = self.output_proj(x)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
