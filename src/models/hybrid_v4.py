"""
Final optimized model v4 - Hybrid approach combining:
1. Location frequency priors (learned)
2. Recent context attention
3. User preferences
4. Temporal patterns

Target: Stable 40% test Acc@1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridLocationPredictor(nn.Module):
    """
    Hybrid model combining multiple prediction signals.
    
    Components:
    - Global location frequency bias
    - User-specific location preferences
    - Recent visit attention
    - Temporal context
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        dropout=0.3,
        max_seq_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Location embeddings - larger for importance
        self.loc_embedding = nn.Embedding(num_locations, d_model // 2, padding_idx=0)
        
        # User embeddings
        self.user_embedding = nn.Embedding(num_users, d_model // 4)
        
        # Temporal embeddings
        self.weekday_embedding = nn.Embedding(7, d_model // 8)
        self.hour_embedding = nn.Embedding(24, d_model // 8)
        
        # Input projection
        input_dim = d_model // 2 + d_model // 4 + d_model // 8 + d_model // 8
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)
        
        # Simple attention (focus on recent context)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.attn_norm = nn.LayerNorm(d_model)
        
        # User-location interaction
        self.user_loc_interaction = nn.Bilinear(d_model // 4, d_model // 2, d_model // 2)
        
        # Global frequency bias (learnable)
        self.global_bias = nn.Parameter(torch.zeros(num_locations))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output = nn.Linear(d_model, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, p in self.named_parameters():
            if 'embedding' in name and 'pos' not in name:
                nn.init.normal_(p, std=0.02)
            elif p.dim() > 1 and 'pos' not in name and 'bias' not in name:
                nn.init.xavier_uniform_(p, gain=0.5)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len):
        """
        Args:
            loc_seq: [B, L]
            user_seq: [B, L]
            weekday_seq: [B, L]
            start_min_seq: [B, L]
            Others: not used
            seq_len: [B]
        
        Returns:
            logits: [B, num_locations]
        """
        B, L = loc_seq.shape
        device = loc_seq.device
        
        # Padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) >= seq_len.unsqueeze(1)
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)  # [B, L, d//2]
        user_emb = self.user_embedding(user_seq)  # [B, L, d//4]
        weekday_emb = self.weekday_embedding(weekday_seq)  # [B, L, d//8]
        
        # Hour embedding from start_min
        # Denormalize and extract hour
        hour = ((start_min_seq * 120) + 720) / 60  # Rough estimate
        hour = hour.clamp(0, 23).long()
        hour_emb = self.hour_embedding(hour)  # [B, L, d//8]
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb], dim=-1)
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :L, :]
        
        # Attention over sequence
        x_attn, _ = self.context_attention(x, x, x, key_padding_mask=mask)
        x = self.attn_norm(x + x_attn)
        
        # Extract last valid position
        idx = (seq_len - 1).long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.d_model)
        context = x.gather(1, idx).squeeze(1)  # [B, d_model]
        
        # Get last location and user for interaction
        last_loc_idx = (seq_len - 1).long()
        batch_idx = torch.arange(B, device=device)
        last_loc_emb = loc_emb[batch_idx, last_loc_idx]  # [B, d//2]
        last_user_emb = user_emb[batch_idx, last_loc_idx]  # [B, d//4]
        
        # User-location interaction
        interaction = self.user_loc_interaction(last_user_emb, last_loc_emb)  # [B, d//2]
        
        # Fuse context and interaction
        combined = torch.cat([context, interaction], dim=-1)
        fused = self.fusion(combined)
        
        # Predict
        logits = self.output(fused)
        
        # Add global bias
        logits = logits + self.global_bias
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
