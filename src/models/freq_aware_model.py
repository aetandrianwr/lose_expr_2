"""
Optimized Model with Explicit Frequency Modeling

Strategy:
1. Use proven Temporal Fusion architecture (Model 1 base)
2. Add explicit frequency-based predictions
3. Ensemble implicit (learned) + explicit (frequency) predictions
4. Simpler, more stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FrequencyAwareTemporalModel(nn.Module):
    """
    Temporal model with explicit frequency modeling.
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        location_freq=None,
        user_loc_freq=None,
        d_model=128,
        num_heads=4,
        dropout=0.2,
        max_seq_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # === EMBEDDINGS ===
        self.loc_embedding = nn.Embedding(num_locations, d_model // 2, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 4)
        self.weekday_embedding = nn.Embedding(7, d_model // 8)
        self.hour_embedding = nn.Embedding(24, d_model // 8)
        
        # Input fusion
        input_dim = d_model // 2 + d_model // 4 + d_model // 8 + d_model // 8
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # === TEMPORAL ENCODING ===
        # Multi-scale temporal convolutions
        self.temp_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temp_conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.temp_conv3 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        
        # === ATTENTION ===
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # === FREQUENCY MODELING ===
        # Global frequency bias
        if location_freq is not None:
            freq_normalized = torch.tensor(location_freq, dtype=torch.float32)
            freq_normalized = freq_normalized / (freq_normalized.sum() + 1e-8)
            self.register_buffer('global_freq', freq_normalized)
        else:
            self.register_buffer('global_freq', torch.ones(num_locations) / num_locations)
        
        # Learnable frequency adjustment
        self.freq_proj = nn.Linear(d_model, num_locations)
        
        # === OUTPUT ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Gating for frequency vs learned
        self.freq_gate = nn.Linear(d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(p, std=0.02)
            elif p.dim() > 1 and 'bias' not in name and 'freq' not in name:
                nn.init.xavier_uniform_(p, gain=0.5)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len):
        B, L = loc_seq.shape
        device = loc_seq.device
        
        # Padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) >= seq_len.unsqueeze(1)
        
        # === EMBED ===
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        hour = ((start_min_seq * 120) + 720) / 60
        hour = hour.clamp(0, 23).long()
        hour_emb = self.hour_embedding(hour)
        
        # Fuse
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb], dim=-1)
        x = self.input_proj(x)  # [B, L, d_model]
        
        # === TEMPORAL MODELING ===
        # Multi-scale convolutions
        x_t = x.transpose(1, 2)  # [B, d_model, L]
        conv1 = F.gelu(self.temp_conv1(x_t))
        conv2 = F.gelu(self.temp_conv2(x_t))
        conv3 = F.gelu(self.temp_conv3(x_t))
        x_conv = (conv1 + conv2 + conv3) / 3.0
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_model]
        
        x = x + x_conv
        
        # === ATTENTION ===
        x_attn, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.attn_norm(x + x_attn)
        
        # === EXTRACT LAST VALID ===
        idx = (seq_len - 1).long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.d_model)
        context = x.gather(1, idx).squeeze(1)  # [B, d_model]
        
        context = self.output_norm(context)
        context = self.output_dropout(context)
        
        # === DUAL PREDICTION ===
        # 1. Learned prediction
        learned_logits = self.classifier(context)
        
        # 2. Frequency-adjusted prediction
        freq_adjustment = self.freq_proj(context)  # [B, num_locs]
        freq_logits = torch.log(self.global_freq + 1e-8).unsqueeze(0) + freq_adjustment
        
        # 3. Gated combination
        gate = torch.sigmoid(self.freq_gate(context))  # [B, 1]
        final_logits = gate * learned_logits + (1 - gate) * freq_logits
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
