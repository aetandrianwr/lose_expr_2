"""
Advanced Next-Location Prediction Model

Key improvements:
1. Hierarchical location embeddings with clustering
2. Enhanced temporal encoding (hour, day, season)
3. User-location interaction modeling
4. Multi-head cross-attention between user and sequence
5. Frequency-aware prediction head
6. Better positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptivePositionalEncoding(nn.Module):
    """Learnable positional encoding that adapts to sequence length"""
    def __init__(self, d_model, max_len=150, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Sinusoidal positional encoding as initialization
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Initialize with sinusoidal
        self.pos_embedding.weight.data.copy_(pe)

    def forward(self, x, seq_lens):
        """
        Args:
            x: (batch, seq_len, d_model)
            seq_lens: (batch,) actual sequence lengths
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        return self.dropout(x + pos_emb)


class HierarchicalLocationEmbedding(nn.Module):
    """
    Hierarchical location embeddings (parameter-efficient):
    - Location embedding (primary)
    - Cluster embedding (additive, smaller)
    - Frequency embedding (additive, smaller)
    """
    def __init__(self, num_locations, d_model, num_clusters=50):
        super().__init__()
        self.num_locations = num_locations
        self.num_clusters = num_clusters
        self.d_model = d_model

        # Location to cluster mapping (will be set externally)
        self.register_buffer('loc_to_cluster', torch.zeros(num_locations, dtype=torch.long))

        # Main location embedding
        self.location_embed = nn.Embedding(num_locations, d_model)

        # Smaller additive embeddings for hierarchy
        self.cluster_embed = nn.Embedding(num_clusters, d_model // 4)
        self.cluster_proj = nn.Linear(d_model // 4, d_model, bias=False)

        # Frequency embedding (smaller)
        self.freq_embed = nn.Embedding(10, d_model // 4)  # Reduced from 20 to 10 buckets
        self.freq_proj = nn.Linear(d_model // 4, d_model, bias=False)
        self.register_buffer('loc_freq_bucket', torch.zeros(num_locations, dtype=torch.long))

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, loc_ids):
        """
        Args:
            loc_ids: (batch, seq_len) location IDs
        Returns:
            (batch, seq_len, d_model)
        """
        # Main location embedding
        loc_emb = self.location_embed(loc_ids)

        # Add cluster information (projected to d_model)
        cluster_ids = self.loc_to_cluster[loc_ids]
        cluster_emb = self.cluster_embed(cluster_ids)
        cluster_emb = self.cluster_proj(cluster_emb)

        # Add frequency information (projected to d_model)
        freq_bucket = self.loc_freq_bucket[loc_ids]
        freq_emb = self.freq_embed(freq_bucket)
        freq_emb = self.freq_proj(freq_emb)

        # Combine with residual connections
        combined = loc_emb + 0.3 * cluster_emb + 0.2 * freq_emb
        return self.layer_norm(combined)


class EnhancedTemporalEncoding(nn.Module):
    """
    Enhanced temporal features:
    - Hour of day (cyclical)
    - Day of week (cyclical)
    - Time since previous visit
    - Duration
    """
    def __init__(self, d_model):
        super().__init__()

        # Cyclical encodings
        self.hour_embed = nn.Linear(2, d_model // 4)  # sin/cos for hour
        self.weekday_embed = nn.Embedding(7, d_model // 4)

        # Duration and time diff encodings
        self.duration_proj = nn.Linear(1, d_model // 4)
        self.diff_embed = nn.Embedding(50, d_model // 4)  # Bucketed time differences

        self.combiner = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, start_min, weekday, duration, diff):
        """
        Args:
            start_min: (batch, seq_len) start minute of day (0-1439)
            weekday: (batch, seq_len) day of week (0-6)
            duration: (batch, seq_len) duration in minutes
            diff: (batch, seq_len) time difference bucket
        Returns:
            (batch, seq_len, d_model)
        """
        # Cyclical hour encoding
        hour = start_min / 60.0  # Convert to hours
        hour_rad = hour * (2 * math.pi / 24.0)
        hour_sin = torch.sin(hour_rad).unsqueeze(-1)
        hour_cos = torch.cos(hour_rad).unsqueeze(-1)
        hour_emb = self.hour_embed(torch.cat([hour_sin, hour_cos], dim=-1))

        # Weekday embedding
        weekday_emb = self.weekday_embed(weekday)

        # Duration embedding
        duration_emb = self.duration_proj(duration.unsqueeze(-1))

        # Time difference embedding
        diff_emb = self.diff_embed(diff)

        # Combine all temporal features
        temporal_emb = torch.cat([hour_emb, weekday_emb, duration_emb, diff_emb], dim=-1)
        temporal_emb = self.combiner(temporal_emb)
        return self.layer_norm(temporal_emb)


class UserLocationInteraction(nn.Module):
    """
    Models interaction between user preferences and location sequences
    Uses cross-attention mechanism
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_repr, user_repr, seq_mask=None):
        """
        Args:
            seq_repr: (batch, seq_len, d_model) sequence representations
            user_repr: (batch, 1, d_model) user representation
            seq_mask: (batch, seq_len) attention mask
        Returns:
            (batch, seq_len, d_model) enhanced sequence representations
        """
        # Cross attention: sequence attends to user
        attn_out, _ = self.cross_attn(
            query=seq_repr,
            key=user_repr,
            value=user_repr,
            key_padding_mask=None
        )
        seq_repr = self.norm1(seq_repr + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(seq_repr)
        seq_repr = self.norm2(seq_repr + self.dropout(ffn_out))

        return seq_repr


class FrequencyAwareHead(nn.Module):
    """
    Parameter-efficient prediction head with frequency awareness
    """
    def __init__(self, d_model, num_locations):
        super().__init__()

        # Single-layer neural prediction with frequency bias
        self.neural_proj = nn.Linear(d_model, num_locations)

        # Learnable frequency scaling
        self.freq_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden, freq_probs):
        """
        Args:
            hidden: (batch, d_model) final hidden state
            freq_probs: (batch, num_locations) frequency-based priors
        Returns:
            (batch, num_locations) logits
        """
        # Neural predictions
        neural_logits = self.neural_proj(hidden)

        # Add frequency bias (scaled log probs)
        freq_bias = torch.log(freq_probs + 1e-8) * self.freq_scale

        # Combine
        combined_logits = neural_logits + freq_bias

        return combined_logits


class AdvancedNextLocationModel(nn.Module):
    """
    Advanced model for next-location prediction with multiple enhancements
    """
    def __init__(
        self,
        num_locations,
        num_users,
        location_freq,
        d_model=128,
        num_heads=8,
        num_layers=3,
        dropout=0.2,
        max_seq_len=150,
        num_clusters=50
    ):
        super().__init__()

        self.d_model = d_model
        self.num_locations = num_locations

        # Hierarchical location embedding
        self.loc_embed = HierarchicalLocationEmbedding(num_locations, d_model, num_clusters)

        # User embedding
        self.user_embed = nn.Embedding(num_users, d_model)

        # Enhanced temporal encoding
        self.temporal_encode = EnhancedTemporalEncoding(d_model)

        # Positional encoding
        self.pos_encode = AdaptivePositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # User-location interaction
        self.user_loc_interaction = UserLocationInteraction(d_model, num_heads, dropout)

        # Frequency-aware prediction head
        self.prediction_head = FrequencyAwareHead(d_model, num_locations)

        # Store frequency information
        self.register_buffer('location_freq', torch.FloatTensor(location_freq))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len):
        """
        Args:
            loc_seq: (batch, max_seq_len) location IDs
            user_seq: (batch, max_seq_len) user IDs (same user repeated)
            weekday_seq: (batch, max_seq_len) weekday
            start_min_seq: (batch, max_seq_len) start minute
            dur_seq: (batch, max_seq_len) duration
            diff_seq: (batch, max_seq_len) time difference bucket
            seq_len: (batch,) actual sequence lengths
        Returns:
            (batch, num_locations) prediction logits
        """
        batch_size, max_len = loc_seq.size()

        # Create attention mask for padding
        mask = torch.arange(max_len, device=loc_seq.device).unsqueeze(0) >= seq_len.unsqueeze(1)

        # Location embeddings (hierarchical)
        loc_emb = self.loc_embed(loc_seq)

        # Temporal encodings
        temporal_emb = self.temporal_encode(start_min_seq, weekday_seq, dur_seq, diff_seq)

        # Combine location and temporal information
        seq_repr = loc_emb + temporal_emb

        # Add positional encoding
        seq_repr = self.pos_encode(seq_repr, seq_len)

        # Transform through encoder
        seq_repr = self.transformer(seq_repr, src_key_padding_mask=mask)

        # Get user representation (use first timestep user ID)
        user_repr = self.user_embed(user_seq[:, 0]).unsqueeze(1)

        # User-location interaction
        seq_repr = self.user_loc_interaction(seq_repr, user_repr, mask)

        # Get final representation (last non-padded position)
        # Gather the last valid position for each sequence
        indices = (seq_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
        final_repr = torch.gather(seq_repr, 1, indices).squeeze(1)

        # Get frequency-based priors
        freq_probs = self.location_freq.unsqueeze(0).expand(batch_size, -1)

        # Final prediction with frequency awareness
        logits = self.prediction_head(final_repr, freq_probs)

        return logits

    def get_num_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
