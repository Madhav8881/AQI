"""
Phase 4: Deep Learning Model Architectures in Native PyTorch.
Implements:
1. Persistence Baseline
2. Temporal LSTM Baseline
3. Temporal Transformer Baseline
4. Spatio-Temporal GAT-Transformer (Proposed Model)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PersistenceModel:
    """Naive baseline: Outputs the latest observed PM2.5 at t=23 forward to all horizons."""
    def __init__(self, horizons=[1, 3, 6, 12]):
        self.horizons = horizons
        
    def predict(self, X):
        # X shape: [Batch, Time=24, Nodes, Features]
        # Feature 0 is normalized PM2.5 at time t=23
        latest_pm25 = X[:, -1, :, 0] # [Batch, Nodes]
        # Repeat across the 4 horizons -> [Batch, Horizons=4, Nodes]
        n_horizons = len(self.horizons)
        preds = latest_pm25.unsqueeze(1).repeat(1, n_horizons, 1)
        return preds

class TemporalLSTM(nn.Module):
    """Temporal Baseline: 2-layer LSTM modeling station time series independently."""
    def __init__(self, in_features=5, hidden_dim=64, num_layers=2, num_horizons=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_horizons = num_horizons
        
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_horizons)
        )
        
    def forward(self, x):
        # x: [Batch, Time=24, Nodes=12, Features=5]
        B, T, N, F_dim = x.shape
        # Flatten nodes into batch dimension: [B * N, T, F_dim]
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, F_dim)
        
        lstm_out, (hn, cn) = self.lstm(x_flat) # hn: [num_layers, B*N, hidden_dim]
        last_hidden = hn[-1] # [B * N, hidden_dim]
        
        out = self.head(last_hidden) # [B * N, num_horizons]
        # Reshape to [Batch, Nodes, Horizons] -> permute to [Batch, Horizons, Nodes]
        out = out.view(B, N, self.num_horizons).permute(0, 2, 1)
        return out

class TemporalTransformer(nn.Module):
    """Temporal Baseline: Multi-head self-attention encoder over 24 hours."""
    def __init__(self, in_features=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_horizons=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_horizons = num_horizons
        self.proj = nn.Linear(in_features, d_model)
        
        # Positional Encoding for 24 time steps
        self.pos_emb = nn.Parameter(torch.randn(1, 24, 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_horizons)
        )
        
    def forward(self, x):
        # x: [Batch, Time=24, Nodes=12, Features=5]
        B, T, N, F_dim = x.shape
        # Project features: [B, T, N, d_model]
        h = self.proj(x) + self.pos_emb
        
        # Reshape to treat each node's sequence independently: [B * N, T, d_model]
        h_flat = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        trans_out = self.transformer_encoder(h_flat) # [B * N, T, d_model]
        
        # Take the representation of the final time step t=23
        last_step = trans_out[:, -1, :] # [B * N, d_model]
        
        out = self.head(last_step) # [B * N, num_horizons]
        out = out.view(B, N, self.num_horizons).permute(0, 2, 1) # [B, num_horizons, N]
        return out

class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention Network (GAT) layer implemented in native PyTorch.
    Learns spatial aggregation weights between connected monitoring stations.
    """
    def __init__(self, in_features, out_features, num_heads=2, dropout=0.1, alpha=0.2):
        super().__init__()
        if out_features % num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads")

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim, 1) * 0.02)
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim, 1) * 0.02)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        
    def forward(self, h, adj_mask):
        # h: [Batch, Nodes=12, in_features]
        # adj_mask: [Nodes, Nodes] (1.0 where connected, 0.0 otherwise)
        B, N, _ = h.shape
        
        # 1. Linear transformation: [B, N, num_heads, head_dim] -> permute to [B, num_heads, N, head_dim]
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 2. Self-attention scores: e_ij = a_src^T Wh_i + a_dst^T Wh_j
        # score_src: [B, heads, N, 1]
        score_src = torch.matmul(Wh, self.a_src)
        # score_dst: [B, heads, 1, N]
        score_dst = torch.matmul(Wh, self.a_dst).transpose(-1, -2)
        
        # Pairwise raw scores: [B, heads, N, N]
        attn_scores = self.leaky_relu(score_src + score_dst)
        
        # Mask out non-edges with large negative value (-1e9)
        # adj_mask: [N, N] -> [1, 1, N, N]
        mask = (adj_mask > 0).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        
        # Softmax over neighbors (dim=-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 3. Aggregate neighbor features: [B, heads, N, head_dim]
        h_out = torch.matmul(attn_weights, Wh) # [B, heads, N, head_dim]
        
        # Concatenate heads: [B, N, out_features]
        h_out = h_out.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_features)
        
        # Residual connection + LayerNorm (if dimensions match)
        if self.in_features == self.out_features:
            h_out = self.norm(h + h_out)
        else:
            h_out = self.norm(h_out)
            
        return h_out, attn_weights

class SpatiotemporalGATTransformer(nn.Module):
    """
    Proposed Integrated Architecture:
    1. Spatial GAT: Aggregates spatial pollution context across stations at each timestep.
    2. Temporal Transformer: Models multi-hour evolution across the 24h history.
    3. Multi-Horizon Forecasting Head: Direct projection to [+1h, +3h, +6h, +12h].
    """
    def __init__(self, in_features=5, d_model=64, num_gat_heads=2, num_trans_heads=4,
                 num_trans_layers=2, dim_feedforward=128, num_horizons=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_horizons = num_horizons
        
        # Input Feature Projection: 5 -> d_model
        self.in_proj = nn.Linear(in_features, d_model)
        
        # Spatial Graph Attention
        self.gat = GraphAttentionLayer(
            in_features=d_model,
            out_features=d_model,
            num_heads=num_gat_heads,
            dropout=dropout
        )
        
        # Temporal Positional Encoding (for 24 time steps)
        self.time_pos_emb = nn.Parameter(torch.randn(1, 24, 1, d_model) * 0.02)
        
        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_trans_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        
        # Multi-Horizon Forecasting Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_horizons)
        )
        
    def forward(self, x, adj_matrix):
        # x: [Batch, Time=24, Nodes=12, Features=5]
        # adj_matrix: [Nodes=12, Nodes=12]
        B, T, N, F_dim = x.shape
        
        # 1. Project input features: [B, T, N, d_model]
        h = self.in_proj(x)
        
        # 2. Spatial GAT aggregation at each time step t in [0..23]
        # Reshape to batch across time: [B * T, N, d_model]
        h_spatial = h.view(B * T, N, self.d_model)
        h_spatial_out, last_attn_weights = self.gat(h_spatial, adj_matrix)
        h_spatial_out = h_spatial_out.view(B, T, N, self.d_model)
        
        # 3. Add Temporal Positional Encoding
        h_temporal_in = h_spatial_out + self.time_pos_emb # [B, T, N, d_model]
        
        # 4. Temporal Transformer: Pass each station's 24-step sequence
        # Reshape to [B * N, T, d_model]
        h_trans_flat = h_temporal_in.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        trans_out = self.transformer_encoder(h_trans_flat) # [B * N, T, d_model]
        
        # Representation at final timestep t=23: [B * N, d_model]
        final_rep = trans_out[:, -1, :]
        
        # 5. Multi-Horizon Forecasting Head
        out = self.head(final_rep) # [B * N, num_horizons]
        out = out.view(B, N, self.num_horizons).permute(0, 2, 1) # [B, num_horizons, N]
        
        return out

if __name__ == "__main__":
    # Self-test model shapes and forward pass
    B, T, N, F_dim = 16, 24, 12, 5
    x = torch.randn(B, T, N, F_dim)
    adj = torch.ones(N, N)
    
    print("Testing Model Forward Passes...")
    
    # 1. Persistence
    pers = PersistenceModel()
    y_pers = pers.predict(x)
    print(f"Persistence output shape: {y_pers.shape} (Expected: [16, 4, 12])")
    
    # 2. LSTM
    lstm = TemporalLSTM()
    y_lstm = lstm(x)
    print(f"LSTM output shape:        {y_lstm.shape} (Expected: [16, 4, 12])")
    
    # 3. Transformer
    trans = TemporalTransformer()
    y_trans = trans(x)
    print(f"Transformer output shape: {y_trans.shape} (Expected: [16, 4, 12])")
    
    # 4. GAT-Transformer
    gat_trans = SpatiotemporalGATTransformer()
    y_gat = gat_trans(x, adj)
    print(f"GAT-Transformer output:   {y_gat.shape} (Expected: [16, 4, 12])")
    
    total_params = sum(p.numel() for p in gat_trans.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters in GAT-Transformer: {total_params:,}")
    print("All forward passes verified successfully!")
