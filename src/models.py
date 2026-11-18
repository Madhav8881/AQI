"""
Forecasting models (native PyTorch).

All neural models share the same interface and tricks so the comparison is fair:
- input  x: [B, T, N, F] (F[0] = standardised PM2.5, F[1] = observed flag, rest calendar)
- output   : [B, H, N] standardised PM2.5 at each horizon
- a learnable station embedding (stations have very different pollution levels)
- residual forecasting: the network predicts the change from the last input value

Models
1. Persistence            - last value repeated (no training)
2. SeasonalNaive          - value at the same hour on the previous day (no training)
3. TemporalLSTM           - each station independently
4. TemporalTransformer    - each station independently
5. GATTransformer         - proposed: spatial GAT at each hour -> temporal Transformer
                            -> spatial GAT on the temporal summary (distance-biased attention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Persistence:
    def __init__(self, horizons):
        self.horizons = horizons

    def __call__(self, x):
        return x[:, -1, :, 0].unsqueeze(1).repeat(1, len(self.horizons), 1)


class SeasonalNaive:
    """Forecast for t+h is the observation at t+h-24 (same hour, previous day)."""
    def __init__(self, horizons, window=24):
        self.idx = [window - 24 + h - 1 for h in horizons]

    def __call__(self, x):
        return x[:, self.idx, :, 0]


class _Base(nn.Module):
    def __init__(self, n_nodes, in_features, d_model, n_horizons, node_emb):
        super().__init__()
        self.n_horizons = n_horizons
        self.node_emb = nn.Embedding(n_nodes, node_emb)
        self.in_proj = nn.Linear(in_features + node_emb, d_model)

    def embed(self, x):
        B, T, N, _ = x.shape
        e = self.node_emb.weight.view(1, 1, N, -1).expand(B, T, N, -1)
        return self.in_proj(torch.cat([x, e], dim=-1))  # [B, T, N, d]

    @staticmethod
    def make_head(d, h, dropout):
        return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, h))

    def residual(self, x, delta):  # delta [B, N, H] -> [B, H, N]
        return x[:, -1, :, 0].unsqueeze(1) + delta.permute(0, 2, 1)


class TemporalLSTM(_Base):
    def __init__(self, n_nodes, in_features=8, d_model=64, num_layers=2, n_horizons=5,
                 node_emb=8, dropout=0.1):
        super().__init__(n_nodes, in_features, d_model, n_horizons, node_emb)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True, dropout=dropout)
        self.head = self.make_head(d_model, n_horizons, dropout)

    def forward(self, x, adj=None):
        B, T, N, _ = x.shape
        h = self.embed(x).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, (hn, _) = self.lstm(h)
        return self.residual(x, self.head(hn[-1]).view(B, N, -1))


class TemporalTransformer(_Base):
    def __init__(self, n_nodes, in_features=8, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, n_horizons=5, node_emb=8, dropout=0.1, window=24):
        super().__init__(n_nodes, in_features, d_model, n_horizons, node_emb)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = self.make_head(d_model, n_horizons, dropout)

    def forward(self, x, adj=None):
        B, T, N, _ = x.shape
        h = self.embed(x).permute(0, 2, 1, 3).reshape(B * N, T, -1) + self.pos
        h = self.encoder(h)[:, -1]
        return self.residual(x, self.head(h).view(B, N, -1))


class GraphAttention(nn.Module):
    """Multi-head GAT layer with a distance prior: score_ij += log(w_ij) on graph edges."""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dh = num_heads, d_model // num_heads
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.dh) * 0.1)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        # x: [B, N, d]; adj: [N, N] edge weights (0 = no edge)
        B, N, _ = x.shape
        Wh = self.W(x).view(B, N, self.h, self.dh)                      # [B, N, h, dh]
        s_src = (Wh * self.a_src).sum(-1).permute(0, 2, 1).unsqueeze(-1)  # [B, h, N, 1]
        s_dst = (Wh * self.a_dst).sum(-1).permute(0, 2, 1).unsqueeze(-2)  # [B, h, 1, N]
        bias = torch.log(adj.clamp_min(1e-6)).masked_fill(adj <= 0, float("-inf"))
        attn = F.softmax(F.leaky_relu(s_src + s_dst, 0.2) + bias, dim=-1)  # [B, h, N, N]
        msg = torch.matmul(self.drop(attn), Wh.permute(0, 2, 1, 3))      # [B, h, N, dh]
        msg = msg.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.norm(x + self.drop(self.out(F.elu(msg)))), attn


class GATTransformer(_Base):
    def __init__(self, n_nodes, in_features=8, d_model=64, gat_heads=4, nhead=4, num_layers=2,
                 dim_ff=128, n_horizons=5, node_emb=8, dropout=0.1, window=24):
        super().__init__(n_nodes, in_features, d_model, n_horizons, node_emb)
        self.gat_in = GraphAttention(d_model, gat_heads, dropout)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.gat_out = GraphAttention(d_model, gat_heads, dropout)
        self.head = self.make_head(d_model, n_horizons, dropout)

    def forward(self, x, adj, return_attn=False):
        B, T, N, _ = x.shape
        h = self.embed(x).reshape(B * T, N, -1)
        h, _ = self.gat_in(h, adj)                                       # spatial, every hour
        h = h.view(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1) + self.pos
        h = self.encoder(h)[:, -1].view(B, N, -1)                         # temporal
        h, attn = self.gat_out(h, adj)                                    # spatial, summary
        out = self.residual(x, self.head(h))
        return (out, attn.mean(1)) if return_attn else out


MODEL_CLASSES = {
    "LSTM": TemporalLSTM,
    "Transformer": TemporalTransformer,
    "GAT-Transformer": GATTransformer,
    # "Robust" = same architecture, trained with random station dropout (whole input windows of
    # ~15% of stations hidden) so it learns to cope with failed sensors.
    "Robust LSTM": TemporalLSTM,
    "Robust GAT-Transformer": GATTransformer,
}
STATION_DROPOUT = {"Robust LSTM": 0.15, "Robust GAT-Transformer": 0.15}


def slug(name):
    """File-name-safe model name, e.g. 'Robust GAT-Transformer' -> 'robust_gat_transformer'."""
    return name.lower().replace("-", "_").replace(" ", "_")


def build_model(name, n_nodes, n_horizons, in_features=8, window=24):
    cls = MODEL_CLASSES[name]
    kw = dict(n_nodes=n_nodes, in_features=in_features, n_horizons=n_horizons)
    if cls is not TemporalLSTM:
        kw["window"] = window
    return cls(**kw)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, T, N, F_ = 4, 24, 36, 8
    x, adj = torch.randn(B, T, N, F_), torch.rand(N, N)
    for name in MODEL_CLASSES:
        m = build_model(name, N, 5)
        print(f"{name:16s} out={tuple(m(x, adj).shape)} params={count_params(m):,}")
    print("Persistence", tuple(Persistence([1, 3, 6, 12, 24])(x).shape),
          "SeasonalNaive", tuple(SeasonalNaive([1, 3, 6, 12, 24])(x).shape))
