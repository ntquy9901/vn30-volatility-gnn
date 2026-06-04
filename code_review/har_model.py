"""
GNN-HAR model: 2-layer GraphSAGE backbone + MIMO prediction heads.

Input:  (31, 3) node features [rv_d, rv_w, rv_m] per snapshot
Output: (31, 4) predicted RV per node per horizon [h=1,5,10,20]
Loss computed on nodes 1-30 only (VNINDEX node_0 masked out).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GNNHARModel(nn.Module):
    """2-layer GraphSAGE + MIMO output heads for RV forecasting."""

    def __init__(self, in_channels=3, hidden=16, horizons=None, dropout=0.3):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 10, 20]
        self.horizons = list(horizons)
        self.conv1 = SAGEConv(in_channels, hidden, aggregator_type="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggregator_type="mean")
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, g, x: torch.Tensor) -> torch.Tensor:
        """
        g : DGLGraph, 31 nodes
        x : (31, in_channels)
        Returns (31, len(horizons))
        """
        x = F.elu(self.conv1(g, x))
        x = self.drop(x)
        x = F.elu(self.conv2(g, x))
        x = self.drop(x)
        return torch.cat([h(x) for h in self.heads], dim=1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
