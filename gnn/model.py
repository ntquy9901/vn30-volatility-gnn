"""
VolatilityGNN: 2-layer GraphSAGE + MLP head for realized volatility prediction.

Architecture (FR18-FR19):
    node features (31, 384) -> SAGEConv(384->64) -> ReLU -> Dropout
                             -> SAGEConv(64->32)  -> ReLU -> Dropout
                             -> Linear(32->16)    -> ReLU
                             -> Linear(16->1)
    Loss computed only on VN30 stocks (loss_mask excludes node_0 = VNINDEX).

Usage:
    python gnn/model.py
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class VolatilityGNN(nn.Module):
    """
    GraphSAGE encoder + MLP regression head.

    Args:
        in_dim    : Node feature dimension (384 for Moirai2-small).
        hidden    : List of GraphSAGE output dimensions, e.g. [64, 32].
        mlp_hidden: List of MLP hidden dimensions, e.g. [16].
        dropout   : Dropout probability applied after each GNN layer.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden: list[int] | None = None,
        mlp_hidden: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]
        if mlp_hidden is None:
            mlp_hidden = [16]

        self.dropout = nn.Dropout(dropout)

        # GraphSAGE layers
        dims = [in_dim] + hidden
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(len(hidden))]
        )

        # MLP head
        mlp_dims = [hidden[-1]] + mlp_hidden + [1]
        layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                layers.append(nn.ReLU())
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : Node features (num_nodes, in_dim)
            edge_index : (2, E) edge connectivity

        Returns:
            Predicted RV per node, shape (num_nodes, 1).
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        return self.head(x)   # (num_nodes, 1)

    def masked_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss over VN30 stocks only (VNINDEX excluded via loss_mask).

        Args:
            pred      : (num_nodes, 1)
            target    : (num_nodes, 1)
            loss_mask : (num_nodes,) bool — True for nodes included in loss
        """
        pred_m   = pred[loss_mask].squeeze(-1)
        target_m = target[loss_mask].squeeze(-1)
        return torch.mean((pred_m - target_m) ** 2)


# ── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    model = VolatilityGNN(in_dim=384, hidden=[64, 32], mlp_hidden=[16], dropout=0.3)
    print(f"VolatilityGNN parameter count: {sum(p.numel() for p in model.parameters()):,}")
    for name, p in model.named_parameters():
        print(f"  {name:40s}  {tuple(p.shape)}")

    # Dummy forward pass with 31 nodes
    x          = torch.randn(31, 384)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    loss_mask  = torch.ones(31, dtype=torch.bool)
    loss_mask[0] = False

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    print(f"\nForward pass OK: output shape = {out.shape}")

    # Loss check
    target = torch.rand(31, 1) * 0.02
    loss   = model.masked_loss(out, target, loss_mask)
    print(f"Masked MSE loss = {loss.item():.6f}  (over 30 VN30 nodes)")
    print("\nmodel.py OK.")
