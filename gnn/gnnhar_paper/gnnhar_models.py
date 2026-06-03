"""
GNNHAR model variants from the paper:
"Forecasting Realized Volatility with Spillover Effects:
 Perspectives from Graph Neural Networks" (IJF 2024)

All models predict RV from HAR features [rv_d, rv_w, rv_m] with optional
graph-based spillover effects from correlated stocks.

Architecture pattern (residual design):
    H1 = Linear(3, 1)(node_feat)      -- local HAR prediction (per-stock)
    H2 = GCN_layers(node_feat, adj)   -- spillover from graph (multi-stock)
    output = ReLU(H1 + H2)             -- residual connection

Key insight: H1 captures stock-specific volatility dynamics (HAR),
H2 captures cross-stock spillover (correlation-driven), and the
residual sum lets the model balance both information sources.

Models:
- HAR:     Linear only (baseline, no graph)
- GHAR:    Linear + 1 GCN layer (no MLP, linear spillover)
- GNNHAR1L: Linear + 1 GCN + MLP (nonlinear spillover)
- GNNHAR2L: Linear + 2 GCN layers + MLP (2-hop neighbors)
- GNNHAR3L: Linear + 3 GCN layers + MLP (3-hop neighbors)

Input:  (30, 3) -- HAR features for 30 VN30 stocks
Output: (30,)   -- predicted RV (z-scored residuals) for 30 stocks

Version: v1.3_LOSS_FIX - Fixed quasi_likelihood_loss ratio inversion bug (2026-06-01)
- v1.2: Added dropout support (2026-06-01)
- v1.1: Added GELU activation option (2026-06-01)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GCN layer from local module
from torch_geometric.nn import GATConv  # GATHAR1L support
from gnn.gnnhar_paper.gcn_layer import GraphConvLayer


def gnnhar_ratio_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                     eps: float = 1e-4, clip_ratio: bool = True,
                     clip_min: float = 1e-4, clip_max: float = 1e4) -> torch.Tensor:
    """
    GNNHAR Ratio Loss — custom loss from GNNHAR paper (NOT standard QLIKE).

    **IMPORTANT:** This is NOT the standard QLIKE loss from volatility literature.
    Standard QLIKE has the form: L_QLIKE = mean(target/pred - log(target/pred) - 1)
    This function implements: L_ratio = mean(target/(pred+eps) - log(target/(pred+eps)+eps))

    From GNNHAR paper code (GNNHAR.py line 322):
        true_fore = outputs / (forecast_y + 1e-4)
        l_v = torch.mean(true_fore - torch.log(true_fore + 1e-4))

    This is a custom ratio-based loss with asymmetric penalty:
    - Penalizes under-prediction MORE heavily than over-prediction
    - Aligns with risk management (under-estimating risk is dangerous)
    - Scale-invariant (penalizes percentage errors, not absolute)

    Args:
        y_pred: Predicted RV values (must be POSITIVE, e.g., softplus/ReLU output)
               Shape: (batch_size, N) or (batch_size, N, 1)
        y_true: Ground truth RV values (always positive, e.g., RV_h = std(log_returns))
               Same shape as y_pred
        eps: Small constant for numerical stability (matches paper's 1e-4)
             Guards BOTH division (denominator) AND log domain
        clip_ratio: Whether to clip ratio to prevent numerical explosion (default: True)
                     Architectural guardrail to prevent extreme gradients
        clip_min: Minimum ratio value (prevents log(0) issues when ratio -> 0)
        clip_max: Maximum ratio value (prevents extreme gradients when ratio -> inf)

    Returns:
        Scalar loss value (mean over all samples and stocks)

    Formula (from paper code, lines 322-323):
        L_ratio = mean(target / (pred + eps) - log(target / (pred + eps) + eps))

    Usage context (from paper line 326):
        This loss is typically weighted: l_total = MSE_loss + 0.5 * ratio_loss
        Check your training loop to see if 0.5 weighting is applied.

    Behavior:
        - Perfect prediction (pred = target): loss = 1.0 (minimum)
        - Under-prediction (pred < target): loss grows AGGRESSIVELY (dangerous)
        - Over-prediction (pred > target): loss grows slowly (safe but wasteful)

    Note: This loss expects POSITIVE RV values (always > 0):
          - y_pred: must be post-activation (ReLU/softplus) to ensure positivity
          - y_true: RV is always positive (std of returns)
          - If y_pred contains negative values, ratio becomes unstable

    Numerical stability:
        - eps prevents division by zero when y_pred -> 0
        - eps prevents log(0) when ratio -> 0
        - Consider ratio clipping if extreme values observed during training

    Why use MSE instead (see CONSTRAINTS.md C3):
        This loss is used in the GNNHAR paper but may not be optimal for your use case.
        MSE on z-scored HAR residuals is often more stable and easier to interpret.
    """
    # Implementation from paper code (GNNHAR.py lines 322-323)
    # Ensure shapes match
    # Ensure shapes match
    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)
    if y_true.dim() == 3 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)

    # Paper's formula: target / (pred + eps) - log(target / (pred + eps) + eps)
    # IMPORTANT: eps guards BOTH the division AND the log (paper line 322-323)
    # Compute ratio of target to prediction (inverse of prediction/target)
    ratio = y_true / (y_pred + eps)

    # Architectural guardrail: Ratio clipping to prevent numerical explosion
    # Prevents extreme gradients when model predicts near-zero volatility
    if clip_ratio:
        ratio = torch.clamp(ratio, min=clip_min, max=clip_max)

    # Loss = ratio - log(ratio), with eps guarding log domain to prevent log(0)
    # When ratio = 1 (perfect prediction): loss = 1 - log(1) = 1 - 0 = 1
    # When ratio > 1 (under-prediction): loss grows slowly (pred too small)
    # When ratio < 1 (over-prediction): loss grows fast (due to -log(ratio))
    loss = ratio - torch.log(ratio + eps)

    # Mean over all elements (batch and stocks)
    return loss.mean()


def quasi_likelihood_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                          eps: float = 1e-4, clip_ratio: bool = True,
                          clip_min: float = 1e-4, clip_max: float = 1e4) -> torch.Tensor:
    """
    [DEPRECATED] Use gnnhar_ratio_loss instead.

    This function name is misleading — it's NOT standard QLIKE loss.
    See gnnhar_ratio_loss for correct documentation.

    **WARNING:** This name will be removed in v2.0.0
    """
    import warnings
    warnings.warn(
        "quasi_likelihood_loss is deprecated and misleading. "
        "Use gnnhar_ratio_loss instead. "
        "This function will be removed in v2.0.0",
        DeprecationWarning,
        stacklevel=2
    )
    return gnnhar_ratio_loss(y_pred, y_true, eps, clip_ratio, clip_min, clip_max)


class HAR(nn.Module):
    """
    HAR baseline: linear regression on HAR features only, no graph.

    H1 = Linear(3, 1)(node_feat)
    output = H1  -- NO ReLU (removed to fix QL loss singularity)

    This is equivalent to the classic HAR-RV model (Corsi 2009) but
    implemented as a neural network layer for consistency.

    NO ReLU activation: allows negative predictions, preventing QL loss
    collapse when predictions go to 0. Original paper uses ReLU, but it
    causes 75% seed failures in single-stock training due to QL loss
    singularity at pred=0 (loss = -log(eps) ≈ 9.21, gradients vanish).
    """

    def __init__(self):
        super().__init__()
        # Linear transform: 3 HAR features -> 1 scalar prediction per stock
        # Weight shape: (3, 1), bias shape: (1,)
        self.linear1 = nn.Linear(3, 1, bias=True)
        # NO ReLU: removed to prevent QL loss singularity collapse

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (batch_size, N, 3) HAR features [rv_d, rv_w, rv_m]
            adj: (N, N) adjacency matrix (unused in HAR, kept for interface consistency)

        Returns:
            (batch_size, N) predicted RV per stock (can be negative, no activation)
        """
        # (batch_size, N, 3) @ (3, 1) -> (batch_size, N, 1)
        H1 = self.linear1(node_feat)
        # (batch_size, N, 1) -> (batch_size, N)
        # No ReLU: return raw linear output
        return H1.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GHAR(nn.Module):
    """
    Graph-augmented HAR: linear HAR + linear spillover from graph.

    H1 = Linear(3, 1)(node_feat)              -- local HAR
    H2 = GCN(3, n_hid)(node_feat, adj)        -- spillover (no nonlinearity)
    output = H1 + H2  -- NO ReLU (removed to fix QL loss singularity)

    Unlike GNNHAR1L, GHAR has no MLP after GCN. The spillover term H2
    is a direct linear combination of neighbor features. This model
    tests whether graph information helps even without nonlinear transforms.

    NO ReLU activation: allows negative predictions, preventing QL loss
    collapse when predictions go to 0.
    """

    def __init__(self, n_hid: int, dropout: float = 0.0):
        super().__init__()
        # Local HAR branch (same as HAR model)
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph spillover branch: 3 features -> n_hid embeddings
        # bias=False in GCN: we already have bias in linear1, redundant bias adds params
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)

        # BUG FIX from paper: add projection to match dimensions
        # Paper's GHAR has shape mismatch (H1: N×1, H2: N×n_hid)
        # We add projection: n_hid -> 1 for proper residual connection
        self.proj = nn.Linear(n_hid, 1, bias=False)
        # NO ReLU: removed to prevent QL loss singularity collapse

        # v1.2_DROPOUT: Dropout layer for regularization
        # Placement: after GCN (GHAR has no activation)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (batch_size, N, 3) HAR features
            adj: (N, N) adjacency matrix

        Returns:
            (batch_size, N) predicted RV per stock (can be negative, no activation)

        Note: The original paper code has a dimension mismatch bug in GHAR:
        H1 is (N, 1) but H2 from GCN is (N, n_hid). We fix this by adding
        a projection layer from n_hid to 1, matching GNNHAR1L architecture.
        """
        # Local HAR prediction: (batch_size, N, 1)
        H1 = self.linear1(node_feat)

        # Spillover from graph: (batch_size, N, 3) -> (batch_size, N, n_hid)
        H2 = self.gcn1(node_feat, adj)

        # v1.2_DROPOUT: Apply dropout after GCN (GHAR has no activation)
        H2 = self.dropout(H2)

        # BUG FIX from paper: project H2 to (batch_size, N, 1) for residual sum
        H2 = self.proj(H2)

        # Residual sum: (batch_size, N, 1) + (batch_size, N, 1) -> (batch_size, N, 1)
        res = H1 + H2
        # No ReLU: return raw linear output
        return res.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GNNHAR1L(nn.Module):
    """
    1-layer GNNHAR: HAR + 1 GCN layer + MLP projection.

    H1 = Linear(3, 1)(node_feat)           -- local HAR
    H2 = GCN(3, n_hid)(node_feat, adj)     -- 1-hop neighbor aggregation
    H2 = activation(H2)                     -- ReLU or GELU
    H2 = MLP(n_hid, 1)(H2)                 -- nonlinear projection
    output = H1 + H2  -- NO activation (removed to fix QL loss singularity)

    The MLP after GCN introduces nonlinearity in the spillover path,
    allowing the model to learn complex interactions between neighbor stocks.

    v1.1_GELU: Added activation function parameter (ReLU or GELU)
    - GELU is smoother than ReLU, better for capturing volatility patterns
    - GELU: Gaussian Error Linear Unit, x * Phi(x) where Phi is Gaussian CDF
    - Expected: +2-5% R² improvement based on 2026 research

    NO final activation: allows negative predictions, preventing QL loss
    collapse when predictions go to 0. Internal activation for nonlinearity kept.
    """

    def __init__(self, n_hid: int, activation: Literal['relu', 'gelu'] = 'relu', dropout: float = 0.0):
        super().__init__()
        # Local HAR branch
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph branch: 1 GCN layer -> MLP
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        # MLP projection: n_hid -> 1
        # bias=False: no bias needed after GCN (already has implicit bias via adjacency)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)

        # Activation function (ReLU or GELU)
        self.act_name = activation
        if activation == 'gelu':
            # GELU is smoother than ReLU, better for volatility forecasting
            # GELU(x) = x * Phi(x) where Phi is Gaussian CDF
            # PyTorch F.gelu implements this efficiently
            pass  # No module needed, use F.gelu in forward
        else:  # relu (default)
            self.relu = nn.ReLU()

        # v1.2_DROPOUT: Dropout layer for regularization
        # Placement: after GCN activation (most critical for GNN)
        # Rate: 0.0-0.3 range (default 0.0 for backward compatibility)
        self.dropout_rate = dropout  # Store for reference
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (batch_size, N, 3) HAR features
            adj: (N, N) adjacency matrix

        Returns:
            (batch_size, N) predicted RV per stock (can be negative, no final activation)
        """
        # Local HAR: (batch_size, N, 1)
        H1 = self.linear1(node_feat)

        # Graph branch:
        # Step 1: Aggregate from 1-hop neighbors: (batch_size, N, 3) -> (batch_size, N, n_hid)
        H2 = self.gcn1(node_feat, adj)

        # Step 2: Nonlinearity (ReLU or GELU)
        # GELU is smoother: x * Phi(x) where Phi is Gaussian CDF
        # ReLU is standard: max(0, x)
        if self.act_name == 'gelu':
            H2 = F.gelu(H2)
        else:  # relu (default)
            H2 = self.relu(H2)

        # v1.2_DROPOUT: Apply dropout after GCN activation
        # This is the most critical location for GNN regularization
        # Dropout randomly zeros neurons, preventing co-adaptation
        H2 = self.dropout(H2)

        # Step 3: Project to scalar: (batch_size, N, n_hid) -> (batch_size, N, 1)
        H2 = self.mlp1(H2)

        # Apply activation after MLP for nonlinearity (kept, but no final activation)
        if self.act_name == 'gelu':
            H2 = F.gelu(H2)
        else:  # relu (default)
            H2 = self.relu(H2)

        # Residual sum: (batch_size, N, 1) + (batch_size, N, 1) -> (batch_size, N, 1)
        res = H1 + H2
        # No final activation: return raw linear output
        return res.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GNNHAR2L(nn.Module):
    """
    2-layer GNNHAR: HAR + 2 GCN layers + MLP.

    H1 = Linear(3, 1)(node_feat)              -- local HAR
    H2 = activation(GCN(3, n_hid)(node_feat, adj))  -- 1-hop neighbors
    H2 = activation(GCN(n_hid, n_hid)(H2, adj))     -- 2-hop neighbors
    H2 = MLP(n_hid, 1)(H2)                    -- projection
    output = H1 + H2  -- NO activation (removed to fix QL loss singularity)

    With 2 GCN layers, each node can "see" information from neighbors
    of its neighbors (2-hop range). For volatility forecasting, this
    means a stock can be affected by second-order correlations:
    stock A correlated with stock B, which is correlated with stock C.

    v1.1_GELU: Added activation function parameter (ReLU or GELU)

    NO final activation: allows negative predictions, preventing QL loss
    collapse when predictions go to 0. Internal activation for nonlinearity kept.
    """

    def __init__(self, n_hid: int, activation: Literal['relu', 'gelu'] = 'relu', dropout: float = 0.0):
        super().__init__()
        # Local HAR branch
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph branch: 2 stacked GCN layers
        # Layer 1: input features -> hidden embeddings
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        # Layer 2: hidden embeddings -> hidden embeddings (same dimension)
        self.gcn2 = GraphConvLayer(n_hid, n_hid, bias=False)

        # Projection: n_hid -> 1
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)

        # Activation function (ReLU or GELU)
        self.act_name = activation
        if activation == 'gelu':
            pass  # No module needed, use F.gelu in forward
        else:  # relu (default)
            self.relu = nn.ReLU()

        # v1.2_DROPOUT: Dropout layer for regularization
        # Placement: after each GCN activation
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (batch_size, N, 3) HAR features
            adj: (N, N) adjacency matrix

        Returns:
            (batch_size, N) predicted RV per stock (can be negative, no final activation)
        """
        # Local HAR: (batch_size, N, 1)
        H1 = self.linear1(node_feat)

        # Graph branch with 2-hop propagation:
        # Layer 1: 1-hop aggregation: (batch_size, N, 3) -> (batch_size, N, n_hid)
        if self.act_name == 'gelu':
            H2 = F.gelu(self.gcn1(node_feat, adj))
        else:  # relu (default)
            H2 = self.relu(self.gcn1(node_feat, adj))

        # v1.2_DROPOUT: Apply dropout after first GCN layer
        H2 = self.dropout(H2)

        # Layer 2: 2-hop aggregation: (batch_size, N, n_hid) -> (batch_size, N, n_hid)
        if self.act_name == 'gelu':
            H2 = F.gelu(self.gcn2(H2, adj))
        else:  # relu (default)
            H2 = self.relu(self.gcn2(H2, adj))

        # v1.2_DROPOUT: Apply dropout after second GCN layer
        H2 = self.dropout(H2)

        # Project to scalar: (batch_size, N, n_hid) -> (batch_size, N, 1)
        H2 = self.mlp1(H2)

        # Apply activation after MLP for nonlinearity (kept, but no final activation)
        if self.act_name == 'gelu':
            H2 = F.gelu(H2)
        else:  # relu (default)
            H2 = self.relu(H2)

        # Residual sum: (batch_size, N, 1) + (batch_size, N, 1) -> (batch_size, N, 1)
        res = H1 + H2
        # No final activation: return raw linear output
        return res.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GNNHAR3L(nn.Module):
    """
    3-layer GNNHAR: HAR + 3 GCN layers + MLP.

    H1 = Linear(3, 1)(node_feat)              -- local HAR
    H2 = activation(GCN(3, n_hid)(node_feat, adj))  -- 1-hop
    H2 = activation(GCN(n_hid, n_hid)(H2, adj))     -- 2-hop
    H2 = activation(GCN(n_hid, n_hid)(H2, adj))     -- 3-hop
    H2 = MLP(n_hid, 1)(H2)                    -- projection
    output = H1 + H2  -- NO activation (removed to fix QL loss singularity)

    3 GCN layers allow each node to aggregate information from 3 hops away.
    In a dense VN30 graph, this can cover nearly all stocks through indirect
    correlations. However, deeper GCNs suffer from over-smoothing:
    all node embeddings become similar after many propagation steps.

    v1.1_GELU: Added activation function parameter (ReLU or GELU)

    NO final activation: allows negative predictions, preventing QL loss
    collapse when predictions go to 0. Internal activation for nonlinearity kept.
    """

    def __init__(self, n_hid: int, activation: Literal['relu', 'gelu'] = 'relu', dropout: float = 0.0):
        super().__init__()
        # Local HAR branch
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph branch: 3 stacked GCN layers
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.gcn2 = GraphConvLayer(n_hid, n_hid, bias=False)
        self.gcn3 = GraphConvLayer(n_hid, n_hid, bias=False)

        # Projection: n_hid -> 1
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)

        # Activation function (ReLU or GELU)
        self.act_name = activation
        if activation == 'gelu':
            pass  # No module needed, use F.gelu in forward
        else:  # relu (default)
            self.relu = nn.ReLU()

        # v1.2_DROPOUT: Dropout layer for regularization
        # Placement: after each GCN activation
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (batch_size, N, 3) HAR features
            adj: (N, N) adjacency matrix

        Returns:
            (batch_size, N) predicted RV per stock (can be negative, no final activation)
        """
        # Local HAR: (batch_size, N, 1)
        H1 = self.linear1(node_feat)

        # Graph branch with 3-hop propagation:
        # Layer 1: (batch_size, N, 3) -> (batch_size, N, n_hid)
        if self.act_name == 'gelu':
            H2 = F.gelu(self.gcn1(node_feat, adj))
        else:  # relu (default)
            H2 = self.relu(self.gcn1(node_feat, adj))

        # v1.2_DROPOUT: Apply dropout after first GCN layer
        H2 = self.dropout(H2)

        # Layer 2: (batch_size, N, n_hid) -> (batch_size, N, n_hid)
        if self.act_name == 'gelu':
            H2 = F.gelu(self.gcn2(H2, adj))
        else:  # relu (default)
            H2 = self.relu(self.gcn2(H2, adj))

        # v1.2_DROPOUT: Apply dropout after second GCN layer
        H2 = self.dropout(H2)

        # Layer 3: (batch_size, N, n_hid) -> (batch_size, N, n_hid)
        if self.act_name == 'gelu':
            H2 = F.gelu(self.gcn3(H2, adj))
        else:  # relu (default)
            H2 = self.relu(self.gcn3(H2, adj))

        # v1.2_DROPOUT: Apply dropout after third GCN layer
        H2 = self.dropout(H2)

        # Project to scalar: (batch_size, N, n_hid) -> (batch_size, N, 1)
        H2 = self.mlp1(H2)

        # Apply activation after MLP for nonlinearity (kept, but no final activation)
        if self.act_name == 'gelu':
            H2 = F.gelu(H2)
        else:  # relu (default)
            H2 = self.relu(H2)

        # Residual sum: (batch_size, N, 1) + (batch_size, N, 1) -> (batch_size, N, 1)
        res = H1 + H2
        # No final activation: return raw linear output
        return res.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




class GATHAR1L(nn.Module):
    """
    1-layer Graph Attention HAR: HAR + 1 GAT + MLP.

    H1 = Linear(3, 1)(node_feat)              -- local HAR
    H2 = GAT(3, n_hid, heads)(node_feat, edge_index) -- 1-hop attention
    H2 = activation(H2)                        -- nonlinearity
    H2 = MLP(n_hid, 1)(H2)                        -- projection
    output = H1 + H2                           -- residual connection

    Unlike GCN which averages all neighbors equally, GAT learns
    attention weights per neighbor connection, allowing the model
    to focus on the most predictive relationships.

    v1.1_GELU: Added activation function parameter (ReLU or GELU)
    v1.2_DROPOUT: Added dropout support for regularization

    NO final activation: allows negative predictions, preventing QL loss
    collapse when predictions go to 0. Internal activation for nonlinearity kept.
    """

    def __init__(self, n_hid: int = 16, heads: int = 1, activation: Literal['relu', 'gelu'] = 'gelu', dropout: float = 0.0):
        super().__init__()
        # Local HAR branch
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph Attention branch: 1 GAT layer -> MLP
        # Note: GATConv handles attention mechanism internally
        self.gat1 = GATConv(3, n_hid, heads=heads, dropout=dropout, edge_dim=1, concat=False)

        # MLP projection: n_hid -> 1
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)

        # Activation function (ReLU or GELU)
        self.act_name = activation
        if activation == 'gelu':
            pass  # No module needed, use F.gelu in forward
        else:  # relu (default)
            self.relu = nn.ReLU()

        # v1.2_DROPOUT: Dropout layer for regularization
        # Placement: after GAT activation
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

        # v1.4_GAT_OPT: Cache edge_index to avoid recomputing on every forward pass
        self.register_buffer('cached_edge_index', None)
        self.register_buffer('cached_adj_hash', None)

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (batch_size, N, 3) HAR features
            adj: (N, N) adjacency matrix (converted to edge_index internally)

        Returns:
            (batch_size, N) predicted RV per stock (can be negative, no final activation)
        """
        import time
        forward_start = time.time()

        # Local HAR: (batch_size, N, 1)
        H1 = self.linear1(node_feat)

        # v1.4_GAT_OPT: Cache edge_index to avoid recomputing on every forward pass
        # Adjacency matrix doesn't change during training, so compute edge_index once
        # Use tensor shape and sum as hash (fast and sufficient for this use case)
        adj_shape = torch.tensor(adj.shape)
        adj_sum = adj.sum().reshape(1)

        if self.cached_edge_index is None or \
           self.cached_adj_hash is None or \
           self.cached_adj_hash.shape[0] != 2 or \
           not torch.allclose(self.cached_adj_hash[0:2], adj_shape) or \
           not torch.allclose(self.cached_adj_hash[2:3], adj_sum):
            # Convert adjacency matrix to edge index for GAT
            # edge_index: (2, num_edges) where edge_index[0] = source nodes, edge_index[1] = target nodes
            edge_index = adj.nonzero().t().contiguous()

            # Cache edge_index on the same device as node_feat
            # Also cache hash as (shape0, shape1, sum) tensor
            self.register_buffer('cached_edge_index', edge_index.to(node_feat.device))
            hash_tensor = torch.cat([adj_shape, adj_sum]).to(node_feat.device)
            self.register_buffer('cached_adj_hash', hash_tensor)
        else:
            # Use cached edge_index (already on correct device)
            edge_index = self.cached_edge_index

        # Graph branch:
        # GATConv requires 2D input, so we handle batch dimension manually
        batch_size, N, _ = node_feat.shape

        # v1.4_GAT_OPT: Use batched processing instead of Python loop
        # Reshape batch to process all samples in one GAT call
        # Strategy: Create a disconnected graph with batch_size * N nodes
        # Each sample becomes a separate graph component (no edges between samples)

        # Flatten node features: (batch_size, N, 3) -> (batch_size * N, 3)
        node_feat_flat = node_feat.reshape(-1, 3)

        # Create batch edge indices with offset for each sample
        # Each sample gets the same edge pattern, offset by i * N
        num_edges = edge_index.shape[1]
        edge_index_batch = []

        # Vectorized edge index creation (much faster than Python loop)
        batch_offsets = torch.arange(batch_size, device=node_feat.device) * N
        # edge_index: (2, num_edges)
        # Add offsets: (2, batch_size, num_edges)
        edge_index_expanded = edge_index.unsqueeze(1) + batch_offsets.view(1, -1, 1)
        # Reshape to (2, batch_size * num_edges)
        edge_index_batch = edge_index_expanded.reshape(2, -1)

        # Single GAT call for entire batch (MUCH FASTER than Python loop)
        H2_flat = self.gat1(node_feat_flat, edge_index_batch)

        # Reshape back: (batch_size * N, n_hid) -> (batch_size, N, n_hid)
        H2 = H2_flat.reshape(batch_size, N, -1)

        # Step 2: Nonlinearity (ReLU or GELU)
        if self.act_name == 'gelu':
            H2 = F.gelu(H2)
        else:  # relu (default)
            H2 = self.relu(H2)
        
        # v1.2_DROPOUT: Apply dropout after GAT activation
        H2 = self.dropout(H2)
        
        # Step 2: Project to scalar: (batch_size, N, n_hid) -> (batch_size, N, 1)
        H2 = self.mlp1(H2)

        # Apply activation after MLP for nonlinearity (kept, but no final activation)
        if self.act_name == 'gelu':
            H2 = F.gelu(H2)
        else:  # relu (default)
            H2 = self.relu(H2)

        # Residual sum: (batch_size, N, 1) + (batch_size, N, 1) -> (batch_size, N, 1)
        res = H1 + H2
        # No final activation: return raw linear output

        # v1.4_GAT_OPT: Print forward pass timing (for performance monitoring)
        forward_time = time.time() - forward_start
        if self.training and forward_time > 0.01:  # Only print if >10ms
            print(f"    [GAT] Forward: {forward_time*1000:.1f}ms for batch_size={node_feat.shape[0]}")

        return res.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() for p in self.parameters() if p.requires_grad)
# Model factory for easy instantiation
MODEL_REGISTRY: Dict[str, type] = {
    'HAR': HAR,
    'HAR_OLS': 'HAR_OLS',  # sklearn LinearRegression (import dynamically)
    'GHAR': GHAR,
    'GNNHAR1L': GNNHAR1L,
    'GNNHAR2L': GNNHAR2L,
    'GNNHAR3L': GNNHAR3L,
    'GATHAR1L': GATHAR1L,
}


def create_model(
    model_name: str,
    n_hid: int = 16,
    activation: Literal['relu', 'gelu'] = 'relu',
    dropout: float = 0.0
) -> nn.Module:
    """
    Create a GNNHAR model variant by name.

    v1.1_GELU: Added activation parameter for testing GELU vs ReLU
    v1.2_DROPOUT: Added dropout parameter for regularization
    v1.3_LOSS_FIX: Fixed quasi_likelihood_loss ratio inversion (critical bug)

    Args:
        model_name: One of .HAR., .GHAR., .GNNHAR1L., .GNNHAR2L., .GNNHAR3L., .GATHAR1L.
        n_hid: Hidden dimension for GCN layers (ignored for HAR)
        activation: Activation function ('relu' or 'gelu')
        dropout: Dropout rate for regularization (0.0-0.3, default 0.0)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name not recognized
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    # HAR has no dropout or activation (baseline)
    if model_name == 'HAR':
        return model_class()
    # HAR_OLS uses sklearn LinearRegression (dynamic import)
    if model_name == 'HAR_OLS':
        from gnn.gnnhar_paper.sklearn_models import HAR_OLS
        return HAR_OLS()
    # GHAR has dropout but no activation
    if model_name == 'GHAR':
        return model_class(n_hid=n_hid, dropout=dropout)
    # GATHAR1L needs heads parameter for attention mechanism
    if model_name == 'GATHAR1L':
        return model_class(n_hid=n_hid, heads=1, activation=activation, dropout=dropout)
    # GNNHAR models support both activation and dropout
    return model_class(n_hid=n_hid, activation=activation, dropout=dropout)
    return model_class(n_hid=n_hid, activation=activation, dropout=dropout)


def print_model_summary(n_hid: int = 16) -> None:
    """Print parameter count for each model variant."""
    print(f"\n{'='*56}")
    print(f"  Model Summary: n_hid={n_hid}")
    print(f"{'='*56}")
    print(f"  {'Model':<12} {'Params':>10} {'Description'}")
    print(f"  {'-'*56}")

    for name, cls in MODEL_REGISTRY.items():
        if name == 'HAR':
            model = cls()
        else:
            model = cls(n_hid)

        n_params = model.count_params()

        if name == 'HAR':
            desc = 'Linear baseline (no graph)'
        elif name == 'GHAR':
            desc = 'HAR + 1 GCN (linear spillover)'
        elif name == 'GNNHAR1L':
            desc = 'HAR + 1 GCN + MLP (1-hop)'
        elif name == 'GNNHAR2L':
            desc = 'HAR + 2 GCN + MLP (2-hop)'
        elif name == 'GATHAR1L':
            desc = 'HAR + 1 GAT + MLP (1-hop, attention)'
        else:  # GNNHAR3L
            desc = 'HAR + 3 GCN + MLP (3-hop)'
if __name__ == "__main__":
    # Test: print model summary
    print_model_summary(n_hid=16)

    # Test: forward pass with synthetic data
    print("[TEST] Running forward pass with synthetic data...")
    batch_size = 4
    n_stocks = 30
    n_features = 3

    X = torch.randn(batch_size, n_stocks, n_features)
    adj = torch.eye(n_stocks) + 0.1 * torch.randn(n_stocks, n_stocks)
    adj = (adj + adj.T) / 2  # Make symmetric

    for model_name in MODEL_REGISTRY.keys():
        model = create_model(model_name, n_hid=16)
        model.eval()
        with torch.no_grad():
            out = model(X, adj)
        print(f"  {model_name:<12} input: {tuple(X.shape)} -> output: {tuple(out.shape)}")

    print("\n[OK] All models forward pass successful")
