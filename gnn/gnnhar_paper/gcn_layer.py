"""
GraphConvLayer - GCN (Graph Convolutional Network) layer from GNNHAR paper.

Paper: "Forecasting Realized Volatility with Spillover Effects:
         Perspectives from Graph Neural Networks" (IJF 2024)

GCN formulation (Kipf & Welling 2017, simplified version used in paper):
    H = A * X * W
where:
    A : (N, N) adjacency matrix (pre-normalized)
    X : (N, D_in) node feature matrix
    W : (D_in, D_out) learnable weight matrix
    H : (N, D_out) output embeddings

Key differences from GraphSAGE:
- GCN: full-batch convolution AXW (spectral approach)
- GraphSAGE: sampling-based aggregation (spatial approach)
- GCN requires normalized adjacency for stability
- GraphSAGE is inductive, GCN is transductive

Input:  (N, in_features)  node features
Output: (N, out_features) node embeddings
"""
import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):
    """
    Single GCN layer: H = A @ (X @ W) [+ b]

    The paper uses a simplified version without the symmetric
    normalization D^(-1/2) A D^(-1/2) from Kipf & Welling.
    Normalization is handled externally in GLASSO adjacency construction.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Weight matrix: (in_features, out_features)
        # Xavier initialization with ReLU gain: preserves variance through
        # the linear transform when followed by ReLU activation.
        # gain=2.0 for ReLU means sqrt(2/fan_in) instead of sqrt(1/fan_in).
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features)
        )
        nn.init.xavier_uniform_(
            self.weight,
            gain=nn.init.calculate_gain('relu')
        )

        # Bias: (1, out_features) broadcast across all nodes
        # Initialized to ones (paper choice, unlike typical zeros)
        # Ones initialization ensures initial outputs are in positive range
        # before ReLU, avoiding "dying neurons" in early training.
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)
        else:
            self.bias = None

    def forward(
        self,
        node_feature: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: H = A @ (X @ W) [+ b]

        Args:
            node_feature: (N, in_features) node features at current timestep
            adj: (N, N) normalized adjacency matrix

        Returns:
            (N, out_features) aggregated node embeddings
        """
        # Step 1: Linear transform per-node: X @ W
        # Shape: (N, in_features) @ (in_features, out_features) -> (N, out_features)
        # Each node independently transforms its features (no message passing yet)
        h = torch.matmul(node_feature, self.weight)

        # Step 2: Message passing via adjacency: A @ h
        # Shape: (N, N) @ (N, out_features) -> (N, out_features)
        # Each node's output = weighted sum of its neighbors' h values
        # In the paper's GLASSO adjacency, weights are normalized correlation strengths.
        output = torch.matmul(adj, h)

        # Step 3: Add bias if present
        # Shape: (N, out_features) + (1, out_features) -> (N, out_features)
        # Broadcast: same bias added to all nodes
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return f'GraphConvLayer(in_features={self.weight.shape[0]}, ' \
               f'out_features={self.weight.shape[1]}, bias={self.bias is not None})'
