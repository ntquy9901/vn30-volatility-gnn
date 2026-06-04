"""
Graph Builder for VN30 multi-stock volatility forecasting.

Constructs adjacency matrices for 30 VN30 stocks using:
- Pearson correlation threshold (simple, interpretable)
- GLASSO / Graphical Lasso (paper's method, captures partial correlations)

Output: (30, 30) adjacency matrix with row-wise normalization.

Usage:
    builder = GraphBuilder(method='pearson', threshold=0.3)
    adj = builder.build_adjacency(returns, end_date='2025-12-31')

    builder_glasso = GraphBuilder(method='glasso')
    adj_glasso = builder_glasso.build_adjacency(returns, end_date='2025-12-31')
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_log_returns
from gnn.build_graph import VN30_TICKERS


class GraphBuilder:
    """
    Build adjacency matrix for VN30 stocks.

    Methods:
        - 'pearson': Threshold on Pearson correlation (fast, simple)
        - 'glasso': Graphical Lasso on precision matrix (paper's method, slower)

    Normalization: Row-wise normalization (sum to 1 per row)
    This ensures GCN aggregation computes weighted average of neighbors.
    """

    def __init__(
        self,
        method: Literal['pearson', 'glasso'] = 'pearson',
        threshold: float = 0.3,
        corr_window: int = 60,
        glasso_alpha: float = 0.01,
    ):
        """
        Initialize graph builder.

        Args:
            method: 'pearson' or 'glasso'
            threshold: Correlation threshold for pearson method (edge if |corr| > threshold)
            corr_window: Rolling window for correlation calculation (days)
            glasso_alpha: Sparsity parameter for GLASSO (higher = more sparse)
        """
        self.method = method
        self.threshold = threshold
        self.corr_window = corr_window
        self.glasso_alpha = glasso_alpha

        # Cached adjacency matrix
        self.adj_matrix = None
        self.build_date = None

    def compute_correlation(
        self,
        returns: pd.DataFrame,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix using trailing window.

        Args:
            returns: DataFrame with columns = stock tickers, DatetimeIndex
            end_date: Compute correlation using data <= end_date

        Returns:
            (30, 30) correlation matrix
        """
        end_date = pd.Timestamp(end_date)

        # Filter data up to end_date
        mask = returns.index <= end_date
        hist_returns = returns.loc[mask].iloc[-self.corr_window:]

        # Compute Pearson correlation
        # corr(i,j) = cov(returns_i, returns_j) / (std_i * std_j)
        corr_matrix = hist_returns.corr()

        return corr_matrix

    def build_adjacency_pearson(
        self,
        corr_matrix: pd.DataFrame,
    ) -> np.ndarray:
        """
        Build adjacency matrix from correlation threshold.

        Rule: Edge exists if |corr(i,j)| > threshold
        Edge weight = |corr(i,j)| (absolute correlation strength)

        Args:
            corr_matrix: (30, 30) correlation DataFrame

        Returns:
            (30, 30) unnormalized adjacency matrix
        """
        n = corr_matrix.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-loop: each stock connected to itself
                    # This ensures each stock's own features are included in aggregation
                    adj[i, i] = 1.0
                else:
                    # Get correlation value
                    corr_val = corr_matrix.iloc[i, j]

                    # Edge exists if |corr| > threshold
                    if abs(corr_val) > self.threshold:
                        # Edge weight = absolute correlation
                        adj[i, j] = abs(corr_val)

        return adj

    def build_adjacency_glasso(
        self,
        returns: pd.DataFrame,
        end_date: str | pd.Timestamp,
    ) -> np.ndarray:
        """
        Build adjacency matrix using Graphical Lasso (GLASSO).

        GLASSO estimates precision matrix (inverse covariance).
        Precision matrix encodes partial correlations:
            - Zero in precision matrix = conditional independence
            - Non-zero = direct dependency even after controlling for other variables

        Args:
            returns: DataFrame with columns = stock tickers, DatetimeIndex
            end_date: Use data <= end_date for GLASSO estimation

        Returns:
            (30, 30) unnormalized adjacency matrix
        """
        from sklearn.covariance import GraphicalLasso

        end_date = pd.Timestamp(end_date)

        # Filter data up to end_date
        mask = returns.index <= end_date
        hist_returns = returns.loc[mask].iloc[-self.corr_window:]

        # Drop any stocks with NaN values
        hist_returns = hist_returns.dropna(axis=1)

        # Fit GLASSO
        # alpha: Sparsity parameter (higher = more zeros in precision matrix)
        model = GraphicalLasso(alpha=self.glasso_alpha, max_iter=100, verbose=False)
        model.fit(hist_returns)

        # Extract precision matrix
        precision = model.precision_

        # Use absolute partial correlations as edge weights
        # Partial correlation = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
        n = precision.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, i] = 1.0  # Self-loop
                else:
                    # Compute partial correlation
                    partial_corr = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
                    adj[i, j] = abs(partial_corr)

        return adj

    def normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """
        Row-wise normalization of adjacency matrix.

        Normalization: adj_norm[i,j] = adj[i,j] / sum_k(adj[i,k])

        This ensures:
            - Each row sums to 1
            - GCN computes weighted average of neighbors
            - No scale explosion in deep GCN layers

        Args:
            adj: (30, 30) unnormalized adjacency matrix

        Returns:
            (30, 30) normalized adjacency matrix
        """
        # Row-wise sum
        row_sums = adj.sum(axis=1, keepdims=True)

        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0

        # Normalize
        adj_norm = adj / row_sums

        return adj_norm.astype(np.float32)

    def build_adjacency(
        self,
        returns: pd.DataFrame,
        end_date: str | pd.Timestamp,
    ) -> np.ndarray:
        """
        Build normalized adjacency matrix.

        Args:
            returns: DataFrame with columns = stock tickers, DatetimeIndex
            end_date: Build graph using data <= end_date

        Returns:
            (30, 30) normalized adjacency matrix (row-wise sum = 1)
        """
        end_date = pd.Timestamp(end_date)

        print(f"[Graph] Building adjacency matrix (method={self.method}, date={end_date.date()})...")

        if self.method == 'pearson':
            # Compute correlation matrix
            corr_matrix = self.compute_correlation(returns, end_date)
            print(f"  Computed correlation matrix using {self.corr_window}-day window")

            # Build adjacency from threshold
            adj = self.build_adjacency_pearson(corr_matrix)
            print(f"  Threshold: |corr| > {self.threshold}")

        elif self.method == 'glasso':
            # Build adjacency using GLASSO
            adj = self.build_adjacency_glasso(returns, end_date)
            print(f"  GLASSO alpha: {self.glasso_alpha}")

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'pearson' or 'glasso'.")

        # Normalize
        adj_norm = self.normalize_adjacency(adj)

        # Cache result
        self.adj_matrix = adj_norm
        self.build_date = end_date

        # Print statistics
        n_edges = (adj_norm > 0).sum()
        density = n_edges / (30 * 30)
        print(f"  Edges: {n_edges}/900 (density={density:.2%})")

        return adj_norm

    def get_adjacency(self) -> np.ndarray:
        """Return cached adjacency matrix."""
        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix not built. Call build_adjacency() first.")
        return self.adj_matrix


def build_identity_adjacency(n_stocks: int = 30) -> np.ndarray:
    """
    Build identity adjacency matrix (HAR baseline, no graph structure).

    Identity matrix: Each stock only connected to itself.
    This is the HAR baseline (no cross-stock spillover).

    Args:
        n_stocks: Number of stocks (default 30 for VN30)

    Returns:
        (n_stocks, n_stocks) identity matrix
    """
    return np.eye(n_stocks, dtype=np.float32)


# ── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("Graph Builder Test")
    print("="*70)

    # Load returns
    print("\n[Data] Loading returns...")
    close = load_close_prices(PROJECT_ROOT / "data/raw/prices", tickers=VN30_TICKERS)
    returns = compute_log_returns(close)
    print(f"  Returns shape: {returns.shape}")

    end_date = "2025-12-31"

    # Test Pearson method
    print("\n" + "="*70)
    print("Method 1: Pearson Correlation Threshold")
    print("="*70)
    builder_pearson = GraphBuilder(method='pearson', threshold=0.3, corr_window=60)
    adj_pearson = builder_pearson.build_adjacency(returns, end_date)

    print(f"\nAdjacency matrix shape: {adj_pearson.shape}")
    print(f"  Row sums (should be 1.0): {adj_pearson.sum(axis=1)[:5]}")
    print(f"  Min value: {adj_pearson.min():.4f}")
    print(f"  Max value: {adj_pearson.max():.4f}")

    # Test GLASSO method
    print("\n" + "="*70)
    print("Method 2: Graphical Lasso (GLASSO)")
    print("="*70)
    builder_glasso = GraphBuilder(method='glasso', glasso_alpha=0.01, corr_window=60)
    adj_glasso = builder_glasso.build_adjacency(returns, end_date)

    print(f"\nAdjacency matrix shape: {adj_glasso.shape}")
    print(f"  Row sums (should be 1.0): {adj_glasso.sum(axis=1)[:5]}")
    print(f"  Min value: {adj_glasso.min():.4f}")
    print(f"  Max value: {adj_glasso.max():.4f}")

    # Test identity baseline
    print("\n" + "="*70)
    print("Baseline: Identity Adjacency (HAR, no graph)")
    print("="*70)
    adj_identity = build_identity_adjacency(n_stocks=30)
    print(f"\nAdjacency matrix shape: {adj_identity.shape}")
    print(f"  Diagonal ones: {np.diag(adj_identity).sum()}")
    print(f"  Off-diagonal zeros: {(adj_identity == 0).sum() - 30}")

    # Compare sparsity
    print("\n" + "="*70)
    print("Sparsity Comparison")
    print("="*70)
    pearson_density = (adj_pearson > 0).sum() / (30 * 30)
    glasso_density = (adj_glasso > 0).sum() / (30 * 30)
    identity_density = (adj_identity > 0).sum() / (30 * 30)

    print(f"  Pearson: {pearson_density:.2%} non-zero")
    print(f"  GLASSO:  {glasso_density:.2%} non-zero")
    print(f"  Identity: {identity_density:.2%} non-zero")

    print("\ngraph_builder.py OK.")
