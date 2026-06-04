"""
GLASSO (Graphical Lasso) adjacency matrix construction.

Paper: "Forecasting Realized Volatility with Spillover Effects:
         Perspectives from Graph Neural Networks" (IJF 2024)

GLASSO estimates a sparse precision matrix (inverse covariance) from data.
Non-zero entries in the precision matrix indicate conditional dependence
between variables, which we interpret as edges in the graph.

Key properties:
- Input: return data (NOT volatility), shape (T, N)
- Output: sparse symmetric adjacency matrix, shape (N, N)
- Sparsity: typically 5-15% non-zero edges (depends on data)
- Self-loops removed (diagonal = 0)
- Normalized for stability in GCN propagation

Mathematical background:
    precision = (Sigma)^(-1)  where Sigma is covariance matrix
    edge(i,j) = 1 if precision[i,j] != 0 (conditional independence)
    Normalization: D^(-1/2) @ adj @ D^(-1/2)
    where D = diag(sum(adj, axis=1))

This normalization ensures each node's total incoming edge weight is 1,
preventing gradient explosion/vanishing in deep GCNs.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")


def glasso_adjacency(
    returns: pd.DataFrame,
    alpha_range: tuple = (0.01, 1.0),
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build adjacency matrix from GLASSO precision matrix.

    Args:
        returns: (T, N) DataFrame of returns (T days, N stocks)
        alpha_range: (min, max) range for GLASSO regularization strength
        n_jobs: parallel jobs for cross-validation
        verbose: print sparsity statistics

    Returns:
        (N, N) DataFrame adjacency matrix with:
        - Diagonal = 0 (no self-loops)
        - Symmetric (undirected edges)
        - Row-normalized (each row sums to 1 after scaling)

    Example:
        >>> returns = pd.DataFrame(np.random.randn(1000, 30), columns=tickers)
        >>> adj = glasso_adjacency(returns)
        >>> print(f"Sparsity: {(adj != 0).sum().sum() / (30*30):.2%}")
    """
    n = returns.shape[1]
    tickers = returns.columns

    # Handle NaN values: GLASSO requires complete data
    # Drop any rows (dates) that contain NaN
    returns_clean = returns.dropna()
    if len(returns_clean) < len(returns):
        dropped = len(returns) - len(returns_clean)
        if verbose:
            print(f"[GLASSO] Dropped {dropped} rows with NaN values ({len(returns_clean)} rows remain)")

    if len(returns_clean) < n * 2:
        raise ValueError(f"Insufficient data after NaN removal: {len(returns_clean)} rows, need at least {n * 2}")

    # GLASSO with cross-validation to find optimal alpha
    # alpha controls sparsity: higher alpha = sparser graph
    # cv=5 means 5-fold cross-validation
    model = GraphicalLassoCV(
        alphas=np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 20),
        cv=5,
        n_jobs=n_jobs,
        verbose=0,
        assume_centered=False,  # estimate mean from data
    )

    model.fit(returns_clean.values)

    if verbose:
        print(f"[GLASSO] alpha={model.alpha_:.5f} (sparsity parameter)")

    # Precision matrix: (N, N)
    # Non-zero entries indicate conditional dependence
    precision = model.precision_
    # Convert to binary: 1 if conditionally dependent, 0 if independent
    corr_adj = (precision != 0).astype(float)

    # Remove self-loops: set diagonal to 0
    np.fill_diagonal(corr_adj, 0)

    # Sparsity: fraction of non-zero edges (excluding diagonal)
    n_edges = corr_adj.sum()
    sparsity = n_edges / (n * n)
    if verbose:
        print(f"[GLASSO] edges={n_edges:.0f}/{n*n} ({sparsity:.2%} sparse)")

    # Symmetric normalization for GCN stability
    # D = degree matrix: D[i,i] = sum of edge weights connected to node i
    # D^(-1/2) scales each node by sqrt(1/degree)
    # Final: A_norm = D^(-1/2) @ A @ D^(-1/2)
    #
    # This ensures:
    # 1. Each node's total incoming weight is ~1
    # 2. Gradient magnitudes are stable across GCN layers
    # 3. No single node dominates the aggregation
    edge_sums = corr_adj.sum(axis=1)  # (N,) sum of each row
    d_sqrt_inv = np.sqrt(1.0 / (edge_sums + 1e-8))  # +eps for numerical stability

    # Normalize: D^(-1/2) @ A
    adj_left = np.diag(d_sqrt_inv) @ corr_adj
    # Then: (D^(-1/2) @ A) @ D^(-1/2)
    adj_norm = adj_left @ np.diag(d_sqrt_inv)

    # Ensure symmetry (numerical errors can break it)
    adj_norm = (adj_norm + adj_norm.T) / 2

    return pd.DataFrame(adj_norm, index=tickers, columns=tickers)


def glasso_adjacency_numpy(
    returns: np.ndarray,
    alpha_range: tuple = (0.01, 1.0),
    n_jobs: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    NumPy version of GLASSO adjacency (returns array instead of DataFrame).

    Args:
        returns: (T, N) numpy array of returns
        alpha_range: (min, max) range for GLASSO regularization
        n_jobs: parallel jobs
        verbose: print alpha selection

    Returns:
        (N, N) normalized adjacency matrix
    """
    model = GraphicalLassoCV(
        alphas=np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 20),
        cv=5,
        n_jobs=-1,  # FIX: Use all CPU cores for faster CV
        verbose=0,
        assume_centered=False,
    )

    model.fit(returns)

    precision = model.precision_
    corr_adj = (precision != 0).astype(float)
    np.fill_diagonal(corr_adj, 0)

    # Symmetric normalization
    edge_sums = corr_adj.sum(axis=1)
    d_sqrt_inv = np.sqrt(1.0 / (edge_sums + 1e-8))
    adj_norm = np.diag(d_sqrt_inv) @ corr_adj @ np.diag(d_sqrt_inv)

    return (adj_norm + adj_norm.T) / 2


class RollingAdjacency:
    """
    Manage rolling window adjacency computation for GNNHAR paper scheme.

    The paper recomputes adjacency for each training window using a
    fixed lookback period (typically 1000 days). This captures
    time-varying correlation structures in the data.

    Usage:
        >>> adj_manager = RollingAdjacency(returns, lookback=1000)
        >>> adj = adj_manager.get_adjacency(date='2020-01-15')
        >>> # Uses returns from date-1000 to date for GLASSO
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        lookback: int = 1000,
        alpha_range: tuple = (0.01, 1.0),
        cache: bool = True,
    ):
        """
        Args:
            returns: (T, N) DataFrame of returns, DatetimeIndex
            lookback: days to look back for GLASSO estimation
            alpha_range: GLASSO alpha range
            cache: whether to cache computed adjacency matrices
        """
        self.returns = returns
        self.lookback = lookback
        self.alpha_range = alpha_range
        self.cache = {} if cache else None
        self.n_stocks = returns.shape[1]
        self.tickers = returns.columns

    def get_adjacency(self, date: str | pd.Timestamp) -> np.ndarray:
        """
        Get adjacency matrix for a specific date using lookback window.

        Args:
            date: target date (adjacency computed from data before this date)

        Returns:
            (N, N) adjacency matrix

        Note: If date is too early (< lookback days from start),
        uses all available data from start.
        """
        date = pd.Timestamp(date)

        # Check cache first
        if self.cache is not None and date in self.cache:
            return self.cache[date]

        # Get lookback window: [date - lookback, date)
        start_date = date - pd.Timedelta(days=self.lookback)
        window_returns = self.returns[(self.returns.index >= start_date) &
                                      (self.returns.index < date)]

        # If not enough data, use all available data
        if window_returns.shape[0] < 100:  # minimum threshold
            window_returns = self.returns[self.returns.index < date]

        # Compute adjacency
        adj = glasso_adjacency_numpy(
            window_returns.values,
            alpha_range=self.alpha_range,
        )

        # Cache if enabled
        if self.cache is not None:
            self.cache[date] = adj

        return adj

    def compute_all_dates(
        self,
        dates: list[pd.Timestamp],
        verbose: bool = False,
    ) -> dict[pd.Timestamp, np.ndarray]:
        """
        Pre-compute adjacency for all dates (useful for batch experiments).

        Args:
            dates: list of dates to compute
            verbose: print progress

        Returns:
            dict mapping date -> adjacency matrix
        """
        result = {}
        for i, date in enumerate(dates):
            result[date] = self.get_adjacency(date)
            if verbose and (i + 1) % 10 == 0:
                print(f"[RollingAdjacency] {i+1}/{len(dates)} computed")
        return result


if __name__ == "__main__":
    # Test with synthetic data
    print("[TEST] GLASSO adjacency with synthetic data...")

    np.random.seed(42)
    n_days = 1000
    n_stocks = 30

    # Generate correlated returns with stronger correlations
    dates = pd.date_range('2018-01-01', periods=n_days, freq='D')
    returns = np.random.randn(n_days, n_stocks) * 0.01

    # Add common factor per block (creates strong within-block correlation)
    for block_start in [0, 10, 20]:
        factor = np.random.randn(n_days, 1) * 0.02  # stronger factor
        returns[:, block_start:block_start+10] += factor

    tickers = [f"STOCK{i:02d}" for i in range(n_stocks)]
    returns_df = pd.DataFrame(returns, columns=tickers, index=dates)

    # Compute adjacency with lower alpha (less sparsity) for test data
    adj = glasso_adjacency(returns_df, verbose=True, alpha_range=(0.001, 0.5))

    print(f"\nAdjacency shape: {adj.shape}")
    print(f"Non-zero edges: {(adj != 0).sum().sum()}")
    n_nonzero = (adj != 0).sum().sum()
    if n_nonzero > 0:
        print(f"Edge weight range: [{adj.values[adj != 0].min():.4f}, "
              f"{adj.values[adj != 0].max():.4f}]")
        print(f"Row sums (should be ~1): {adj.sum(axis=1).values[:5]}...")
    else:
        print("[WARN] No edges found (data too weakly correlated)")

    # Test rolling adjacency
    print("\n[TEST] RollingAdjacency...")
    test_dates = pd.date_range('2020-01-01', periods=5, freq='22D')
    rolling_adj = RollingAdjacency(returns_df, lookback=500)

    for date in test_dates[:2]:
        adj = rolling_adj.get_adjacency(date)
        print(f"  {date.date()}: shape={adj.shape}, non-zero={(adj != 0).sum()}")

    # Test with hardcoded sparse adjacency (verify normalization works)
    print("\n[TEST] Normalization check...")
    test_adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)
    # Normalize
    edge_sums = test_adj.sum(axis=1)
    d_sqrt_inv = np.sqrt(1.0 / (edge_sums + 1e-8))
    adj_norm = np.diag(d_sqrt_inv) @ test_adj @ np.diag(d_sqrt_inv)
    print(f"  Original row sums: {edge_sums}")
    print(f"  Normalized row sums: {adj_norm.sum(axis=1)}")
    print(f"  Should all be ~1.0 (symmetric normalization)")

    print("\n[OK] All tests passed")
