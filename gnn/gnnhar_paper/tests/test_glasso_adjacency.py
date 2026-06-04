"""
Unit tests for GLASSO adjacency construction.

Tests:
- GLASSO graph properties
- No self-loops
- Symmetry
- Sparsity
- NaN handling

Date: 2026-05-30
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now we can import directly
import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
from glasso_adjacency import glasso_adjacency


def generate_test_returns(n_stocks=30, n_days=1000, seed=42):
    """Generate synthetic returns data for testing."""
    np.random.seed(seed)

    # Generate correlation matrix
    corr = make_spd_matrix(n_stocks, random_state=seed)

    # Normalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    # Ensure diagonal is 1
    np.fill_diagonal(corr, 1.0)

    # Generate returns with correlation structure
    volatilities = np.random.uniform(0.01, 0.03, n_stocks)  # 1-3% daily vol
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_stocks),
        cov=np.outer(volatilities, volatilities) * corr,
        size=n_days
    )

    # Convert to DataFrame
    tickers = [f'STOCK{i:02d}' for i in range(n_stocks)]
    dates = pd.date_range('2014-01-01', periods=n_days, freq='D')
    return pd.DataFrame(returns, index=dates, columns=tickers)


def test_glasso_no_self_loops():
    """Verify GLASSO excludes self-loops from diagonal."""
    print("\n[TEST] GLASSO: No self-loops...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)
    adj = glasso_adjacency(returns, verbose=False)

    # Check diagonal is zero
    diag_sum = np.diag(adj).sum()
    assert diag_sum == 0, f"Diagonal should be zero, got {diag_sum}"

    print(f"  [OK] Diagonal sum: {diag_sum} (no self-loops)")


def test_glasso_symmetry():
    """Verify adjacency matrix is symmetric."""
    print("\n[TEST] GLASSO: Symmetry...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)
    adj = glasso_adjacency(returns, verbose=False)

    # Check symmetry
    is_symmetric = np.allclose(adj, adj.T, atol=1e-6)
    assert is_symmetric, "Adjacency matrix is not symmetric"

    # Compute max asymmetry (convert DataFrame to numpy first)
    max_asymmetry = float(np.abs(adj.values - adj.T.values).max())
    print(f"  [OK] Max asymmetry: {max_asymmetry:.2e} (symmetric)")


def test_glasso_sparsity():
    """Verify GLASSO produces sparse graph."""
    print("\n[TEST] GLASSO: Sparsity...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)
    # Use lower alpha_range for synthetic data to ensure some edges
    adj = glasso_adjacency(returns, alpha_range=(1e-4, 0.1), verbose=False)

    # Compute density
    n = adj.shape[0]
    n_edges = (adj != 0).sum().sum()  # DataFrame: need double sum for total
    density = n_edges / (n * n)

    # Should be sparse (5-50% density)
    assert 0.05 < density < 0.5, \
        f"Graph density {density:.2%} outside reasonable range [5%, 50%]"

    print(f"  [OK] Edges: {n_edges}/{n*n} ({density:.2%} sparse)")


def test_glasso_nan_handling():
    """Verify GLASSO handles NaN values correctly."""
    print("\n[TEST] GLASSO: NaN handling...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)

    # Add some NaN values
    returns.iloc[10:15, 5] = np.nan
    returns.iloc[100:105, 10] = np.nan

    n_total = len(returns)
    n_nan = returns.isna().any(axis=1).sum()
    n_valid = n_total - n_nan

    # Should not raise error
    try:
        adj = glasso_adjacency(returns, verbose=True)
        print(f"  [OK] Dropped {n_nan} rows with NaN ({n_valid} rows used)")
    except ValueError as e:
        if "contains NaN" in str(e):
            raise AssertionError("GLASSO did not handle NaN values")
        else:
            raise


def test_glasso_minimum_samples():
    """Verify GLASSO requires minimum samples."""
    print("\n[TEST] GLASSO: Minimum samples...")

    # Too few samples
    returns = generate_test_returns(n_stocks=30, n_days=40)  # Only 40 days
    returns = returns.dropna()  # Remove NaN

    try:
        adj = glasso_adjacency(returns, verbose=False)
        # If succeeded, check if it has enough data
        if len(returns) < 30 * 2:
            raise AssertionError("Should have failed with insufficient data")
    except ValueError as e:
        if "Insufficient data" in str(e) or "needs at least" in str(e):
            print(f"  [OK] Correctly rejects insufficient data ({len(returns)} rows)")
        else:
            raise


def test_glasso_positive_edges():
    """Verify all edge weights are positive."""
    print("\n[TEST] GLASSO: Positive edges...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)
    # Use lower alpha_range for synthetic data to ensure some edges
    adj = glasso_adjacency(returns, alpha_range=(1e-4, 0.1), verbose=False)

    # Get non-zero values properly (mask and drop NaN)
    mask = adj.values != 0
    non_zero_values = adj.values[mask]

    # Handle case where graph might be empty
    if len(non_zero_values) > 0:
        assert (non_zero_values >= 0).all(), \
            f"Graph has negative edge weights: min={non_zero_values.min()}"
        print(f"  [OK] All {len(non_zero_values)} edges have non-negative weights")
    else:
        print(f"  [WARN] No edges found (graph is empty)")


def test_glasso_connectivity():
    """Verify graph is reasonably connected."""
    print("\n[TEST] GLASSO: Connectivity...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)
    adj = glasso_adjacency(returns, alpha_range=(1e-4, 0.1), verbose=False)

    # Count isolated nodes (no edges)
    row_sums = adj.sum(axis=1)
    col_sums = adj.sum(axis=0)
    isolated = ((row_sums + col_sums) == 0).sum()

    # Should have minimal isolated nodes
    assert isolated <= 5, f"Too many isolated nodes: {isolated}/30"

    print(f"  [OK] Isolated nodes: {isolated}/30")


def test_glasso_reproducibility():
    """Verify GLASSO produces reproducible results with same seed."""
    print("\n[TEST] GLASSO: Reproducibility...")

    returns = generate_test_returns(n_stocks=30, n_days=1000, seed=42)

    # Run twice with same data
    adj1 = glasso_adjacency(returns, alpha_range=(1e-4, 0.1), verbose=False)
    adj2 = glasso_adjacency(returns, alpha_range=(1e-4, 0.1), verbose=False)

    # Should be identical
    assert np.allclose(adj1, adj2), "GLASSO not reproducible with same data"

    print(f"  [OK] Reproducible: same adjacency from same data")


def test_glasso_different_alpha():
    """Verify GLASSO responds to different alpha ranges."""
    print("\n[TEST] GLASSO: Different alpha ranges...")

    returns = generate_test_returns(n_stocks=30, n_days=1000)

    # Low alpha (less sparse)
    adj_low_alpha = glasso_adjacency(returns, alpha_range=(1e-5, 1e-2), verbose=False)

    # High alpha (more sparse)
    adj_high_alpha = glasso_adjacency(returns, alpha_range=(1e-2, 1.0), verbose=False)

    # High alpha should produce sparser graph
    density_low = (adj_low_alpha != 0).sum().sum() / (30 * 30)  # DataFrame: double sum
    density_high = (adj_high_alpha != 0).sum().sum() / (30 * 30)  # DataFrame: double sum

    assert density_low > density_high, \
        f"Low alpha should be less sparse: low={density_low:.3f}, high={density_high:.3f}"

    print(f"  [OK] Alpha controls sparsity:")
    print(f"    Low alpha: {density_low:.2%} density")
    print(f"    High alpha: {density_high:.2%} density")


def run_all_tests():
    """Run all GLASSO tests."""
    print("\n" + "="*60)
    print("  GLASSO ADJACENCY TEST SUITE")
    print("="*60)

    try:
        test_glasso_no_self_loops()
        test_glasso_symmetry()
        test_glasso_sparsity()
        test_glasso_nan_handling()
        test_glasso_minimum_samples()
        test_glasso_positive_edges()
        test_glasso_connectivity()
        test_glasso_reproducibility()
        test_glasso_different_alpha()

        print("\n" + "="*60)
        print("  ALL GLASSO TESTS PASSED!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
