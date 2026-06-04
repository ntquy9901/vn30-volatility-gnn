"""
Unit tests for evaluation metrics.

Tests:
- R² computation and range
- QLIKE metric properties
- HMSE, HMAE metrics
- Metric finiteness
- Statistical tests

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
import torch
from evaluation import (
    compute_metrics,
    compute_qlike,
    compute_hmse,
    compute_hmae,
    diebold_mariano_test,
    mincer_zarnowitz_regression,
)


def test_r2_properties():
    """Test R² metric properties."""
    print("\n[TEST] R² properties...")

    # Perfect prediction
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y_true, y_pred, include_qlike=False, include_hetero=False)

    assert metrics['r2'] == 1.0, f"Perfect prediction: R² should be 1.0, got {metrics['r2']}"
    print(f"  [OK] Perfect prediction: R² = {metrics['r2']:.4f}")

    # Same as mean
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # Mean of [1,2,3]
    metrics = compute_metrics(y_true, y_pred, include_qlike=False, include_hetero=False)

    assert abs(metrics['r2']) < 1e-6, f"Mean prediction: R² should be ~0, got {metrics['r2']}"
    print(f"  [OK] Mean prediction: R² = {metrics['r2']:.6f}")

    # Worse than mean
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    metrics = compute_metrics(y_true, y_pred, include_qlike=False, include_hetero=False)

    assert metrics['r2'] < 0, f"Worse than mean: R² should be negative, got {metrics['r2']}"
    print(f"  [OK] Worse than mean: R² = {metrics['r2']:.4f}")

    # Random predictions
    np.random.seed(42)
    y_true = np.random.gamma(2, 0.001, 1000)
    y_pred = y_true + np.random.randn(1000) * 0.0001
    metrics = compute_metrics(y_true, y_pred, include_qlike=False, include_hetero=False)

    assert -1 < metrics['r2'] < 1, f"Random: R² should be in [-1, 1], got {metrics['r2']}"
    print(f"  [OK] Random predictions: R² = {metrics['r2']:.4f}")


def test_qlike_properties():
    """Test QLIKE metric properties."""
    print("\n[TEST] QLIKE properties...")

    # Perfect prediction
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    qlike = compute_qlike(y_true, y_pred)

    assert abs(qlike) < 1e-6, f"Perfect prediction: QLIKE should be ~0, got {qlike}"
    print(f"  [OK] Perfect prediction: QLIKE = {qlike:.6f}")

    # Underprediction (positive QLIKE)
    y_true = np.array([2.0])
    y_pred = np.array([1.0])
    qlike = compute_qlike(y_true, y_pred)

    assert qlike > 0, f"Underprediction: QLIKE should be positive, got {qlike}"
    print(f"  [OK] Underprediction: QLIKE = {qlike:.6f} (positive)")

    # Overprediction (negative QLIKE)
    y_true = np.array([1.0])
    y_pred = np.array([2.0])
    qlike = compute_qlike(y_true, y_pred)

    assert qlike < 0, f"Overprediction: QLIKE should be negative, got {qlike}"
    print(f"  [OK] Overprediction: QLIKE = {qlike:.6f} (negative)")

    # Asymmetry check
    y_true = np.array([1.0])
    qlike_under = compute_qlike(y_true, np.array([0.8]))
    qlike_over = compute_qlike(y_true, np.array([1.2]))

    assert qlike_under > qlike_over, \
        f"QLIKE should penalize underprediction more: under={qlike_under:.4f}, over={qlike_over:.4f}"

    ratio = qlike_under / abs(qlike_over)
    print(f"  [OK] Asymmetry: underprediction penalized {ratio:.2f}× more")


def test_heteroskedastic_metrics():
    """Test HMSE and HMAE metrics."""
    print("\n[TEST] Heteroskedastic metrics...")

    # Low volatility period
    y_true_low = np.array([0.001, 0.001, 0.001])
    y_pred_low = np.array([0.0012, 0.0008, 0.0011])  # Small errors

    hmse_low = compute_hmse(y_true_low, y_pred_low)
    hmae_low = compute_hmae(y_true_low, y_pred_low)

    # High volatility period
    y_true_high = np.array([0.01, 0.01, 0.01])
    y_pred_high = np.array([0.0102, 0.0098, 0.0101])  # Same absolute errors

    hmse_high = compute_hmse(y_true_high, y_pred_high)
    hmae_high = compute_hmae(y_true_high, y_pred_high)

    # HMSE penalizes low-vol errors more
    assert hmse_low > hmse_high, \
        f"HMSE should penalize low-vol more: low={hmse_low:.6f}, high={hmse_high:.6f}"

    print(f"  [OK] HMSE: low-vol penalty {hmse_low/hmse_high:.2f}× higher")

    # HMAE also penalizes low-vol more (but less than HMSE)
    assert hmae_low > hmae_high, \
        f"HMAE should penalize low-vol more: low={hmae_low:.6f}, high={hmae_high:.6f}"

    print(f"  [OK] HMAE: low-vol penalty {hmae_low/hmae_high:.2f}× higher")


def test_metrics_finiteness():
    """Test all metrics are finite."""
    print("\n[TEST] Metrics finiteness...")

    # Generate realistic RV data
    np.random.seed(42)
    y_true = np.random.gamma(shape=2, scale=0.001, size=1000)
    y_pred = y_true + np.random.randn(1000) * 0.0001

    # Ensure positive
    y_true = np.maximum(y_true, 1e-8)
    y_pred = np.maximum(y_pred, 1e-8)

    # Compute all metrics
    metrics = compute_metrics(y_true, y_pred, include_qlike=True, include_hetero=True)

    # Check all are finite
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise AssertionError(f"{key} is not finite: {value}")

    print(f"  [OK] All metrics finite:")
    for key, value in metrics.items():
        print(f"    {key:8s} = {value:.6f}")


def test_diebold_mariano_test():
    """Test Diebold-Mariano statistical test."""
    print("\n[TEST] Diebold-Mariano test...")

    np.random.seed(42)
    n = 100

    # True values
    actual = np.random.gamma(2, 0.001, n)

    # Model 1: Better predictions
    pred1 = actual + np.random.randn(n) * 0.0001

    # Model 2: Worse predictions
    pred2 = actual + np.random.randn(n) * 0.0002

    # Run DM test
    result = diebold_mariano_test(pred1, pred2, actual, metric='mse')

    # Check result structure
    assert 'statistic' in result, "Missing 'statistic' in DM result"
    assert 'p_value' in result, "Missing 'p_value' in DM result"
    assert 'significant' in result, "Missing 'significant' in DM result"
    assert 'better_model' in result, "Missing 'better_model' in DM result"

    # Check value ranges
    assert 0 <= result['p_value'] <= 1, f"p-value out of range: {result['p_value']}"
    assert isinstance(result['significant'], bool), "significant should be bool"
    assert result['better_model'] in ['pred1', 'pred2', 'none'], \
        f"Invalid better_model: {result['better_model']}"

    print(f"  [OK] DM test:")
    print(f"    Statistic: {result['statistic']:.4f}")
    print(f"    P-value: {result['p_value']:.4f}")
    print(f"    Significant: {result['significant']}")
    print(f"    Better model: {result['better_model']}")


def test_mincer_zarnowitz_regression():
    """Test Mincer-Zarnowitz forecast optimality test."""
    print("\n[TEST] Mincer-Zarnowitz regression...")

    np.random.seed(42)
    n = 100

    # Good forecasts
    y_true = np.random.gamma(2, 0.001, n)
    y_pred = y_true + np.random.randn(n) * 0.0001

    # Run MZ regression
    result = mincer_zarnowitz_regression(y_true, y_pred)

    # Check result structure
    assert 'alpha' in result, "Missing 'alpha' in MZ result"
    assert 'beta' in result, "Missing 'beta' in MZ result"
    assert 'rz2' in result, "Missing 'rz2' in MZ result"
    assert 'unbiased' in result, "Missing 'unbiased' in MZ result"
    assert 'efficient' in result, "Missing 'efficient' in MZ result"

    # Check value ranges
    assert 0 <= result['rz2'] <= 1, f"R² out of range: {result['rz2']}"
    assert isinstance(result['unbiased'], bool), "unbiased should be bool"
    assert isinstance(result['efficient'], bool), "efficient should be bool"

    # For good forecasts, beta should be close to 1
    assert 0.5 < result['beta'] < 1.5, f"beta should be ~1 for good forecasts: {result['beta']}"

    print(f"  [OK] MZ regression:")
    print(f"    Alpha: {result['alpha']:.6f}")
    print(f"    Beta: {result['beta']:.4f}")
    print(f"    R²: {result['rz2']:.4f}")
    print(f"    Unbiased: {result['unbiased']}")
    print(f"    Efficient: {result['efficient']}")


def test_metric_consistency():
    """Test consistency between different metrics."""
    print("\n[TEST] Metric consistency...")

    np.random.seed(42)
    n = 1000

    y_true = np.random.gamma(2, 0.001, n)

    # Perfect predictions
    y_pred_perfect = y_true.copy()
    metrics_perfect = compute_metrics(y_true, y_pred_perfect, include_qlike=True, include_hetero=True)

    assert metrics_perfect['r2'] == 1.0, "Perfect: R² should be 1.0"
    assert abs(metrics_perfect['qlike']) < 1e-6, "Perfect: QLIKE should be ~0"
    assert metrics_perfect['mae'] < 1e-6, "Perfect: MAE should be ~0"
    print(f"  [OK] Perfect predictions: All metrics optimal")

    # Terrible predictions (constant small value)
    y_pred_terrible = np.full_like(y_true, 1e-8)  # Predict minimum for all
    metrics_terrible = compute_metrics(y_true, y_pred_terrible, include_qlike=True, include_hetero=True)

    assert metrics_terrible['r2'] < 0, "Terrible: R² should be negative"
    assert metrics_terrible['mae'] > metrics_perfect['mae'], "Terrible: MAE should be worse"
    print(f"  [OK] Terrible predictions: R² = {metrics_terrible['r2']:.2f} (negative)")


def run_all_tests():
    """Run all evaluation tests."""
    print("\n" + "="*60)
    print("  EVALUATION METRICS TEST SUITE")
    print("="*60)

    try:
        test_r2_properties()
        test_qlike_properties()
        test_heteroskedastic_metrics()
        test_metrics_finiteness()
        test_diebold_mariano_test()
        test_mincer_zarnowitz_regression()
        test_metric_consistency()

        print("\n" + "="*60)
        print("  ALL EVALUATION TESTS PASSED!")
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
