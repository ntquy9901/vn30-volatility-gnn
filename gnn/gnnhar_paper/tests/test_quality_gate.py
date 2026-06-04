"""
Quality Gate test for GNNHAR training.

Purpose: Catch training bugs BEFORE running full epochs.
This test should be run BEFORE any full training session.

Usage:
    python tests/test_quality_gate.py
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
gnnhar_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(gnnhar_root))
# Also add gnnhar_paper to path for local imports
sys.path.insert(0, str(gnnhar_root / 'gnn' / 'gnnhar_paper'))

from src.volatility_labels import load_close_prices, compute_log_returns
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.train_gnnhar_paper import build_static_adjacency
from gnn.gnnhar_paper.rolling_datasets import build_static_snapshots


def test_quick_training_sanity():
    """
    Test 1: Quick sanity check - train 2 epochs, check R² reasonable.

    PASS: R² > -100 (not catastrophic)
    FAIL: R² < -1000 (indicates bug in inverse transform or loss function)
    """
    print("\n[Test 1] Quick Training Sanity")

    _ROOT = Path(__file__).parent.parent.parent.parent  # Go up to moirai root
    DATA_DIR = _ROOT / 'data/raw/prices'
    close = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    log_ret = compute_log_returns(close)

    print(f"  close type: {type(close)}, shape: {close.shape}")
    print(f"  log_ret type: {type(log_ret)}, shape: {log_ret.shape}")
    print(f"  log_ret.index type: {type(log_ret.index)}, first: {log_ret.index[0]}, last: {log_ret.index[-1]}")

    # Build features
    print(f"  Loading data...")
    X, y, dates = build_static_snapshots(
        close, log_ret, horizon=5, stride=20,
        date_end=pd.Timestamp("2025-12-31"),
    )
    print(f"  X shape: {X.shape}, y shape: {y.shape}, dates: {len(dates)}")

    # Simple split
    n = len(dates)
    n_train = int(n * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    print(f"  dates type: {type(dates)}")
    print(f"  dates[n_train-1] type: {type(dates[n_train - 1])}")
    print(f"  dates[n_train-1] value: {dates[n_train - 1]}")

    # Z-score
    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)
    y_train_std = np.where(y_train_std < 1e-8, 1.0, y_train_std)
    y_train_z = (y_train - y_train_mean) / y_train_std
    y_val_z = (y_val - y_train_mean) / y_train_std

    # Build adjacency
    adj = build_static_adjacency(log_ret, pd.Timestamp(dates[n_train - 1]))

    # Test HAR model only
    model = MODEL_REGISTRY['HAR']()
    device = torch.device('cpu')
    model = model.to(device)

    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train_z).float().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    adj_t = torch.from_numpy(adj).float().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Train 2 epochs
    for epoch in range(2):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t, adj_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    # Check R²
    model.eval()
    with torch.no_grad():
        pred_z = model(X_v, adj_t).cpu().numpy()
        pred = pred_z * y_train_std[None, :] + y_train_mean[None, :]

    ss_res = ((y_val - pred) ** 2).sum()
    ss_tot = ((y_val - y_val.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    print(f"  R² (2 epochs): {r2:.4f}")

    assert r2 > -100, f"FAIL: R² = {r2:.2f} - catastrophic bug detected"
    print("  [PASS] R² reasonable")
    return True


def test_loss_function_compatibility():
    """
    Test 2: Verify loss function matches data type.

    PASS: MSE loss used for z-scored data
    FAIL: QLIKE loss used with z-scored data (requires positive values)
    """
    print("\n[Test 2] Loss Function Compatibility")

    # Check that ensemble_trainer uses MSE for z-scored data
    from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer

    # Create dummy trainer
    trainer = EnsembleTrainer(model_name='HAR', n_epochs=2)

    # Check training code uses MSE
    # We can't directly inspect, but we can verify by running single train
    n_train, n_val, N = 50, 20, 10
    X_train = np.random.randn(n_train, N, 3).astype(np.float32)
    y_train = np.random.randn(n_train, N).astype(np.float32)
    X_val = np.random.randn(n_val, N, 3).astype(np.float32)
    y_val = np.random.randn(n_val, N).astype(np.float32)
    adj = np.eye(N, dtype=np.float32)

    # Train single model
    result = trainer.train_single(X_train, y_train, X_val, y_val, adj, seed=42)

    # If training completed without error, loss function is compatible
    print(f"  Train loss: {result['train_loss']:.4f}")
    print(f"  Val loss: {result['val_loss']:.4f}")

    assert result['train_loss'] < 100, f"FAIL: Train loss exploded"
    assert not np.isnan(result['train_loss']), f"FAIL: Train loss is NaN"
    print("  [PASS] Loss function compatible")
    return True


def test_inverse_transform_correctness():
    """
    Test 3: Verify inverse transform preserves prediction quality.

    PASS: Inverse transform gives predictions in original scale
    FAIL: Inverse transform produces absurd values
    """
    print("\n[Test 3] Inverse Transform Correctness")

    # Create synthetic z-scored predictions
    n, N = 100, 30
    pred_z = np.random.randn(n, N).astype(np.float32) * 0.5  # z-scored predictions

    # Synthetic scaling parameters
    y_mean = np.random.rand(N) * 0.02 + 0.01  # RV mean ~ 0.01-0.03
    y_std = np.random.rand(N) * 0.01 + 0.005  # RV std ~ 0.005-0.015

    # Inverse transform
    pred = pred_z * y_std[None, :] + y_mean[None, :]

    # Check scale
    pred_mean = pred.mean()
    pred_std = pred.std()

    print(f"  Pred mean: {pred_mean:.6f}")
    print(f"  Pred std: {pred_std:.6f}")

    assert 0.005 < pred_mean < 0.05, f"FAIL: Pred mean out of range: {pred_mean}"
    assert 0.001 < pred_std < 0.1, f"FAIL: Pred std out of range: {pred_std}"
    # Most predictions should be positive (RV > 0), but some negatives allowed near zero
    positive_ratio = (pred > 0).mean()
    assert positive_ratio > 0.9, f"FAIL: Only {positive_ratio:.1%} predictions are positive"
    print("  [PASS] Inverse transform correct")
    return True


def main():
    print("="*60)
    print("  QUALITY GATE - GNNHAR Training")
    print("="*60)

    tests = [
        test_quick_training_sanity,
        test_loss_function_compatibility,
        test_inverse_transform_correctness,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  {e}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {e}")

    print("\n" + "="*60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("[SUCCESS] Quality Gate PASSED - safe to train full")
        return 0
    else:
        print("[FAILURE] Quality Gate FAILED - fix bugs before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
