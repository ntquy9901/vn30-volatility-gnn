"""
Quick sanity test for GNNHAR training (BMAD: Build-Measure-Assess-Dev).

Purpose: Catch training bugs BEFORE running full 5000 epochs.
Tests: 1 epoch, 1 model, 1 horizon. Check if R² is reasonable.

Usage:
    python test_sanity.py
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.train_gnnhar_paper import build_static_adjacency

# CONFIG (minimal for speed)
TEST_HORIZON = 5
TEST_N_EPOCHS = 2
N_STOCKS = len(VN30_TICKERS)

print("="*60)
print("  QUICK SANITY TEST (BMAD)")
print("="*60)
print(f"  Horizon: {TEST_HORIZON}")
print(f"  Epochs: {TEST_N_EPOCHS}")
print(f"  Stocks: {N_STOCKS}")
print("="*60)

# Load data
print("\n[1] Loading data...")
_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / 'data/raw/prices'
close = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
log_ret = compute_log_returns(close)
print(f"  Data shape: {close.shape}")

# Build features/target for test horizon
print(f"\n[2] Building features for h={TEST_HORIZON}...")
from gnn.gnnhar_paper.rolling_datasets import build_static_snapshots

X, y, dates = build_static_snapshots(
    close, log_ret, horizon=TEST_HORIZON, stride=20,
    date_end=pd.Timestamp("2025-12-31"),
)
print(f"  X shape: {X.shape}  (n_snapshots, n_stocks, n_features)")
print(f"  y shape: {y.shape}  (n_snapshots, n_stocks)")

# Simple split (80/20)
n = len(dates)
n_train = int(n * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:], y[n_train:]
train_dates = dates[:n_train]
print(f"  Train: {n_train} snapshots | Val: {len(dates) - n_train} snapshots")

# Build adjacency
print(f"\n[3] Building adjacency matrix...")
train_end = train_dates[-1]
adj = build_static_adjacency(log_ret, train_end)
n_edges = (adj != 0).sum()
print(f"  Adjacency: {N_STOCKS}x{N_STOCKS}, {n_edges} edges")

# Z-score targets (per stock)
print(f"\n[4] Z-scoring targets...")
y_train_mean = y_train.mean(axis=0)
y_train_std = y_train.std(axis=0)
y_train_std = np.where(y_train_std < 1e-8, 1.0, y_train_std)

y_train_z = (y_train - y_train_mean) / y_train_std
y_val_z = (y_val - y_train_mean) / y_train_std

print(f"  Train mean (raw): {y_train_mean.mean():.6f}")
print(f"  Train std (raw): {y_train_std.mean():.6f}")
print(f"  Train mean (z-scored): {y_train_z.mean():.6f}")
print(f"  Train std (z-scored): {y_train_z.std():.6f}")

# Test each model
print(f"\n[5] Testing models (quick train)...")
results = {}

for model_name in ['HAR', 'GHAR', 'GNNHAR1L']:
    print(f"\n  [{model_name}]")

    # Create model
    model_class = MODEL_REGISTRY[model_name]
    if model_name == 'HAR':
        model = model_class()
    else:
        model = model_class(n_hid=16)

    # Quick training
    device = torch.device('cpu')
    model = model.to(device)

    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train_z).float().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val_z).float().to(device)
    adj_t = torch.from_numpy(adj).float().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.MSELoss()  # Use MSE for quick test

    train_losses = []
    val_losses = []

    for epoch in range(TEST_N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t, adj_t)
        loss = criterion(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v, adj_t)
            val_loss = criterion(val_pred, y_v).item()
            val_losses.append(val_loss)

        print(f"    Epoch {epoch+1}/{TEST_N_EPOCHS}: train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}")

    # Quick prediction check
    model.eval()
    with torch.no_grad():
        pred_z = model(X_v, adj_t).cpu().numpy()

    # Inverse transform to original scale
    pred = pred_z * y_train_std[None, :] + y_train_mean[None, :]

    # Check prediction scale
    print(f"    Pred mean: {pred.mean():.6f}")
    print(f"    Pred std: {pred.std():.6f}")
    print(f"    Pred min/max: {pred.min():.6f} / {pred.max():.6f}")

    # Simple R² check (overall, not per-stock)
    y_val_raw = y_val
    ss_res = ((y_val_raw - pred) ** 2).sum()
    ss_tot = ((y_val_raw - y_val_raw.mean()) ** 2).sum()
    r2_overall = 1 - (ss_res / ss_tot)

    print(f"    Overall R² (val): {r2_overall:.4f}")

    results[model_name] = {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'r2_overall': r2_overall,
        'pred_mean': pred.mean(),
        'pred_std': pred.std(),
    }

# Sanity checks
print(f"\n" + "="*60)
print("  SANITY CHECK RESULTS")
print("="*60)

for model_name, res in results.items():
    print(f"\n{model_name}:")
    print(f"  Train loss: {res['train_loss']:.4f}")
    print(f"  Val loss:   {res['val_loss']:.4f}")
    print(f"  R² (overall): {res['r2_overall']:>8.2f}")
    print(f"  Pred scale: mean={res['pred_mean']:.6f}, std={res['pred_std']:.6f}")

# Critical checks
print(f"\n" + "="*60)
print("  CRITICAL CHECKS")
print("="*60)

all_passed = True

for model_name, res in results.items():
    r2 = res['r2_overall']

    # Check 1: R² should be reasonable for 2 epochs
    if r2 < -100:
        print(f"  [FAIL] {model_name}: R² = {r2:.2f} - CATASTROPHIC (likely bug)")
        all_passed = False
    elif r2 < -10:
        print(f"  [WARN] {model_name}: R² = {r2:.2f} - Very bad (check inverse transform)")
    elif r2 < 0:
        print(f"  [OK]  {model_name}: R² = {r2:.2f} - Negative but reasonable for early training")
    else:
        print(f"  [OK]  {model_name}: R² = {r2:.2f} - Positive")

    # Check 2: Prediction scale should match data scale
    if res['pred_std'] < 1e-6 or res['pred_std'] > 1:
        print(f"  [WARN] {model_name}: Pred std = {res['pred_std']:.6f} - unusual scale")

    # Check 3: Loss should decrease
    if res['val_loss'] > res['train_loss'] * 10:
        print(f"  [WARN] {model_name}: Val loss >> Train loss - possible overfit")

print(f"\n" + "="*60)
if all_passed:
    print("  [PASS] All critical checks passed - safe to train full")
else:
    print("  [FAIL] Critical issues found - fix before full training")
print("="*60)
