"""
Improved VIC HAR training with:
1. Stride = 1 for ALL sets (5x more data)
2. Rolling normalization (RV_ratio = RV_d / RV_m)
3. Walk-forward validation

Expected: R2 >= HAR baseline = +0.55
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent.parent  # This is moirai folder
MOIRAI_ROOT = _ROOT  # Already moirai
sys.path.insert(0, str(MOIRAI_ROOT))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from baselines.har_rv_baseline import fit_har, predict_har

warnings.filterwarnings("ignore")

print("\n" + "="*70)
print("  IMPROVED VIC HAR TRAINING")
print("  1. Stride = 1 (ALL sets)")
print("  2. Rolling normalization")
print("  3. Walk-forward validation")
print("="*70 + "\n")

# Config
HORIZON = 5
TICKER = 'VIC'
GLOBAL_TEST_START = "2026-01-01"
N_EPOCHS = 1000
LR = 1e-3
PATIENCE = 200  # Increased for 1000 epochs
BATCH_SIZE = 32

# Load data
import yaml
with open(MOIRAI_ROOT / 'config.yaml') as f:
    _CFG = yaml.safe_load(f)
DATA_DIR = MOIRAI_ROOT / _CFG['data']['prices_dir']

print("[Data] Loading VIC prices...")
close = load_close_prices(DATA_DIR, tickers=[TICKER])
print(f"  Shape: {close.shape}")

# Compute RV
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()
print(f"  RV (h={HORIZON}): {len(rv)} samples")

# =============================================================================
# BUILD SNAPSHOTS WITH STRIDE = 1
# =============================================================================

def build_snapshots_stride1(rv_series, horizon=5):
    """
    Build HAR snapshots with STRIDE = 1.
    Returns: X (n, 3), y (n,), dates
    """
    # Need at least 22 days for monthly window + horizon days ahead
    min_history = 22 + horizon

    X_list = []
    y_list = []
    date_list = []

    for i in range(min_history, len(rv_series) - horizon):
        # Get current date
        current_date = rv_series.index[i]

        # Target: RV from i to i+horizon
        target = rv_series.iloc[i:i+horizon].mean()

        # Features: look back from day i
        # RV_d: 1 day back (i-1 to i)
        rv_d = rv_series.iloc[i-1:i].mean()

        # RV_w: 5 days back (i-5 to i)
        rv_w = rv_series.iloc[i-5:i].mean()

        # RV_m: 22 days back (i-22 to i)
        rv_m = rv_series.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)
        date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

print(f"\n[Build] Building snapshots with STRIDE = 1...")
X, y, dates = build_snapshots_stride1(rv, HORIZON)
print(f"  Total snapshots: {len(X)} (vs {len(rv)//5} with stride=5)")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# =============================================================================
# SPLIT DATA
# =============================================================================

test_ts = pd.Timestamp(GLOBAL_TEST_START)
pre_test_mask = dates < test_ts

X_pre = X[pre_test_mask]
y_pre = y[pre_test_mask]
dates_pre = dates[pre_test_mask]

# 80/20 split for train/val
n_train = int(len(X_pre) * 0.8)

X_train = X_pre[:n_train]
y_train = y_pre[:n_train]
dates_train = dates_pre[:n_train]

X_val = X_pre[n_train:]
y_val = y_pre[n_train:]
dates_val = dates_pre[n_train:]

# Test set
test_mask = dates >= test_ts
X_test = X[test_mask]
y_test = y[test_mask]
dates_test = dates[test_mask]

print(f"\n[Split] With STRIDE = 1:")
print(f"  Train: {len(X_train)} snapshots ({dates_train[0].date()} to {dates_train[-1].date()})")
print(f"  Val:   {len(X_val)} snapshots ({dates_val[0].date()} to {dates_val[-1].date()})")
print(f"  Test:  {len(X_test)} snapshots ({dates_test[0].date()} to {dates_test[0].date()})")

# =============================================================================
# METHOD 1: RAW HAR (BASELINE)
# =============================================================================

print(f"\n{'='*70}")
print(f"  METHOD 1: RAW HAR (Linear regression on raw RV)")
print(f"{'='*70}")

class HAR_VIC(nn.Module):
    """Simple HAR model for VIC."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.linear(x)
        h = self.relu(h)
        return h.squeeze(-1)

def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - (ss_res / ss_tot)

# Train HAR on raw RV
print("\n[Train] Training HAR on raw RV...")
model_raw = HAR_VIC()
optimizer = optim.Adam(model_raw.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
patience_cnt = 0

train_losses = []
val_losses = []

for epoch in range(N_EPOCHS):
    # Train
    model_raw.train()
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()

    pred = model_raw(X_t)
    loss = criterion(pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Val
    model_raw.eval()
    with torch.no_grad():
        X_v = torch.from_numpy(X_val).float()
        y_v = torch.from_numpy(y_val).float()
        val_pred = model_raw(X_v)
        val_loss = criterion(val_pred, y_v).item()

    val_losses.append(val_loss)

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model_raw.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# Load best model
model_raw.load_state_dict(best_state)

# Evaluate
model_raw.eval()
with torch.no_grad():
    X_ts = torch.from_numpy(X_test).float()
    raw_pred = model_raw(X_ts).numpy()

raw_r2 = compute_r2(y_test, raw_pred)
raw_mae = np.mean(np.abs(y_test - raw_pred))

print(f"\n[Results] RAW HAR:")
print(f"  R² = {raw_r2:+.4f}")
print(f"  MAE = {raw_mae:.6f}")
print(f"  Train mean: {y_train.mean():.6f}")
print(f"  Test mean:  {y_test.mean():.6f}")
print(f"  Pred mean:   {raw_pred.mean():.6f}")

# =============================================================================
# METHOD 2: ROLLING NORMALIZATION
# =============================================================================

print(f"\n{'='*70}")
print(f"  METHOD 2: ROLLING NORMALIZATION (RV / RV_m)")
print(f"{'='*70}")

def normalize_by_rolling(X, y, window_size=22):
    """
    Normalize by rolling mean to handle distribution shift.

    Input: X (n, 3) = [RV_d, RV_w, RV_m]
    Output: X_norm (n, 3) = [RV_d/RV_m, RV_w/RV_m, 1]
             y_norm (n,) = y / RV_m
    """
    X_norm = X.copy()
    y_norm = y.copy()

    for i in range(len(X)):
        # Get RV_m (3rd column) as baseline
        rv_m = X[i, 2]

        if rv_m > 1e-8:  # Avoid division by zero
            # Normalize features
            X_norm[i, 0] = X[i, 0] / rv_m  # RV_d / RV_m
            X_norm[i, 1] = X[i, 1] / rv_m  # RV_w / RV_m
            X_norm[i, 2] = 1.0              # Normalized baseline

            # Normalize target
            y_norm[i] = y[i] / rv_m
        else:
            # Keep original if RV_m is too small
            X_norm[i] = X[i]
            y_norm[i] = y[i]

    return X_norm, y_norm

# Normalize train/val/test
X_train_norm, y_train_norm = normalize_by_rolling(X_train, y_train)
X_val_norm, y_val_norm = normalize_by_rolling(X_val, y_val)
X_test_norm, y_test_norm = normalize_by_rolling(X_test, y_test)

print("\n[Normalize] Applied rolling normalization:")
print(f"  Train X range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
print(f"  Train y range: [{y_train_norm.min():.4f}, {y_train_norm.max():.4f}]")

# Train on normalized data
print("\n[Train] Training HAR on normalized data...")
model_norm = HAR_VIC()
optimizer = optim.Adam(model_norm.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
patience_cnt = 0

for epoch in range(N_EPOCHS):
    # Train
    model_norm.train()
    X_t = torch.from_numpy(X_train_norm).float()
    y_t = torch.from_numpy(y_train_norm).float()

    pred = model_norm(X_t)
    loss = criterion(pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Val
    model_norm.eval()
    with torch.no_grad():
        X_v = torch.from_numpy(X_val_norm).float()
        y_v = torch.from_numpy(y_val_norm).float()
        val_pred = model_norm(X_v)
        val_loss = criterion(val_pred, y_v).item()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model_norm.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# Load best model
model_norm.load_state_dict(best_state)

# Inverse transform predictions
model_norm.eval()
with torch.no_grad():
    X_ts = torch.from_numpy(X_test_norm).float()
    norm_pred = model_norm(X_ts).numpy()

# Inverse normalize: pred_raw = pred_norm * RV_m
norm_pred_raw = norm_pred * X_test[:, 2]  # Multiply by RV_m baseline

norm_r2 = compute_r2(y_test, norm_pred_raw)
norm_mae = np.mean(np.abs(y_test - norm_pred_raw))

print(f"\n[Results] NORMALIZED HAR:")
print(f"  R² = {norm_r2:+.4f}")
print(f"  MAE = {norm_mae:.6f}")

# =============================================================================
# METHOD 3: WALK-FORWARD VALIDATION
# =============================================================================

print(f"\n{'='*70}")
print(f"  METHOD 3: WALK-FORWARD VALIDATION (Recent 1000 days)")
print(f"{'='*70}")

# Use last 1000 days of pre-test data for training
WALK_FORWARD_WINDOW = 1000

X_train_wf = X_pre[-WALK_FORWARD_WINDOW:]
y_train_wf = y_pre[-WALK_FORWARD_WINDOW:]
dates_train_wf = dates_pre[-WALK_FORWARD_WINDOW:]

print(f"\n[Walk-Forward] Training on last {WALK_FORWARD_WINDOW} days:")
print(f"  From: {dates_train_wf[0].date()}")
print(f"  To:   {dates_train_wf[-1].date()}")
print(f"  Mean: {y_train_wf.mean():.6f}")

print(f"\n[Test] Test period:")
print(f"  From: {dates_test[0].date()}")
print(f"  Mean: {y_test.mean():.6f}")

# Train on recent data
print("\n[Train] Training HAR on recent data...")
model_wf = HAR_VIC()
optimizer = optim.Adam(model_wf.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
patience_cnt = 0

# Use last 20% of walk-forward window for validation
n_train_wf = int(len(X_train_wf) * 0.8)
X_train_wf_train = X_train_wf[:n_train_wf]
y_train_wf_train = y_train_wf[:n_train_wf]
X_train_wf_val = X_train_wf[n_train_wf:]
y_train_wf_val = y_train_wf[n_train_wf:]

for epoch in range(N_EPOCHS):
    # Train
    model_wf.train()
    X_t = torch.from_numpy(X_train_wf_train).float()
    y_t = torch.from_numpy(y_train_wf_train).float()

    pred = model_wf(X_t)
    loss = criterion(pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Val
    model_wf.eval()
    with torch.no_grad():
        X_v = torch.from_numpy(X_train_wf_val).float()
        y_v = torch.from_numpy(y_train_wf_val).float()
        val_pred = model_wf(X_v)
        val_loss = criterion(val_pred, y_v).item()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model_wf.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# Load best model
model_wf.load_state_dict(best_state)

# Evaluate
model_wf.eval()
with torch.no_grad():
    X_ts = torch.from_numpy(X_test).float()
    wf_pred = model_wf(X_ts).numpy()

wf_r2 = compute_r2(y_test, wf_pred)
wf_mae = np.mean(np.abs(y_test - wf_pred))

print(f"\n[Results] WALK-FORWARD HAR:")
print(f"  R² = {wf_r2:+.4f}")
print(f"  MAE = {wf_mae:.6f}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"  SUMMARY COMPARISON")
print(f"{'='*70}")
print(f"{'Method':<25} {'R2':>10} {'MAE':>12}")
print(f"{'-'*50}")
print(f"{'Raw HAR (stride=5, baseline)':25} {-2.46:>+10.4f} {0.0187:12.6f}")
print(f"{'Raw HAR (stride=1)':25} {raw_r2:>+10.4f} {raw_mae:12.6f}")
print(f"{'Normalized HAR':25} {norm_r2:>+10.4f} {norm_mae:12.6f}")
print(f"{'Walk-Forward HAR':25} {wf_r2:>+10.4f} {wf_mae:12.6f}")

# Compare with OLS baseline
har_coeffs = fit_har(rv, dates_train[-1])  # Use train end date
har_pred_ols = predict_har(rv, har_coeffs, test_ts)
har_pred_aligned = har_pred_ols.reindex(dates_test)
har_pred_valid = har_pred_aligned.dropna()
y_test_aligned = y_test[:len(har_pred_valid)]

if len(har_pred_valid) > 0:
    har_r2 = compute_r2(y_test_aligned, har_pred_valid)
    print(f"{'HAR OLS (baseline)':25} {har_r2:>+10.4f}")

print(f"\n{'='*70}")

# Save results
results = {
    'stride1_raw_r2': raw_r2,
    'stride1_raw_mae': raw_mae,
    'normalized_r2': norm_r2,
    'normalized_mae': norm_mae,
    'walk_forward_r2': wf_r2,
    'walk_forward_mae': wf_mae,
}

import json
output_dir = MOIRAI_ROOT / 'results' / 'gnnhar_paper' / 'analysis'
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'vic_improved_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] Results saved to {output_dir / 'vic_improved_results.json'}\n")
