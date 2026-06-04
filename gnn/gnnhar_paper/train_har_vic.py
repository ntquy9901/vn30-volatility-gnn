"""
Train HAR (nn.Module) for VIC stock with stride=1.
Based on train_vic_improved.py template.

Usage:
    python moirai/gnn/gnnhar_paper/train_har_vic.py

Config:
    Horizon: h=5
    Stride: 1
    Test from: 2026-01-01
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

warnings.filterwarnings("ignore")

# NO random seed - use stochastic optimization (like train_vic_improved.py)

# =============================================================================
# PATH SETUP
# =============================================================================
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.volatility_labels import load_close_prices, compute_rv
from baselines.har_rv_baseline import fit_har, predict_har

print("\n" + "="*70)
print("  HAR (nn.Module) TRAINING FOR VIC")
print("  Stride = 1, h = 5")
print("="*70 + "\n")

# =============================================================================
# CONFIG
# =============================================================================
HORIZON = 5
TICKER = 'VIC'
GLOBAL_TEST_START = "2026-01-01"
N_EPOCHS = 200
LR = 1e-3
PATIENCE = 50

# =============================================================================
# LOAD DATA
# =============================================================================
print("[Data] Loading VIC prices...")
import yaml
with open(_ROOT / 'config.yaml') as f:
    cfg = yaml.safe_load(f)
DATA_DIR = _ROOT / cfg['data']['prices_dir']

close = load_close_prices(DATA_DIR, tickers=[TICKER])
print(f"  Shape: {close.shape}")

# Compute RV
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()
print(f"  RV (h={HORIZON}): {len(rv)} samples")

# =============================================================================
# BUILD SNAPSHOTS WITH STRIDE = 1
# =============================================================================
print(f"\n[Data] Building HAR features with STRIDE = 1...")

def build_snapshots_stride1(rv_series, horizon=5):
    """
    Build HAR snapshots with STRIDE = 1.
    Returns: X (n, 3), y (n,), dates
    """
    min_history = 22 + horizon
    X_list = []
    y_list = []
    date_list = []

    for i in range(min_history, len(rv_series) - horizon):
        current_date = rv_series.index[i]

        # Target: RV from i to i+horizon
        target = rv_series.iloc[i:i+horizon].mean()

        # Features: look back from day i
        rv_d = rv_series.iloc[i-1:i].mean()
        rv_w = rv_series.iloc[i-5:i].mean()
        rv_m = rv_series.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)
        date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

X, y, dates = build_snapshots_stride1(rv, HORIZON)
print(f"  Total snapshots: {len(X)} (stride=1)")
print(f"  X shape: {X.shape}")
print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")

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

print(f"\n[Split]")
print(f"  Train: {len(X_train)} samples ({dates_train[0].date()} to {dates_train[-1].date()})")
print(f"  Val:   {len(X_val)} samples ({dates_val[0].date()} to {dates_val[-1].date()})")
print(f"  Test:  {len(X_test)} samples ({dates_test[0].date()} to {dates_test[-1].date()})")

print(f"\n[Statistics]")
print(f"  Train mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")
print(f"  Val   mean: {y_val.mean():.6f}, std: {y_val.std():.6f}")
print(f"  Test  mean: {y_test.mean():.6f}, std: {y_test.std():.6f}")

# =============================================================================
# HAR MODEL
# =============================================================================
print(f"\n{'='*70}")
print(f"  HAR (nn.Module)")
print(f"{'='*70}")

class HAR(nn.Module):
    """HAR model - matching train_vic_improved.py exactly."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.relu = nn.ReLU()
        # NO custom initialization - use PyTorch defaults

    def forward(self, x):
        h = self.linear(x)
        h = self.relu(h)
        return h.squeeze(-1)

def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def compute_mae(y_true, y_pred):
    """Compute MAE."""
    return np.mean(np.abs(y_true - y_pred))

# Train HAR
print("\n[Train] Training HAR...")
model = HAR()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
patience_cnt = 0

for epoch in range(N_EPOCHS):
    # Train
    model.train()
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()

    pred = model(X_t)
    loss = criterion(pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    # NO gradient clipping - like train_vic_improved.py
    optimizer.step()

    # Val
    model.eval()
    with torch.no_grad():
        X_v = torch.from_numpy(X_val).float()
        y_v = torch.from_numpy(y_val).float()
        val_pred = model(X_v)
        val_loss = criterion(val_pred, y_v).item()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(best_state)

# Evaluate
model.eval()
with torch.no_grad():
    X_ts = torch.from_numpy(X_test).float()
    pred = model(X_ts).numpy()

r2 = compute_r2(y_test, pred)
mae = compute_mae(y_test, pred)
rmse = np.sqrt(np.mean((y_test - pred)**2))

print(f"\n[Results] HAR (nn.Module):")
print(f"  R² = {r2:+.4f}")
print(f"  MAE = {mae:.6f}")
print(f"  RMSE = {rmse:.6f}")
print(f"  Pred mean: {pred.mean():.6f}")
print(f"  Pred std:  {pred.std():.6f}")
print(f"  Pred min:  {pred.min():.6f}")
print(f"  Pred max:  {pred.max():.6f}")

# Check if ReLU killing output
zero_pct = (pred == 0).sum() / len(pred) * 100
print(f"  Pred % zero: {zero_pct:.1f}%")

# =============================================================================
# HAR OLS BASELINE
# =============================================================================
print(f"\n{'='*70}")
print(f"  HAR OLS (Baseline)")
print(f"{'='*70}")

har_coeffs = fit_har(rv, dates_train[-1])
print(f"  Coefficients: {har_coeffs}")

har_pred_ols = predict_har(rv, har_coeffs, test_ts)
har_pred_aligned = har_pred_ols.reindex(dates_test).dropna()

if len(har_pred_aligned) > 0:
    y_test_aligned = y_test[:len(har_pred_aligned)]
    har_r2 = compute_r2(y_test_aligned, har_pred_aligned.values)
    har_mae = compute_mae(y_test_aligned, har_pred_aligned.values)
    har_rmse = np.sqrt(np.mean((y_test_aligned - har_pred_aligned.values)**2))

    print(f"\n[Results] HAR OLS:")
    print(f"  R² = {har_r2:+.4f}")
    print(f"  MAE = {har_mae:.6f}")
    print(f"  RMSE = {har_rmse:.6f}")
else:
    har_r2, har_mae, har_rmse = np.nan, np.nan, np.nan

# =============================================================================
# COMPARISON
# =============================================================================
print(f"\n{'='*70}")
print(f"  COMPARISON: HAR_nn vs HAR_OLS")
print(f"{'='*70}")
print(f"{'Model':<20} {'R2':>12} {'MAE':>12} {'RMSE':>12} {'Winner'}")
print(f"{'-'*70}")

print(f"{'HAR OLS':<20} {har_r2:>+12.4f} {har_mae:12.6f} {har_rmse:12.6f}", end="")

if r2 > har_r2:
    print(f" {'':>8} <- OLS")
else:
    print(f" {'':>8}")

print(f"{'HAR (nn.Module)':<20} {r2:>+12.4f} {mae:12.6f} {rmse:12.6f}", end="")

if r2 > har_r2:
    print(f" {'WIN':>8} <- BETTER!")
else:
    print(f" {'':>8}")

print(f"{'-'*70}")

diff_r2 = r2 - har_r2
diff_mae = har_mae - mae

print(f"  R² difference:  {diff_r2:+.4f}")
print(f"  MAE difference: {diff_mae:+.6f} ({'positive' if diff_mae > 0 else 'negative'})")

if r2 > har_r2:
    print(f"\n  [SUCCESS] HAR_nn BEATS HAR_OLS by {diff_r2:+.4f} R²!")
elif r2 > har_r2 - 0.1:
    print(f"\n  [CLOSE] HAR_nn is competitive (within 0.1 R²)")
else:
    print(f"\n  [INFO] HAR_OLS is better (expected - OLS has closed-form solution)")

print(f"\n{'='*70}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = _ROOT / 'results' / 'gnnhar_paper' / 'vic_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'ticker': TICKER,
    'horizon': HORIZON,
    'stride': 1,
    'test_start': GLOBAL_TEST_START,
    'n_train': int(len(X_train)),
    'n_val': int(len(X_val)),
    'n_test': int(len(X_test)),
    'train_mean': float(y_train.mean()),
    'test_mean': float(y_test.mean()),
    'har_nn': {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'pred_mean': float(pred.mean()),
        'pred_zero_pct': float(zero_pct)
    },
    'har_ols': {
        'r2': float(har_r2) if not np.isnan(har_r2) else None,
        'mae': float(har_mae) if not np.isnan(har_mae) else None,
        'rmse': float(har_rmse) if not np.isnan(har_rmse) else None,
    },
    'improvement': {
        'r2_diff': float(diff_r2) if not np.isnan(diff_r2) else None,
        'mae_diff': float(diff_mae) if not np.isnan(diff_mae) else None,
    }
}

with open(output_dir / f'har_nn_vic_h{HORIZON}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] Results saved to {output_dir / f'har_nn_vic_h{HORIZON}.json'}")
print("\n" + "="*70)
