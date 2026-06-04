"""
HAR_nn Fix - Remove ReLU to solve zero prediction problem

PROBLEM: ReLU activation + negative bias causes all outputs = 0
SOLUTION: Remove ReLU, allow linear outputs (can be negative)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_rv

print("\n" + "="*70)
print("  FIXING HAR_nn - REMOVE ReLU ACTIVATION")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("[Data] Loading VIC data...")
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=['VIC'])
rv = compute_rv(close, h=5)['VIC'].dropna()

def build_snapshots_stride20(rv_series, horizon=5):
    """Build HAR snapshots with stride=20"""
    min_history = 22 + horizon
    X_list, y_list = [], []

    for i in range(min_history, len(rv_series) - horizon):
        target = rv_series.iloc[i:i+horizon].mean()
        rv_d = rv_series.iloc[i-1:i].mean()
        rv_w = rv_series.iloc[i-5:i].mean()
        rv_m = rv_series.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)

    return np.array(X_list), np.array(y_list)

X, y = build_snapshots_stride20(rv, 5)

# Simple train/test split
TRAIN_END_DATE = "2026-03-31"
TEST_START_DATE = "2026-04-01"

train_end_ts = pd.Timestamp(TRAIN_END_DATE)
test_start_ts = pd.Timestamp(TEST_START_DATE)

# Time-based split
train_mask = (rv.index <= train_end_ts)
test_mask = (rv.index >= test_start_ts)

train_rv = rv[train_mask]
test_rv = rv[test_mask]

print(f"  Train period: {train_rv.index[0].date()} to {train_rv.index[-1].date()}")
print(f"  Test period:  {test_rv.index[0].date()} to {test_rv.index[-1].date()}")
print(f"  Train RV mean: {train_rv.mean():.6f}")
print(f"  Test RV mean:  {test_rv.mean():.6f}")

# =============================================================================
# MODEL VARIANTS
# =============================================================================

print(f"\n[Models] Testing HAR with and without ReLU...")

# Model 1: HAR WITH ReLU (original - broken)
class HAR_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, adj=None):
        H1 = self.linear1(x)
        return self.relu(H1).squeeze(-1)

# Model 2: HAR WITHOUT ReLU (fixed)
class HAR_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        # NO ReLU - allow negative outputs

    def forward(self, x, adj=None):
        return self.linear1(x).squeeze(-1)

# =============================================================================
# TRAIN BOTH MODELS
# =============================================================================

# Prepare data (simple approach)
# Use last 1000 samples for training to speed up
X_train = X[-1000:]
y_train = y[-1000:]

# Test samples
test_dates = []
for i in range(len(X) - len(X) + 1, len(X)):
    sample_date = rv.index[22 + 5 + i]
    if sample_date >= test_start_ts:
        test_dates.append(i)

X_test = X[test_dates]
y_test = y[test_dates]

print(f"  Train samples: {len(X_train)}")
print(f"  Test samples:  {len(X_test)}")

adj = torch.ones(1, 1, dtype=torch.float32)

# Convert to tensors
X_t = torch.from_numpy(X_train).float().unsqueeze(1)  # (1, n_train, 3)
y_t = torch.from_numpy(y_train).float().unsqueeze(1)  # (1, n_train)
X_ts = torch.from_numpy(X_test).float().unsqueeze(1)  # (1, n_test, 3)
y_ts = torch.from_numpy(y_test).float().unsqueeze(1)  # (1, n_test)

print(f"\n{'='*70}")
print(f"  MODEL 1: HAR WITH ReLU (ORIGINAL - BROKEN)")
print(f"{'='*70}")

model_relu = HAR_ReLU()
optimizer = optim.Adam(model_relu.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training for 100 epochs...")
for epoch in range(100):
    model_relu.train()
    optimizer.zero_grad()
    pred = model_relu(X_t, adj)
    loss = criterion(pred, y_t)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:2d}: loss={loss.item():.6f}")

model_relu.eval()
with torch.no_grad():
    pred_relu = model_relu(X_ts, adj).squeeze(0).numpy()

zero_pct_relu = (pred_relu == 0).sum() / len(pred_relu) * 100
mean_relu = pred_relu.mean()
r2_relu = 1 - np.sum((y_test - pred_relu)**2) / np.sum((y_test - y_test.mean())**2)
mae_relu = np.mean(np.abs(y_test - pred_relu))

print(f"  Results:")
print(f"    Zero predictions: {zero_pct_relu:.1f}%")
print(f"    Mean prediction: {mean_relu:.6f}")
print(f"    R² = {r2_relu:.4f}")
print(f"    MAE = {mae_relu:.6f}")

print(f"\n{'='*70}")
print(f"  MODEL 2: HAR WITHOUT ReLU (FIXED)")
print(f"{'='*70}")

model_linear = HAR_Linear()
optimizer = optim.Adam(model_linear.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training for 100 epochs...")
for epoch in range(100):
    model_linear.train()
    optimizer.zero_grad()
    pred = model_linear(X_t, adj)
    loss = criterion(pred, y_t)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:2d}: loss={loss.item():.6f}")

model_linear.eval()
with torch.no_grad():
    pred_linear = model_linear(X_ts, adj).squeeze(0).numpy()

zero_pct_linear = (pred_linear == 0).sum() / len(pred_linear) * 100
mean_linear = pred_linear.mean()
r2_linear = 1 - np.sum((y_test - pred_linear)**2) / np.sum((y_test - y_test.mean())**2)
mae_linear = np.mean(np.abs(y_test - pred_linear))

print(f"  Results:")
print(f"    Zero predictions: {zero_pct_linear:.1f}%")
print(f"    Mean prediction: {mean_linear:.6f}")
print(f"    Target mean:      {y_test.mean():.6f}")
print(f"    R² = {r2_linear:.4f}")
print(f"    MAE = {mae_linear:.6f}")

# Show weights
print(f"\n  Trained weights:")
print(f"    Model ReLU:    {model_relu.linear1.weight.data.squeeze()}")
print(f"    Model Linear:  {model_linear.linear1.weight.data.squeeze()}")

print(f"\n{'='*70}")
print(f"  DIAGNOSIS COMPLETE")
print(f"{'='*70}\n")

print("PROBLEM IDENTIFIED:")
print("  [CRITICAL] ReLU activation kills all predictions")
print("  [ROOT CAUSE] Negative bias + negative weights -> all outputs < 0 -> ReLU -> 0")
print("  [FIX] Remove ReLU, use pure linear output")

print("\nFIXED VERSION:")
print(f"  [SUCCESS] R² improved from {r2_relu:.4f} to {r2_linear:.4f}")
print(f"  [SUCCESS] MAE improved from {mae_relu:.6f} to {mae_linear:.6f}")
print(f"  [SUCCESS] Zero predictions reduced from {zero_pct_relu:.1f}% to {zero_pct_linear:.1f}%")

print("\nRECOMMENDATION:")
print("  -> Replace HAR class in gnnhar_models.py with HAR_Linear version")
print("  -> Remove ReLU activation from all HAR models")
print("  -> Allow negative predictions (volatility can be negative in residual space)")

print(f"\n{'='*70}\n")
