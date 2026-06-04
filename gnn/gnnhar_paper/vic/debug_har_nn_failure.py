"""
Debug HAR_nn Failure - Line by Line Analysis

HAR_nn is showing catastrophic failure: R² = -3415.95
This means predictions are completely wrong. Let's debug step by step.
"""
import warnings
warnings.filterwarnings("ignore")

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
print("  DEBUGGING HAR_nn CATASTROPHIC FAILURE")
print("="*70 + "\n")

# =============================================================================
# STEP 1: LOAD AND INSPECT DATA
# =============================================================================

print("[STEP 1] Loading VIC data...")
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=['VIC'])
rv = compute_rv(close, h=5)['VIC'].dropna()

print(f"  Total RV samples: {len(rv)}")
print(f"  Date range: {rv.index[0].date()} to {rv.index[-1].date()}")
print(f"  RV mean: {rv.mean():.6f}")
print(f"  RV std:  {rv.std():.6f}")
print(f"  RV range: [{rv.min():.6f}, {rv.max():.6f}]")

# =============================================================================
# STEP 2: BUILD SIMPLE SNAPSHOTS (stride=20 like regime-aware)
# =============================================================================

print(f"\n[STEP 2] Building HAR snapshots...")

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
print(f"  Total snapshots: {len(X)}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

print(f"\n  Data inspection:")
print(f"    X mean: {X.mean(axis=0)}")
print(f"    X std:  {X.std(axis=0)}")
print(f"    y mean: {y.mean():.6f}")
print(f"    y std:  {y.std():.6f}")

# =============================================================================
# STEP 3: SPLIT DATA (mimic regime-aware approach)
# =============================================================================

print(f"\n[STEP 3] Splitting data (regime-aware approach)...")

TRAIN_END_DATE = "2026-03-31"
TEST_START_DATE = "2026-04-01"
TEST_END_DATE = "2026-05-31"

train_end_ts = pd.Timestamp(TRAIN_END_DATE)
test_start_ts = pd.Timestamp(TEST_START_DATE)

# Simple time-based split
train_mask = (rv.index <= train_end_ts)
test_mask = (rv.index >= test_start_ts) & (rv.index <= pd.Timestamp(TEST_END_DATE))

print(f"  Train period: {rv[train_mask].index[0].date()} to {rv[train_mask].index[-1].date()}")
print(f"  Test period:  {rv[test_mask].index[0].date()} to {rv[test_mask].index[-1].date()}")

# Check distributions
train_rv = rv[train_mask]
test_rv = rv[test_mask]

print(f"\n  Distribution check:")
print(f"    Train RV mean: {train_rv.mean():.6f}")
print(f"    Test RV mean:  {test_rv.mean():.6f}")
print(f"    Distribution shift: {(test_rv.mean() - train_rv.mean()) / train_rv.mean() * 100:+.1f}%")

# =============================================================================
# STEP 4: CREATE HAR MODEL (same as gnnhar_models.py)
# =============================================================================

print(f"\n[STEP 4] Creating HAR model (exactly as in gnnhar_models.py)...")

class HAR(nn.Module):
    """HAR baseline: linear regression on HAR features only, no graph."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj=None):
        # (batch_size, N, 3) @ (3, 1) -> (batch_size, N, 1)
        H1 = self.linear1(node_feat)
        # (batch_size, N, 1) -> (batch_size, N)
        res = self.relu(H1)
        return res.squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = HAR()
print(f"  Model created: {model.count_params()} parameters")

# =============================================================================
# STEP 5: PREPARE TRAINING DATA (simple approach)
# =============================================================================

print(f"\n[STEP 5] Preparing training data...")

# Split snapshots by date
train_dates = []
test_dates = []

for i, (rv_d, rv_w, rv_m) in enumerate(X):
    # Approximate date from index
    sample_date = rv.index[22 + 5 + i]  # rough approximation
    if sample_date <= train_end_ts:
        train_dates.append(i)
    elif sample_date >= test_start_ts:
        test_dates.append(i)

X_train = X[train_dates]
y_train = y[train_dates]
X_test = X[test_dates]
y_test = y[test_dates]

print(f"  Train samples: {len(X_train)}")
print(f"  Test samples:  {len(X_test)}")

print(f"\n  Training data statistics:")
print(f"    y_train mean: {y_train.mean():.6f}")
print(f"    y_train std:  {y_train.std():.6f}")
print(f"    y_train range: [{y_train.min():.6f}, {y_train.max():.6f}]")

print(f"  Test data statistics:")
print(f"    y_test mean: {y_test.mean():.6f}")
print(f"    y_test std:  {y_test.std():.6f}")
print(f"    y_test range: [{y_test.min():.6f}, {y_test.max():.6f}]")

# =============================================================================
# STEP 6: INSPECT MODEL INITIALIZATION
# =============================================================================

print(f"\n[STEP 6] Inspecting model initialization...")

print(f"  Initial weights:")
print(f"    linear1.weight: {model.linear1.weight.data}")
print(f"    linear1.bias:   {model.linear1.bias.data}")

# Test forward pass with dummy data
dummy_input = torch.randn(1, 1, 3)
print(f"\n  Test forward pass:")
print(f"    Input: {dummy_input}")
dummy_output = model(dummy_input)
print(f"    Output: {dummy_output}")

# =============================================================================
# STEP 7: TRAIN MODEL (minimal training)
# =============================================================================

print(f"\n[STEP 7] Training HAR model...")

# Convert to tensors - add batch dimension
X_t = torch.from_numpy(X_train).float().unsqueeze(1)  # (1, n_train, 3)
y_t = torch.from_numpy(y_train).float().unsqueeze(1)  # (1, n_train)

# Test data
X_ts = torch.from_numpy(X_test).float().unsqueeze(1)  # (1, n_test, 3)
y_ts = torch.from_numpy(y_test).float().unsqueeze(1)  # (1, n_test)

print(f"  X_train shape: {X_t.shape}")
print(f"  y_train shape: {y_t.shape}")
print(f"  X_test shape:  {X_ts.shape}")
print(f"  y_test shape: {y_ts.shape}")

# Simple adjacency for single node
adj = torch.ones(1, 1, dtype=torch.float32)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"\n  Training for 50 epochs (debugging)...")

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred = model(X_t, adj)  # (1, n_train) -> (1, n_train)
    loss = criterion(pred, y_t)  # scalar

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"    Epoch {epoch:2d}: loss={loss.item():.6f}")

# =============================================================================
# STEP 8: EVALUATE ON TEST DATA
# =============================================================================

print(f"\n[STEP 8] Evaluating on test data...")

model.eval()
with torch.no_grad():
    pred = model(X_ts, adj)  # (1, n_test)

pred_np = pred.squeeze(0).numpy()  # (n_test,)
y_test_np = y_ts.squeeze(0).numpy()  # (n_test,)

print(f"  Predictions shape: {pred_np.shape}")
print(f"  Predictions mean: {pred_np.mean():.6f}")
print(f"  Predictions std:  {pred_np.std():.6f}")
print(f"  Predictions range: [{pred_np.min():.6f}, {pred_np.max():.6f}]")

print(f"\n  Target mean:      {y_test_np.mean():.6f}")
print(f"  Target range:     [{y_test_np.min():.6f}, {y_test_np.max():.6f}]")

# =============================================================================
# STEP 9: CALCULATE METRICS
# =============================================================================

print(f"\n[STEP 9] Calculating metrics...")

# R² calculation
ss_res = np.sum((y_test_np - pred_np) ** 2)
ss_tot = np.sum((y_test_np - y_test_np.mean()) ** 2)
r2 = 1 - (ss_res / (ss_tot + 1e-8))

mae = np.mean(np.abs(y_test_np - pred_np))
rmse = np.sqrt(np.mean((y_test_np - pred_np) ** 2))

print(f"  R² = {r2:.4f}")
print(f"  MAE = {mae:.6f}")
print(f"  RMSE = {rmse:.6f}")

# =============================================================================
# STEP 10: DETAILED PREDICTION ANALYSIS
# =============================================================================

print(f"\n[STEP 10] Detailed prediction analysis...")

# Check if predictions are all zeros
zero_count = (pred_np == 0).sum()
print(f"  Zero predictions: {zero_count}/{len(pred_np)} ({zero_count/len(pred_np)*100:.1f}%)")

# Check prediction distribution
print(f"  Negative predictions: {(pred_np < 0).sum()}/{len(pred_np)} ({(pred_np < 0).sum()/len(pred_np)*100:.1f}%)")

# Sample predictions vs targets
print(f"\n  Sample predictions vs targets:")
for i in range(min(5, len(pred_np))):
    print(f"    Sample {i}: pred={pred_np[i]:.6f}, true={y_test_np[i]:.6f}, error={abs(pred_np[i] - y_test_np[i]):.6f}")

# =============================================================================
# STEP 11: WEIGHT INSPECTION
# =============================================================================

print(f"\n[STEP 11] Trained model weights:")
print(f"  linear1.weight: {model.linear1.weight.data}")
print(f"  linear1.bias:   {model.linear1.bias.data}")

# What should the weights be approximately for HAR?
# OLS solution: RV_t = β0 + β1*RV_d + β2*RV_w + β3*RV_m
print(f"\n  Expected: weights should be positive (positive correlation)")
print(f"  Observed: weight = {model.linear1.weight.data.flatten().numpy()}")

# =============================================================================
# STEP 12: DIAGNOSIS
# =============================================================================

print(f"\n" + "="*70)
print("  DIAGNOSIS")
print("="*70 + "\n")

if r2 < -100:
    print(f"  [CRITICAL] R² = {r2:.2f} indicates CATASTROPHIC failure")
    print(f"  [POSSIBLE CAUSES]:")
    print(f"    1. ReLU activation killing predictions")
    print(f"    2. Training divergence (wrong local minimum)")
    print(f"    3. Data scaling issues")
    print(f"    4. Numerical instability")

if zero_count > len(pred_np) * 0.8:
    print(f"  [CRITICAL] {zero_count/len(pred_np)*100:.1f}% predictions are ZERO!")
    print(f"  [DIAGNOSIS] ReLU is killing the model - all outputs become 0")

if abs(pred_np.mean() - y_test_np.mean()) > 0.02:
    print(f"  [CRITICAL] Prediction mean far from target mean")
    print(f"  [DIAGNOSIS] Model learned to predict wrong level")

print(f"\n[RECOMMENDATION]")
if zero_count > len(pred_np) * 0.5:
    print(f"  -> Remove ReLU activation (use linear output)")
    print(f"  -> Model should predict negative values if needed")
elif r2 < -10:
    print(f"  -> Try longer training or different learning rate")
else:
    print(f"  -> Model behavior seems reasonable")

print(f"\n{'='*70}\n")
