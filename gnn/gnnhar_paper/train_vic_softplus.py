"""
VIC HAR Training with SOFTPLUS Activation

Softplus is the "gold standard" for variance/standard deviation prediction:
- f(x) = ln(1 + e^x)
- Always positive (good for RV prediction)
- Smooth (no sharp corner like ReLU)
- Gradient never zero (no dead neurons)
- For large x: softplus(x) ~ x (linear like ReLU)
- For negative x: softplus(x) ~ 0 (but smooth, not abrupt)

This should solve the "ReLU killing output" problem.
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.volatility_labels import load_close_prices, compute_rv
from baselines.har_rv_baseline import fit_har, predict_har

print("\n" + "="*70)
print("  VIC HAR: SOFTPLUS ACTIVATION (Test Stability)")
print("="*70)

# Config
HORIZON = 5
TICKER = 'VIC'
GLOBAL_TEST_START = "2026-01-01"
N_EPOCHS = 1000
LR = 1e-3
PATIENCE = 200

# Load data
import yaml
with open(_ROOT / 'config.yaml') as f:
    _CFG = yaml.safe_load(f)
DATA_DIR = _ROOT / _CFG['data']['prices_dir']

print("[Data] Loading VIC prices...")
close = load_close_prices(DATA_DIR, tickers=[TICKER])
print(f"  Shape: {close.shape}")

rv = compute_rv(close, h=HORIZON)[TICKER].dropna()
print(f"  RV (h={HORIZON}): {len(rv)} samples")

# Build snapshots with stride = 1
def build_snapshots_stride1(rv_series, horizon=5):
    """Build HAR snapshots with STRIDE = 1."""
    min_history = 22 + horizon
    X_list, y_list, date_list = [], [], []

    for i in range(min_history, len(rv_series) - horizon):
        current_date = rv_series.index[i]
        target = rv_series.iloc[i:i+horizon].mean()
        rv_d = rv_series.iloc[i-1:i].mean()
        rv_w = rv_series.iloc[i-5:i].mean()
        rv_m = rv_series.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)
        date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

X, y, dates = build_snapshots_stride1(rv, HORIZON)
print(f"  Total snapshots: {len(X)} (stride=1)")

# Split - Walk-Forward
test_ts = pd.Timestamp(GLOBAL_TEST_START)
pre_test_mask = dates < test_ts

X_pre = X[pre_test_mask]
y_pre = y[pre_test_mask]
dates_pre = dates[pre_test_mask]

# Walk-Forward: Last 1000 days
WALK_FORWARD_WINDOW = 1000
X_train_wf = X_pre[-WALK_FORWARD_WINDOW:]
y_train_wf = y_pre[-WALK_FORWARD_WINDOW:]
dates_train_wf = dates_pre[-WALK_FORWARD_WINDOW:]

n_train_wf = int(len(X_train_wf) * 0.8)
X_train = X_train_wf[:n_train_wf]
y_train = y_train_wf[:n_train_wf]
X_val = X_train_wf[n_train_wf:]
y_val = y_train_wf[n_train_wf:]

# Test set
test_mask = dates >= test_ts
X_test = X[test_mask]
y_test = y[test_mask]
dates_test = dates[test_mask]

print(f"\n[Split] WALK-FORWARD:")
print(f"  Train: {len(X_train)} samples")
print(f"  Val:   {len(X_val)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Train mean: {y_train.mean():.6f}")
print(f"  Test mean:  {y_test.mean():.6f}")

# =============================================================================
# METHOD 1: HAR with SOFTPLUS (new)
# =============================================================================
print(f"\n{'='*70}")
print(f"  METHOD 1: HAR with SOFTPLUS (smooth ReLU)")
print(f"{'='*70}")

class HAR_Softplus(nn.Module):
    """
    HAR model with Softplus activation.

    Softplus: f(x) = ln(1 + e^x)
    - Always positive
    - Smooth gradient
    - No dead neurons
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.softplus = nn.Softplus()  # Smooth, always positive

    def forward(self, x):
        h = self.linear(x)
        h = self.softplus(h)
        return h.squeeze(-1)

def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - (ss_res / ss_tot)

def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """Train model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(N_EPOCHS):
        model.train()
        X_t = torch.from_numpy(X_train).float()
        y_t = torch.from_numpy(y_train).float()

        pred = model(X_t)
        loss = criterion(pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            X_v = torch.from_numpy(X_val).float()
            y_v = torch.from_numpy(y_val).float()
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model

# Train HAR with Softplus
print("\n[Train] Training HAR with Softplus...")
model_softplus = HAR_Softplus()
model_softplus = train_model(model_softplus, X_train, y_train, X_val, y_val, 'HAR_Softplus')

# Evaluate
model_softplus.eval()
with torch.no_grad():
    X_ts = torch.from_numpy(X_test).float()
    softplus_pred = model_softplus(X_ts).numpy()

softplus_r2 = compute_r2(y_test, softplus_pred)
softplus_mae = np.mean(np.abs(y_test - softplus_pred))

print(f"\n[Results] HAR with Softplus:")
print(f"  R² = {softplus_r2:+.4f}")
print(f"  MAE = {softplus_mae:.6f}")
print(f"  Pred mean: {softplus_pred.mean():.6f}")
print(f"  Pred std:  {softplus_pred.std():.6f}")
print(f"  Pred min:  {softplus_pred.min():.6f}")
print(f"  Pred max:  {softplus_pred.max():.6f}")

zero_pct = (softplus_pred < 1e-6).sum() / len(softplus_pred) * 100
print(f"  Pred ~0: {zero_pct:.1f}%")

# =============================================================================
# METHOD 2: HAR with ReLU (for comparison)
# =============================================================================
print(f"\n{'='*70}")
print(f"  METHOD 2: HAR with ReLU (original)")
print(f"{'='*70}")

class HAR_ReLU(nn.Module):
    """HAR model with ReLU (original - can kill output)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.linear(x)
        h = self.relu(h)
        return h.squeeze(-1)

# Train HAR with ReLU
print("\n[Train] Training HAR with ReLU...")
model_relu = HAR_ReLU()
model_relu = train_model(model_relu, X_train, y_train, X_val, y_val, 'HAR_ReLU')

# Evaluate
model_relu.eval()
with torch.no_grad():
    relu_pred = model_relu(X_ts).numpy()

relu_r2 = compute_r2(y_test, relu_pred)
relu_mae = np.mean(np.abs(y_test - relu_pred))

print(f"\n[Results] HAR with ReLU:")
print(f"  R² = {relu_r2:+.4f}")
print(f"  MAE = {relu_mae:.6f}")
print(f"  Pred mean: {relu_pred.mean():.6f}")
zero_pct = (relu_pred == 0).sum() / len(relu_pred) * 100
print(f"  Pred = 0: {zero_pct:.1f}%")

# =============================================================================
# METHOD 3: HAR OLS (baseline)
# =============================================================================
print(f"\n{'='*70}")
print(f"  METHOD 3: HAR OLS (baseline)")
print(f"{'='*70}")

har_coeffs = fit_har(rv, dates_train_wf[-1])
har_pred_ols = predict_har(rv, har_coeffs, test_ts)
har_pred_aligned = har_pred_ols.reindex(dates_test).dropna()

if len(har_pred_aligned) > 0:
    y_test_aligned = y_test[:len(har_pred_aligned)]
    har_r2 = compute_r2(y_test_aligned, har_pred_aligned.values)
    har_mae = np.mean(np.abs(y_test_aligned - har_pred_aligned.values))
    print(f"\n  Results:")
    print(f"    R² = {har_r2:+.4f}")
    print(f"    MAE = {har_mae:.6f}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*70}")
print(f"  SUMMARY: Softplus vs ReLU")
print(f"{'='*70}")
print(f"{'Method':<20} {'R2':>10} {'MAE':>12} {'Zero%'}")
print(f"{'-'*50}")
print(f"{'HAR OLS':<20} {har_r2:>+10.4f} {har_mae:12.6f} {'N/A'}")
print(f"{'HAR + Softplus':<20} {softplus_r2:>+10.4f} {softplus_mae:12.6f} {zero_pct:.1f}%")
print(f"{'HAR + ReLU':<20} {relu_r2:>+10.4f} {relu_mae:12.6f} {zero_pct:.1f}%")
print(f"{'-'*50}")

if softplus_r2 > relu_r2:
    print(f"\n[SUCCESS] Softplus BEATS ReLU by {softplus_r2 - relu_r2:+.4f} R2!")
elif softplus_r2 > har_r2:
    print(f"\n[SUCCESS] Softplus BEATS HAR OLS by {softplus_r2 - har_r2:+.4f} R2!")
else:
    print(f"\n[INFO] HAR OLS remains best baseline")

print(f"\n{'='*70}")
