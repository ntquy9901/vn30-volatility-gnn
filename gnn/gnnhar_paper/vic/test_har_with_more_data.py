"""
Test HAR model with more training data
Check if the issue is just small training set size
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
from gnn.gnnhar_paper.gnnhar_models import HAR

print("\n" + "="*70)
print("  TESTING HAR WITH DIFFERENT TRAINING SIZES")
print("="*70 + "\n")

# Load data
TICKER = 'VIC'
HORIZON = 5
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=[TICKER])
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()

def build_snapshots(rv_series, stride=1):
    """Build HAR snapshots with given stride."""
    min_history = 22 + HORIZON
    X_list, y_list = [], []

    for i in range(min_history, len(rv_series)):
        # Need HORIZON days ahead for target
        if i + HORIZON > len(rv_series):
            break

        target = rv_series.iloc[i:i+HORIZON].mean()
        rv_d = rv_series.iloc[i-1:i].mean()
        rv_w = rv_series.iloc[i-5:i].mean()
        rv_m = rv_series.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)

    X_array = np.array(X_list)
    y_array = np.array(y_list)

    # Apply stride
    return X_array[::stride], y_array[::stride]

# Train/test split - use simple time split
TRAIN_END_DATE = "2026-03-31"

train_end_ts = pd.Timestamp(TRAIN_END_DATE)

# Build all snapshots first
X_all, y_all = build_snapshots(rv, stride=1)

# Split by index (approximate date split)
n_train = int(len(X_all) * 0.95)  # Use 95% for training
X_train = X_all[:n_train]
y_train = y_all[:n_train]
X_test = X_all[n_train:]
y_test = y_all[n_train:]

print(f"Data split:")
print(f"  Total samples: {len(X_all)}")
print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Train mean RV: {y_train.mean():.6f}")
print(f"  Test mean RV:  {y_test.mean():.6f}")
print(f"  Distribution shift: {(y_test.mean() - y_train.mean()) / y_train.mean() * 100:+.1f}%")

# Test with different training sizes (use subset of training data)
train_sizes = [100, 500, 1000, 2000, 4000]

print(f"\n{'='*70}")
print(f"  TESTING HAR WITH DIFFERENT TRAINING SIZES")
print(f"{'='*70}\n")

for n_samples in train_sizes:
    print(f"[{n_samples} training samples]")

    # Use subset of training data
    X_train_subset = X_train[:n_samples]
    y_train_subset = y_train[:n_samples]

    print(f"  Train samples: {len(X_train_subset)}")
    print(f"  Test samples:  {len(X_test)}")

    # Train HAR
    model = HAR()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train_subset).float().unsqueeze(1)
    y_t = torch.from_numpy(y_train_subset).float().unsqueeze(1)
    X_ts = torch.from_numpy(X_test).float().unsqueeze(1)

    # Train
    n_epochs = min(200, max(50, len(X_train_subset) // 5))
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t, torch.ones(1, 1))
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_ts, torch.ones(1, 1))
        pred_np = pred.squeeze(0).squeeze(-1).numpy()

    # Metrics
    r2 = 1 - np.sum((y_test - pred_np)**2) / np.sum((y_test - y_test.mean())**2)
    mae = np.mean(np.abs(y_test - pred_np))
    pred_std = pred_np.std()

    print(f"  Results: R² = {r2:+.4f}, MAE = {mae:.6f}")
    print(f"    Prediction mean: {pred_np.mean():.6f} (target: {y_test.mean():.6f})")
    print(f"    Prediction std:  {pred_std:.6f}")

    if pred_std < 0.001:
        print(f"  [WARN] Predicting near-constant")

    print()

print(f"{'='*70}")
print(f"  ANALYSIS")
print(f"{'='*70}\n")

print("KEY FINDINGS:")
print("  1. Check if R² improves with more training data")
print("  2. Check if prediction std increases (less constant)")
print("  3. Compare to OLS HAR baseline (R² should be similar)")

print("\nEXPECTED:")
print("  - With more data, HAR should approach OLS performance")
print("  - Prediction std should reflect actual volatility variance")
print("  - R² should be reasonable (not catastrophic)")

print(f"\n{'='*70}\n")
