"""
Test HAR stability across multiple random seeds
Check if catastrophic failure is due to bad initialization
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
print("  TESTING HAR STABILITY ACROSS RANDOM SEEDS")
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
    return X_array[::stride], y_array[::stride]

# Build all snapshots
X_all, y_all = build_snapshots(rv, stride=20)  # Use stride=20 like regime-aware
n_train = int(len(X_all) * 0.8)
X_train = X_all[:n_train]
y_train = y_all[:n_train]
X_test = X_all[n_train:]
y_test = y_all[n_train:]

print(f"Data split (stride=20, like regime-aware):")
print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Train mean RV: {y_train.mean():.6f}")
print(f"  Test mean RV:  {y_test.mean():.6f}")

# Test multiple seeds
n_trials = 10
seeds = [42, 123, 456, 789, 321, 654, 987, 111, 222, 333]

print(f"\n{'='*70}")
print(f"  RUNNING {n_trials} TRIALS WITH DIFFERENT SEEDS")
print(f"{'='*70}\n")

results = []

for trial, seed in enumerate(seeds):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = HAR()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).float().unsqueeze(1)
    y_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_ts = torch.from_numpy(X_test).float().unsqueeze(1)

    # Train
    for epoch in range(200):
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
    pred_mean = pred_np.mean()
    pred_std = pred_np.std()

    results.append({
        'seed': seed,
        'r2': r2,
        'mae': mae,
        'pred_mean': pred_mean,
        'pred_std': pred_std
    })

    status = "GOOD" if r2 > -5 else "FAIL" if r2 < -100 else "POOR"
    print(f"  Trial {trial+1} (seed={seed:3d}): R² = {r2:+8.2f}, MAE = {mae:.6f}, mean={pred_mean:+.6f}, std={pred_std:.4f} [{status}]")

# Analyze results
print(f"\n{'='*70}")
print(f"  SUMMARY STATISTICS")
print(f"{'='*70}\n")

r2_values = [r['r2'] for r in results]
mae_values = [r['mae'] for r in results]
pred_means = [r['pred_mean'] for r in results]
pred_stds = [r['pred_std'] for r in results]

print(f"R² statistics:")
print(f"  Mean:   {np.mean(r2_values):+.2f}")
print(f"  Median: {np.median(r2_values):+.2f}")
print(f"  Std:    {np.std(r2_values):.2f}")
print(f"  Min:    {np.min(r2_values):+.2f}")
print(f"  Max:    {np.max(r2_values):+.2f}")

print(f"\nMAE statistics:")
print(f"  Mean:   {np.mean(mae_values):.6f}")
print(f"  Median: {np.median(mae_values):.6f}")
print(f"  Min:    {np.min(mae_values):.6f}")
print(f"  Max:    {np.max(mae_values):.6f}")

# Count successes
n_good = sum(1 for r in r2_values if r > -5)
n_poor = sum(1 for r in r2_values if -5 <= r < -100)
n_fail = sum(1 for r in r2_values if r <= -100)

print(f"\nSuccess rate:")
print(f"  Good (R² > -5):    {n_good}/{n_trials} ({n_good/n_trials*100:.0f}%)")
print(f"  Poor (-100 < R² < -5): {n_poor}/{n_trials} ({n_poor/n_trials*100:.0f}%)")
print(f"  Fail (R² < -100):  {n_fail}/{n_trials} ({n_fail/n_trials*100:.0f}%)")

print(f"\n{'='*70}")
print(f"  CONCLUSION")
print(f"{'='*70}\n")

if n_good >= n_trials * 0.5:
    print("RESULT: HAR is STABLE with 184 training samples")
    print("  -> Most random seeds lead to reasonable performance")
    print("  -> Ensemble averaging would help")
elif n_fail >= n_trials * 0.5:
    print("RESULT: HAR is UNSTABLE with 184 training samples")
    print("  -> Most random seeds lead to catastrophic failure")
    print("  -> Need more training data or different architecture")
else:
    print("RESULT: HAR performance is HIGHLY VARIABLE")
    print("  -> Success depends on random initialization")
    print("  -> Need ensemble training with screening")

print(f"\n{'='*70}\n")
