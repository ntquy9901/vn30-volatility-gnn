"""
Analyze if VIC models need more epochs or if they've converged
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
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY

print("\n" + "="*70)
print("  ANALYZING IF 750 EPOCHS IS ENOUGH FOR VIC MODELS")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

TICKER = 'VIC'
HORIZON = 5
TRAIN_END_DATE = "2026-03-31"

close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=[TICKER])
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()

def build_snapshots(rv_series, stride=1):
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
    return np.array(X_list), np.array(y_list)

# Build train/val
X_all, y_all = build_snapshots(rv, stride=1)
n_train = int(len(X_all) * 0.8)
X_train = X_all[:n_train]
y_train = y_all[:n_train]
X_val = X_all[n_train:]
y_val = y_all[n_train:]

print(f"Data: {len(X_train)} train, {len(X_val)} val samples")

# =============================================================================
# TEST TRAINING WITH DIFFERENT EPOCH LIMITS
# =============================================================================

configs_to_test = [750, 1500, 2000]
seeds_to_test = [42, 123, 456]  # Test 3 seeds

print(f"\n{'='*70}")
print(f"  TESTING CONVERGENCE WITH DIFFERENT EPOCH LIMITS")
print(f"{'='*70}\n")

results_by_epochs = {}

for max_epochs in configs_to_test:
    print(f"\n[Testing {max_epochs} epochs]")

    seed_results = []

    for seed in seeds_to_test:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = MODEL_REGISTRY['GHAR'](n_hid=16)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train).float().unsqueeze(1)
        y_t = torch.from_numpy(y_train).float().unsqueeze(1)
        X_v = torch.from_numpy(X_val).float().unsqueeze(1)
        y_v = torch.from_numpy(y_val).float().unsqueeze(1)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 100
        stopped_at = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_t, torch.ones(1, 1))
            train_loss = criterion(pred, y_t)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_v, torch.ones(1, 1))
                val_loss = criterion(val_pred, y_v).item()

            train_losses.append(train_loss.item())
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stopped_at = epoch + 1
                break

        if stopped_at == 0:
            stopped_at = max_epochs

        # Check convergence
        last_100 = val_losses[-100:] if len(val_losses) >= 100 else val_losses
        improvement = last_100[0] - last_100[-1] if len(last_100) > 1 else 0

        seed_results.append({
            'seed': seed,
            'stopped_at': stopped_at,
            'best_val_loss': best_val_loss,
            'improvement_last_100': improvement,
            'final_val_loss': val_losses[-1]
        })

    results_by_epochs[max_epochs] = seed_results

    # Print summary
    stopped_ats = [s['stopped_at'] for s in seed_results]
    improvements = [s['improvement_last_100'] for s in seed_results]

    print(f"  Stopped at epochs: {stopped_ats}")
    print(f"  Avg stopped at: {np.mean(stopped_ats):.0f}")
    print(f"  Improvement last 100 epochs: {[f'{i:.6f}' for i in improvements]}")
    print(f"  Avg improvement: {np.mean(improvements):.6f}")

    if all(s == 0 for s in improvements):
        print(f"  [DIAGNOSIS] Models converged - NO improvement in last 100 epochs")
    elif np.mean(improvements) < 0.0001:
        print(f"  [DIAGNOSIS] Models mostly converged - minimal improvement (<0.0001)")
    else:
        print(f"  [DIAGNOSIS] Models still improving - MORE EPOCHS NEEDED")

print(f"\n{'='*70}")
print(f"  RECOMMENDATION")
print(f"{'='*70}\n")

# Analyze if 750 epochs is enough
if results_by_epochs[750]:
    improvements_750 = [s['improvement_last_100'] for s in results_by_epochs[750]]
    avg_stopped_750 = np.mean([s['stopped_at'] for s in results_by_epochs[750]])

    print(f"Current setup (750 epochs, patience=100):")
    print(f"  Average stopping epoch: {avg_stopped_750:.0f}")
    print(f"  Average improvement in last 100 epochs: {np.mean(improvements_750):.6f}")

    if avg_stopped_750 < 750:
        print(f"\n  → Models stop early at epoch {avg_stopped_750:.0f} (patience reached)")
        print(f"  → 750 epochs is SUFFICIENT (early stopping prevents overfitting)")
    elif np.mean(improvements_750) < 0.0001:
        print(f"\n  → Models converge before 750 epochs (improvement < 0.0001)")
        print(f"  → 750 epochs is SUFFICIENT")
    else:
        print(f"\n  → Models still improving at epoch 750")
        print(f"  → RECOMMEND: Increase to {max(1500, avg_stopped_750 + 500):.0f} epochs")

print(f"\n{'='*70}\n")
