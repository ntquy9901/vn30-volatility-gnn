"""
Diagnose why neural models are failing on VIC
Check if models are learning meaningful predictions or just constants
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_rv
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer

print("\n" + "="*70)
print("  DIAGNOSING NEURAL MODEL PREDICTION FAILURE")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA (same as training script)
# =============================================================================

TICKER = 'VIC'
HORIZON = 5
TRAIN_END_DATE = "2026-03-31"
TEST_START_DATE = "2026-04-01"
TEST_END_DATE = "2026-05-31"

print("[Data] Loading VIC data...")
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=[TICKER])
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()

def build_snapshots_for_period(rv_series, start_date, end_date, stride=20):
    """Build HAR snapshots for a specific period with stride."""
    min_history = 22 + HORIZON

    period_mask = (rv_series.index >= start_date) & (rv_series.index <= end_date)
    period_rv = rv_series[period_mask]

    if len(period_rv) < min_history:
        pre_period_start = start_date - pd.Timedelta(days=60)
        pre_mask = (rv_series.index >= pre_period_start) & (rv_series.index < start_date)
        period_rv = pd.concat([rv_series[pre_mask], period_rv])

    X_list, y_list, date_list = [], [], []

    for i in range(min_history, len(period_rv), stride):
        current_date = period_rv.index[i]

        if current_date < start_date:
            continue

        if i + HORIZON > len(period_rv):
            break

        target = period_rv.iloc[i:i+HORIZON].mean()
        rv_d = period_rv.iloc[i-1:i].mean()
        rv_w = period_rv.iloc[i-5:i].mean()
        rv_m = period_rv.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)
        date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

train_end_ts = pd.Timestamp(TRAIN_END_DATE)
test_start_ts = pd.Timestamp(TEST_START_DATE)
test_end_ts = pd.Timestamp(TEST_END_DATE)

# Build splits
X_train_full, y_train_full, train_dates_full = build_snapshots_for_period(
    rv, start_date=rv.index[0], end_date=train_end_ts, stride=20
)

split_point = int(len(X_train_full) * 0.8)
X_train = X_train_full[:split_point]
y_train = y_train_full[:split_point]
X_val = X_train_full[split_point:]
y_val = y_train_full[split_point:]

X_test, y_test, test_dates = build_snapshots_for_period(
    rv, start_date=test_start_ts, end_date=test_end_ts, stride=1
)

print(f"  Train: {len(X_train)} samples, mean RV = {y_train.mean():.6f}")
print(f"  Val:   {len(X_val)} samples, mean RV = {y_val.mean():.6f}")
print(f"  Test:  {len(X_test)} samples, mean RV = {y_test.mean():.6f}")
print(f"  Distribution shift: {(y_test.mean() - y_train.mean()) / y_train.mean() * 100:+.1f}%")

# =============================================================================
# DIAGNOSE PREDICTION PATTERNS
# =============================================================================

print(f"\n{'='*70}")
print(f"  TESTING IF MODELS PREDICT CONSTANTS")
print(f"{'='*70}\n")

adj = np.array([[1.0]])

for model_name in ['HAR', 'GHAR', 'GNNHAR1L']:
    print(f"[{model_name}]")

    # Train briefly (20 epochs)
    if model_name == 'HAR':
        model = MODEL_REGISTRY[model_name]()
    else:
        model = MODEL_REGISTRY[model_name](n_hid=16)

    X_train_3d = X_train.reshape(-1, 1, 3)
    X_val_3d = X_val.reshape(-1, 1, 3)
    X_test_3d = X_test.reshape(-1, 1, 3)

    y_train_2d = y_train.reshape(-1, 1)
    y_val_2d = y_val.reshape(-1, 1)

    trainer = EnsembleTrainer(
        model_name=model_name,
        n_hid=16,
        n_epochs=20,
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=-1,
        patience=40
    )

    history = trainer.train_single(
        X_train_3d, y_train_2d,
        X_val_3d, y_val_2d,
        adj,
        seed=42,
        verbose=False
    )

    # Get predictions
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_3d).float()
        pred = model(X_test_t, torch.from_numpy(adj).float())
        pred = pred.numpy().flatten()

    # Analyze predictions
    pred_mean = pred.mean()
    pred_std = pred.std()
    pred_min = pred.min()
    pred_max = pred.max()

    print(f"  Predictions:")
    print(f"    Mean:   {pred_mean:.6f} (Train mean: {y_train.mean():.6f}, Test mean: {y_test.mean():.6f})")
    print(f"    Std:    {pred_std:.6f}")
    print(f"    Range:  [{pred_min:.6f}, {pred_max:.6f}]")
    print(f"    Unique values: {len(np.unique(np.round(pred, 6)))} / {len(pred)}")

    # Check if predicting constant
    if pred_std < 1e-6:
        print(f"  [CRITICAL] Model is predicting constant {pred_mean:.6f}")
    elif pred_std < 0.001:
        print(f"  [WARN] Model predictions have very low variance")

    # Compare to naive baseline (predict training mean)
    naive_pred = np.full_like(y_test, y_train.mean())
    naive_mae = np.mean(np.abs(y_test - naive_pred))
    model_mae = np.mean(np.abs(y_test - pred))

    print(f"  Naive MAE (predict train mean): {naive_mae:.6f}")
    print(f"  Model MAE: {model_mae:.6f}")

    if abs(model_mae - naive_mae) / naive_mae < 0.1:
        print(f"  [CRITICAL] Model performs same as naive constant!")

    print()

# =============================================================================
# CHECK GRADIENT FLOW
# =============================================================================

print(f"{'='*70}")
print(f"  CHECKING GRADIENT FLOW DURING TRAINING")
print(f"{'='*70}\n")

# Simple test: check if gradients are flowing
model = MODEL_REGISTRY['HAR']()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

X_t = torch.from_numpy(X_train.reshape(-1, 1, 3)).float()
y_t = torch.from_numpy(y_train.reshape(-1, 1)).float()

for epoch in range(5):
    model.train()
    optimizer.zero_grad()

    pred = model(X_t, torch.from_numpy(adj).float())
    loss = torch.nn.functional.mse_loss(pred, y_t)

    loss.backward()

    # Check gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"  Epoch {epoch}: loss={loss.item():.6f}, grad_norm={total_norm:.6f}")

    optimizer.step()

print(f"\n{'='*70}")
print(f"  DIAGNOSIS COMPLETE")
print(f"{'='*70}\n")
