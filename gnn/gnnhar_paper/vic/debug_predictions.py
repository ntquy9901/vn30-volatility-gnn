"""
Debug what models are predicting
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

print("\n" + "="*70)
print("  DEBUG: CHECK MODEL PREDICTIONS")
print("="*70 + "\n")

# Load data
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=['VIC'])
rv = compute_rv(close, h=5)['VIC'].dropna()

# Split
TRAIN_END = '2026-04-30'
TEST_START = '2026-05-01'
TEST_END = '2026-05-31'

train_rv = rv[rv.index <= pd.Timestamp(TRAIN_END)]
test_rv = rv[(rv.index >= pd.Timestamp(TEST_START)) & (rv.index <= pd.Timestamp(TEST_END))]

print(f"Train RV: {len(train_rv)} samples")
print(f"Test RV: {len(test_rv)} samples")
print(f"Train mean: {train_rv.mean():.6f}")
print(f"Test mean: {test_rv.mean():.6f}")
print()

# Build test snapshot (last valid one)
HORIZON = 5
min_history = 22 + HORIZON

# Find last test snapshot
test_indices = []
for i in range(min_history, len(rv)):
    if i + HORIZON > len(rv):
        break
    current_date = rv.index[i]
    if pd.Timestamp(TEST_START) <= current_date <= pd.Timestamp(TEST_END):
        test_indices.append(i)

if len(test_indices) == 0:
    print("[ERROR] No test snapshots found!")
    sys.exit(1)

# Use the last test snapshot
i = test_indices[-1]
current_date = rv.index[i]

print(f"Last test snapshot:")
print(f"  Index: {i}")
print(f"  Date: {current_date.date()}")
print(f"  Target (RV {HORIZON}-day): {rv.iloc[i:i+HORIZON].mean():.6f}")
print()

# Build features
rv_d = rv.iloc[i-1:i].mean()
rv_w = rv.iloc[i-5:i].mean()
rv_m = rv.iloc[i-22:i].mean()

X = np.array([[rv_d, rv_w, rv_m]])
print(f"Features: RV_d={rv_d:.6f}, RV_w={rv_w:.6f}, RV_m={rv_m:.6f}")

# Test each model
print(f"\n{'='*70}")
print(f"  MODEL PREDICTIONS (Last Test Snapshot)")
print(f"{'='*70}\n")

y_true = rv.iloc[i:i+HORIZON].mean()

for model_name in ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']:
    # Load first trained model
    model_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'vic_ensemble_models' / model_name
    model_file = model_dir / 'model_0.pt'

    if not model_file.exists():
        print(f"[{model_name}] No trained model found")
        continue

    # Create model and load weights
    if model_name == 'HAR':
        model = MODEL_REGISTRY[model_name]()
    else:
        model = MODEL_REGISTRY[model_name](n_hid=16)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Predict
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        pred = model(X_t, torch.ones(1, 1))
        y_pred = pred.item()

    error = y_pred - y_true
    pct_error = error / y_true * 100

    print(f"[{model_name}]")
    print(f"  True:     {y_true:.6f}")
    print(f"  Predicted: {y_pred:.6f}")
    print(f"  Error:    {error:+.6f} ({pct_error:+.1f}%)")

    # Check if prediction is suspicious
    if abs(y_pred) < 1e-6:
        print(f"  [WARN] Prediction ~0 (possible ReLU bug!)")
    if abs(pct_error) > 100:
        print(f"  [WARN] Error > 100% (model not learning)")
    print()

print(f"{'='*70}\n")
