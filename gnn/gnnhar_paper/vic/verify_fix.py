"""
Verify the snapshot building fix in train_vic_ensemble.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_rv

# Import the fixed function - we need to execute the training script
# but stop before the training loop
import torch
import torch.nn as nn
import torch.optim as optim

# Copy the build_snapshots_for_period function inline
HORIZON = 5

def build_snapshots_for_period(rv_series, start_date, end_date, stride=1):
    """Build HAR snapshots for a specific period with stride.

    FIXED: Use full RV series for lookback, only filter targets by date.
    Loop through full series and filter by date (not by index calculation).
    If end_date beyond data, use last available date.
    """
    min_history = 22 + HORIZON

    X_list, y_list, date_list = [], [], []

    # Use full RV series for lookback
    full_rv = rv_series
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # If end_date is beyond last data date, use last data date
    if end_ts > full_rv.index[-1]:
        end_ts = full_rv.index[-1]

    # Loop through full series, filter targets by date range
    for i in range(min_history, len(full_rv), stride):
        # Current date for this snapshot
        current_date = full_rv.index[i]

        # Only include if target date is in range
        if start_ts <= current_date <= end_ts:
            # RV_t is already h-day volatility, just use it directly
            target = full_rv.iloc[i]

            # Lookback features (use full series)
            rv_d = full_rv.iloc[i-1:i].mean()
            rv_w = full_rv.iloc[i-5:i].mean()
            rv_m = full_rv.iloc[i-22:i].mean()

            X_list.append([rv_d, rv_w, rv_m])
            y_list.append(target)
            date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

print("[Verify] Testing fixed snapshot building function...")

# Load data
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=['VIC'])
rv = compute_rv(close, h=HORIZON)['VIC'].dropna()

# Test configurations
configs = [
    {'train_end': '2026-04-30', 'test_start': '2026-05-01', 'test_end': '2026-05-31'},
]

for config in configs:
    train_end = config['train_end']
    test_start = config['test_start']
    test_end = config['test_end']

    print(f"\nConfig: Train until {train_end}, Test {test_start} to {test_end}")

    # Build test snapshots
    X_test, y_test, test_dates = build_snapshots_for_period(
        rv, start_date=test_start, end_date=test_end, stride=1
    )

    print(f"  Test samples: {len(X_test)}")
    if len(X_test) > 0:
        print(f"  Date range: {test_dates[0].date()} to {test_dates[-1].date()}")
        print(f"  y_test mean: {y_test.mean():.6f}")

    # Check results
    if len(X_test) == 11:
        print(f"  [OK] Got expected 11 test samples")
    elif len(X_test) == 1:
        print(f"  [FAIL] Only 1 sample - bug still present!")
    else:
        print(f"  [WARN] Got {len(X_test)} samples (expected 11)")

print(f"\n{'='*70}\n")
