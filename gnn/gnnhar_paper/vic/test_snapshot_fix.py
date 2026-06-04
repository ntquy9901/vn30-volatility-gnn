"""
Test if snapshot building fix works
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

print("Testing snapshot building fix...")

# Load data
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=['VIC'])
rv = compute_rv(close, h=5)['VIC'].dropna()

# Split dates
TRAIN_END_DATE = '2026-04-30'
TEST_START_DATE = '2026-05-01'
TEST_END_DATE = '2026-05-31'

# Check what May data exists
may_rv = rv[rv.index >= TEST_START_DATE]
print(f"May RV samples: {len(may_rv)}")
print(f"Date range: {may_rv.index[0]} to {may_rv.index[-1]}")
print()

# Now build snapshots using simple approach
min_history = 22 + 5

X_list, y_list, date_list = [], [], []

# Use the full RV series, but only count May targets
for i in range(min_history, len(rv)):
    # Check we have enough data
    if i + 5 > len(rv):
        break

    current_date = rv.index[i]

    # Only include May targets
    if current_date >= pd.Timestamp(TEST_START_DATE) and current_date <= pd.Timestamp(TEST_END_DATE):
        # Compute features from full series (lookback can be before May)
        target = rv.iloc[i:i+5].mean()
        rv_d = rv.iloc[i-1:i].mean()
        rv_w = rv.iloc[i-5:i].mean()
        rv_m = rv.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)
        date_list.append(current_date)

print(f"Built test snapshots: {len(X_list)} samples")
print(f"Date range: {date_list[0]} to {date_list[-1]}")

if len(X_list) == 15:
    print("[OK] SUCCESS - Got 15 test samples!")
else:
    print(f"[FAIL] Got {len(X_list)} samples (expected 15)")
