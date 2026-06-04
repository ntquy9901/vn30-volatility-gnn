"""
Update VIC stock data using existing project infrastructure
Based on D:/bmad-projects/luanvan/src/data/collect_extended_data.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*70)
print("  UPDATE VIC STOCK DATA")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKER = 'VIC'
START_DATE = '2007-09-19'  # Start from existing data start
END_DATE = '2026-05-31'    # Up to end of May (or use today's date)

DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'prices' / 'VIC_ohlcv.csv'

# =============================================================================
# LOAD EXISTING DATA
# =============================================================================

print(f"[Load] Loading existing VIC data...")
existing_data = pd.read_csv(DATA_FILE)
existing_data['date'] = pd.to_datetime(existing_data['date'])

last_date = existing_data['date'].iloc[-1]
print(f"  Current: {len(existing_data)} rows")
print(f"  Date range: {existing_data['date'].iloc[0].date()} to {last_date.date()}")

# =============================================================================
# FETCH NEW DATA USING VNSTOCK API
# =============================================================================

print(f"\n[Fetch] Fetching VIC data from {START_DATE} to {END_DATE}...")

try:
    from vnstock.api.quote import Quote

    q = Quote(symbol=TICKER, source="VCI")
    df = q.history(start=START_DATE, end=END_DATE, interval="1D")

    if df is None or df.empty:
        print(f"  [ERROR] No data returned")
        sys.exit(1)

    # Standardize format
    df = df.rename(columns={"time": "date"})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  [OK] Fetched {len(df)} rows")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

except Exception as e:
    print(f"  [ERROR] Failed to fetch: {e}")
    print(f"\n[Alternative] Manual download:")
    print(f"  1. Visit: https://finance.vietstock.vn/VIC#hist")
    print(f"  2. Download historical data CSV")
    print(f"  3. Place in: {DATA_FILE}")
    sys.exit(1)

# =============================================================================
# MERGE WITH EXISTING DATA
# =============================================================================

print(f"\n[Merge] Merging with existing data...")

# Remove duplicates (keep newer data)
existing_filtered = existing_data[existing_data['date'] < df['date'].iloc[0]]

if len(existing_filtered) > 0:
    merged_data = pd.concat([existing_filtered, df], ignore_index=True)
else:
    merged_data = df.copy()

merged_data = merged_data.sort_values('date').reset_index(drop=True)

print(f"  Previous: {len(existing_data)} rows")
print(f"  Fetched: {len(df)} rows")
print(f"  After merge: {len(merged_data)} rows")
print(f"  New date range: {merged_data['date'].iloc[0]} to {merged_data['date'].iloc[-1]}")

# =============================================================================
# SAVE UPDATED DATA
# =============================================================================

print(f"\n[Save] Saving to {DATA_FILE}...")
merged_data.to_csv(DATA_FILE, index=False)
print(f"  [OK] Saved {len(merged_data)} rows")

# =============================================================================
# VERIFY MAY 2026 COVERAGE
# =============================================================================

print(f"\n[Verify] Checking May 2026 coverage...")
may_data = merged_data[merged_data['date'] >= '2026-05-01']
print(f"  May 2026 close prices: {len(may_data)} days")

if len(may_data) > 10:
    print(f"  [OK] Good May coverage for testing")

print(f"\n{'='*70}\n")
