"""
Crawl VIC stock price data from VnStock API
Extends existing VIC data with latest prices
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*70)
print("  CRAWLING VIC STOCK DATA FROM VNSTOCK API")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKER = 'VIC'
START_DATE = '2026-05-16'  # Start from where existing data ends
END_DATE = datetime.now().strftime('%Y-%m-%d')  # Up to today

# Data file path
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'prices' / 'VIC_ohlcv.csv'

# =============================================================================
# LOAD EXISTING DATA
# =============================================================================

print(f"[Data] Loading existing VIC data from {DATA_FILE}...")
existing_data = pd.read_csv(DATA_FILE)
existing_data['date'] = pd.to_datetime(existing_data['date'])

last_date = existing_data['date'].iloc[-1]
print(f"  Existing data: {len(existing_data)} rows")
print(f"  Date range: {existing_data['date'].iloc[0].date()} to {last_date.date()}")
print()

# =============================================================================
# CRAWL NEW DATA
# =============================================================================

try:
    from vnstock import Vnstock

    print(f"[Crawl] Fetching VIC data from {START_DATE} to {END_DATE}...")

    # Initialize Vnstock
    vn = Vnstock()

    # Fetch historical data
    vic_data = vn.stock_historical_data(symbol=Vn_TICKER, start_date=START_DATE, end_date=END_DATE)

    if vic_data is not None and len(vic_data) > 0:
        # Parse columns
        df = pd.DataFrame(vic_data)

        # Standardize column names to match existing data
        column_mapping = {
            'time': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in required_cols):
            # Convert date format
            df['date'] = pd.to_datetime(df['date'])

            # Sort by date
            df = df.sort_values('date')

            # Remove duplicates
            df = df.drop_duplicates(subset=['date'])

            print(f"  Fetched {len(df)} new rows")
            print(f"  Date range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
            print()

            # =============================================================================
            # MERGE WITH EXISTING DATA
            # =============================================================================

            print(f"[Merge] Merging new data with existing data...")

            # Filter out data that already exists
            new_data = df[df['date'] > last_date].copy()

            if len(new_data) > 0:
                # Select only required columns
                new_data = new_data[required_cols]

                # Append to existing data
                merged_data = pd.concat([existing_data, new_data], ignore_index=True)
                merged_data = merged_data.sort_values('date')
                merged_data = merged_data.reset_index(drop=True)

                print(f"  Added {len(new_data)} new rows")
                print(f"  Total rows after merge: {len(merged_data)}")
                print(f"  New date range: {merged_data['date'].iloc[0].date()} to {merged_data['date'].iloc[-1].date()}")

                # =============================================================================
                # SAVE MERGED DATA
                # =============================================================================

                print(f"\n[Save] Saving to {DATA_FILE}...")
                merged_data.to_csv(DATA_FILE, index=False)
                print(f"  [OK] Saved {len(merged_data)} rows to {DATA_FILE}")
                print()
                print(f"New samples added:")
                for i, row in new_data.iterrows():
                    print(f"  {row['date'].date()}: {row['close']:.2f}")
            else:
                print(f"  [WARN] No new data to add (all fetched dates already exist)")

        else:
            print(f"  [ERROR] Missing required columns: {required_cols}")
            print(f"  Available columns: {df.columns.tolist()}")

    else:
        print(f"  [ERROR] No data returned from VnStock API")

except ImportError:
    print(f"\n[ERROR] vnstock package not installed")
    print(f"  Install with: pip install vnstock")
    print(f"\nAlternative: Manual download")
    print(f"  1. Go to https://finance.vietstock.vn/VIC-chart")
    print(f"  2. Download historical data")
    print(f"  3. Save as CSV with columns: date, open, high, low, close, volume")
    print(f"  4. Place in: {DATA_FILE}")

except Exception as e:
    print(f"\n[ERROR] Failed to fetch data: {e}")
    print(f"\n[INFO] Existing data unchanged")
    print(f"  Last available date: {last_date.date()}")

print(f"\n{'='*70}\n")

# =============================================================================
# VERIFY UPDATED DATA
# =============================================================================

if Path(DATA_FILE).exists():
    print("[Verify] Checking updated VIC data...")
    updated_data = pd.read_csv(DATA_FILE)
    updated_data['date'] = pd.to_datetime(updated_data['date'])

    print(f"  Total rows: {len(updated_data)}")
    print(f"  Date range: {updated_data['date'].iloc[0].date()} to {updated_data['date'].iloc[-1].date()}")

    # Check May 2026
    may_data = updated_data[updated_data['date'] >= '2026-05-01']
    print(f"  May 2026 close prices: {len(may_data)} days")

    if len(may_data) > 10:
        print(f"  [OK] May 2026 has good coverage for testing")

    print()
