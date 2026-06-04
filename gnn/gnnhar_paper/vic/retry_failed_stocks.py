"""
Retry updating failed stocks with longer delays
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
print("  RETRY FAILED STOCKS")
print("="*70 + "\n")

START_DATE = '2007-09-19'
END_DATE = '2026-05-31'
RAW_PRICES_DIR = PROJECT_ROOT / 'data' / 'raw' / 'prices'

# Stocks that failed (still at 2026-05-15)
FAILED_STOCKS = ['SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB']

print(f"[Retry] Updating {len(FAILED_STOCKS)} failed stocks...\n")

success_count = 0
failed_stocks = []

for ticker in FAILED_STOCKS:
    print(f"[{ticker}] ", end='', flush=True)

    data_file = RAW_PRICES_DIR / f'{ticker}_ohlcv.csv'

    try:
        from vnstock.api.quote import Quote

        q = Quote(symbol=ticker, source="VCI")
        df = q.history(start=START_DATE, end=END_DATE, interval="1D")

        if df is None or df.empty:
            print(f"  [FAIL] No data")
            failed_stocks.append(ticker)
            time.sleep(3)
            continue

        # Standardize format
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

        # Save updated data
        df.to_csv(data_file, index=False)

        new_last_date = df['date'].iloc[-1]
        print(f"  [OK] Updated ({len(df)} rows, to {new_last_date})")
        success_count += 1

        # Longer delay to avoid rate limiting
        time.sleep(5)

    except Exception as e:
        print(f"  [FAIL] Error: {str(e)[:50]}")
        failed_stocks.append(ticker)
        time.sleep(5)

print(f"\n{'='*70}")
print(f"  RETRY SUMMARY")
print(f"{'='*70}\n")

print(f"Total: {len(FAILED_STOCKS)}")
print(f"Success: {success_count}")
print(f"Failed: {len(failed_stocks)}")

if failed_stocks:
    print(f"\nStill failed: {', '.join(failed_stocks)}")
    print(f"You may need to:")
    print(f"  1. Wait longer and retry")
    print(f"  2. Download manually from https://finance.vietstock.vn")

print(f"\n{'='*70}\n")
