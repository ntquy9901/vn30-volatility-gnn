"""
Update ALL VN30 stock data using existing project infrastructure
Updates data from 2007 to end of May 2026
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*70)
print("  UPDATE ALL VN30 STOCK DATA")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

START_DATE = '2007-09-19'  # Start from existing data start
END_DATE = '2026-05-31'    # Up to end of May

RAW_PRICES_DIR = PROJECT_ROOT / 'data' / 'raw' / 'prices'

# =============================================================================
# GET LIST OF STOCKS
# =============================================================================

print("[Load] Getting list of stocks...")

stock_files = list(RAW_PRICES_DIR.glob('*_ohlcv.csv'))
tickers = []

for f in stock_files:
    ticker = f.stem.replace('_ohlcv', '')
    if ticker != 'collection_summary':
        tickers.append(ticker)

tickers = sorted(tickers)

print(f"  Found {len(tickers)} stocks")
print(f"  Stocks: {', '.join(tickers[:10])}...")

# =============================================================================
# UPDATE EACH STOCK
# =============================================================================

print(f"\n[Update] Updating data for all stocks...\n")

success_count = 0
failed_stocks = []
summary = {}

for ticker in tickers:
    print(f"[{ticker}] ", end='', flush=True)

    data_file = RAW_PRICES_DIR / f'{ticker}_ohlcv.csv'

    try:
        # Load existing data to get last date
        existing_data = pd.read_csv(data_file)
        existing_data['date'] = pd.to_datetime(existing_data['date'])
        last_date = existing_data['date'].iloc[-1]

        # Skip if already up to date
        if last_date >= pd.Timestamp(END_DATE):
            print(f"  [OK] Already up to date ({last_date.date()})")
            summary[ticker] = {'status': 'skipped', 'last_date': str(last_date.date())}
            success_count += 1
            continue

        # Fetch new data using vnstock
        from vnstock.api.quote import Quote

        q = Quote(symbol=ticker, source="VCI")
        df = q.history(start=START_DATE, end=END_DATE, interval="1D")

        if df is None or df.empty:
            print(f"  [FAIL] No data")
            failed_stocks.append(ticker)
            summary[ticker] = {'status': 'failed', 'error': 'no data'}
            time.sleep(1)  # Rate limit
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
        summary[ticker] = {'status': 'success', 'rows': len(df), 'last_date': new_last_date}
        success_count += 1

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    except Exception as e:
        print(f"  [FAIL] Error: {str(e)[:50]}")
        failed_stocks.append(ticker)
        summary[ticker] = {'status': 'error', 'error': str(e)}
        time.sleep(1)

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"  UPDATE SUMMARY")
print(f"{'='*70}\n")

print(f"Total stocks: {len(tickers)}")
print(f"  Success: {success_count}")
print(f"  Skipped (already up to date): {len([t for t in summary.values() if t.get('status') == 'skipped'])}")
print(f"  Failed: {len(failed_stocks)}")

if failed_stocks:
    print(f"\nFailed stocks: {', '.join(failed_stocks)}")

# =============================================================================
# VERIFY MAY COVERAGE
# =============================================================================

print(f"\n[Verify] Checking May 2026 coverage for all stocks...\n")

from src.volatility_labels import load_close_prices, compute_rv

close = load_close_prices(RAW_PRICES_DIR, tickers=tickers)
rv_dict = compute_rv(close, h=5)

print(f"{'Ticker':<10} {'May Close Prices':>15} {'May RV Samples':>15}")
print(f"{'-'*45}")

may_rv_counts = {}
for ticker in tickers:
    if ticker in rv_dict:
        may_rv = rv_dict[ticker][rv_dict[ticker].index >= '2026-05-01']
        may_close = close[ticker][close[ticker].index >= '2026-05-01']
        print(f"{ticker:<10} {len(may_close):>15} {len(may_rv):>15}")
        may_rv_counts[ticker] = len(may_rv)
    else:
        print(f"{ticker:<10} {'ERROR':>15} {'ERROR':>15}")

avg_may_rv = sum(may_rv_counts.values()) / len(may_rv_counts)
print(f"\nAverage May RV samples per stock: {avg_may_rv:.1f}")

if avg_may_rv >= 15:
    print(f"  [OK] Excellent May coverage for evaluation")
elif avg_may_rv >= 10:
    print(f"  [OK] Good May coverage")
else:
    print(f"  [WARN] Low May coverage - consider running again later")

print(f"\n{'='*70}\n")
