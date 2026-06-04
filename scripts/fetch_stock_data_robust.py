"""
Robust Stock Data Fetcher for Vietnam Stocks
Uses vnstock as primary (full historical data) with exponential backoff and caching
"""
import warnings
warnings.filterwarnings("ignore")

import time
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*70)
print("  ROBUST STOCK DATA FETCHER (Vietnam Stocks)")
print("Strategy: vnstock (primary) + exponential backoff + caching")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

START_DATE = '2007-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'prices'
MAX_WORKERS = 2  # Conservative to avoid rate limiting
BASE_DELAY = 2.0  # Base delay between requests (seconds)

# VN30 tickers
VN30_TICKERS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SHB', 'SSB',
    'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM'
]

# =============================================================================
# EXPONENTIAL BACKOFF WITH JITTER
# =============================================================================

def fetch_with_backoff(fetch_func, max_retries=5, operation_name="fetch"):
    """
    Exponential backoff: 2s, 4s, 8s, 16s, 32s with random jitter
    Jitter prevents synchronized retries from hitting the API simultaneously
    """
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [{operation_name}] Failed after {max_retries} retries")
                raise

            # Exponential backoff: BASE_DELAY * 2^attempt + random(0, 2)
            wait = BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(f"  [{operation_name}] Retry {attempt+1}/{max_retries} after {wait:.1f}s")
            time.sleep(wait)

# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def get_cache_status(ticker, start, end, output_dir):
    """Check if cached data exists and covers requested range"""
    cache_file = output_dir / f'{ticker}_ohlcv.csv'

    if not cache_file.exists():
        return False, None, None

    try:
        cached = pd.read_csv(cache_file)
        cached['date'] = pd.to_datetime(cached['date'])

        cached_start = cached['date'].min()
        cached_end = cached['date'].max()
        cached_rows = len(cached)

        req_start = pd.Timestamp(start)
        req_end = pd.Timestamp(end)

        # Check if cache covers requested range
        is_complete = (cached_start <= req_start) and (cached_end >= req_end)

        return is_complete, cached, (cached_start, cached_end, cached_rows)

    except Exception as e:
        print(f"  [Cache] Error reading: {e}")
        return False, None, None

def save_to_cache(df, ticker, output_dir):
    """Save data to cache file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / f'{ticker}_ohlcv.csv'
    df.to_csv(cache_file, index=False)
    return cache_file

# =============================================================================
# DATA FETCHER (VNSTOCK)
# =============================================================================

def fetch_vnstock(ticker, start, end):
    """Fetch Vietnam stock data using vnstock library"""
    try:
        from vnstock.api.quote import Quote

        # Disable logging
        q = Quote(symbol=ticker, source="VCI", show_log=False)
        df = q.history(start=start, end=end, interval="1D", show_log=False)

        if df is None or df.empty:
            raise ValueError("No data returned from API")

        # Standardize format
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except ImportError:
        raise ImportError("vnstock not installed. Run: pip install vnstock")
    except Exception as e:
        raise Exception(f"vnstock API failed: {str(e)[:100]}")

# =============================================================================
# INCREMENTAL UPDATE
# =============================================================================

def fetch_incremental_update(ticker, cached_end, target_end, output_dir):
    """
    Fetch only new data since cached_end
    Merges with existing cache to avoid full re-fetch
    """
    # Start from day after cached end
    start_date = (cached_end + timedelta(days=1)).strftime('%Y-%m-%d')

    # Don't fetch if already up to date
    if cached_end >= pd.Timestamp(target_end):
        return None

    print(f"  [Incremental] Fetching {start_date} to {target_end}")

    def fetch_new():
        return fetch_vnstock(ticker, start_date, target_end)

    new_data = fetch_with_backoff(fetch_new, max_retries=3, operation_name="incremental")

    if new_data is None or new_data.empty:
        return None

    # Merge with existing cache
    cache_file = output_dir / f'{ticker}_ohlcv.csv'
    old_data = pd.read_csv(cache_file)
    old_data['date'] = pd.to_datetime(old_data['date'])

    # Remove duplicates (keep newer data)
    merged = pd.concat([old_data, new_data], ignore_index=True)
    merged = merged.sort_values('date').drop_duplicates(subset=['date'], keep='last')

    return merged

# =============================================================================
# MAIN FETCH FUNCTION
# =============================================================================

def fetch_stock_data(ticker, start, end, output_dir, force_refresh=False):
    """
    Fetch stock data with intelligent caching and retry logic:
    1. Check cache - if complete, return immediately
    2. If partial cache, fetch incremental update only
    3. If no cache or force refresh, fetch full range
    4. All fetches use exponential backoff with jitter
    """
    cache_file = output_dir / f'{ticker}_ohlcv.csv'

    # Check cache first
    if not force_refresh:
        is_complete, cached, cache_info = get_cache_status(ticker, start, end, output_dir)

        if is_complete:
            print(f"[{ticker}] [Cache] COMPLETE - {cache_info[2]} rows from {cache_info[0].date()}")
            return cached
        elif cached is not None:
            print(f"[{ticker}] [Cache] PARTIAL - {cache_info[2]} rows to {cache_info[1].date()}")
            # Try incremental update
            try:
                merged = fetch_incremental_update(ticker, cache_info[1], end, output_dir)
                if merged is not None:
                    save_to_cache(merged, ticker, output_dir)
                    print(f"[{ticker}] [Incremental] SUCCESS - {len(merged)} rows total")
                    return merged
            except Exception as e:
                print(f"[{ticker}] [Incremental] FAILED - {str(e)[:50]}, fetching full")
    else:
        print(f"[{ticker}] [Force] Refresh requested")

    # Full fetch (with exponential backoff)
    print(f"[{ticker}] [vnstock] Fetching {start} to {end}...")

    def fetch_full():
        return fetch_vnstock(ticker, start, end)

    df = fetch_with_backoff(fetch_full, max_retries=5, operation_name="vnstock")

    if df is None or df.empty:
        raise Exception(f"Failed to fetch {ticker}")

    save_to_cache(df, ticker, output_dir)
    print(f"[{ticker}] [OK] {len(df)} rows ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")

    return df

# =============================================================================
# BULK FETCH WITH SEQUENTIAL PROCESSING
# =============================================================================

def fetch_all_stocks_sequential(tickers, start, end, output_dir, force_refresh=False):
    """
    Fetch stocks sequentially with delays to avoid rate limiting
    More reliable than parallel for rate-limited APIs
    """
    print(f"[Sequential Fetch] {len(tickers)} stocks")
    print(f"  Strategy: Sequential with {BASE_DELAY}s delay + exponential backoff")
    print(f"  Date range: {start} to {end}")
    print()

    results = {}
    success_count = 0
    failed_stocks = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] ", end='', flush=True)

        try:
            df = fetch_stock_data(ticker, start, end, output_dir, force_refresh)
            results[ticker] = df
            success_count += 1

            # Add delay between successful requests (except last)
            if i < len(tickers):
                delay = BASE_DELAY + random.uniform(0, 1)
                print(f"  (waiting {delay:.1f}s...)")
                time.sleep(delay)
            else:
                print()

        except Exception as e:
            failed_stocks.append(ticker)
            print(f"  [FAIL] {str(e)[:50]}")
            # Add longer delay after failure
            time.sleep(5)

    # Summary
    print(f"\n{'='*70}")
    print(f"  FETCH SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(tickers)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\nFailed stocks: {', '.join(failed_stocks)}")
        print(f"\nTo retry failed stocks only:")
        print(f"  python -c \"")
        print(f"    tickers = {failed_stocks}")
        print(f"    from scripts.fetch_stock_data_robust import fetch_all_stocks_sequential")
        print(f"    fetch_all_stocks_sequential(tickers, '{start}', '{end}', Path('data/raw/prices'))")
        print(f"  \"")

    return results, failed_stocks

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Vietnam stock data robustly")
    parser.add_argument("--ticker", type=str, help="Single ticker to fetch")
    parser.add_argument("--force", action="store_true", help="Force refresh, ignore cache")
    parser.add_argument("--start", type=str, default=START_DATE, help="Start date")
    parser.add_argument("--end", type=str, default=END_DATE, help="End date")
    args = parser.parse_args()

    if args.ticker:
        # Single stock
        try:
            df = fetch_stock_data(
                args.ticker,
                args.start,
                args.end,
                OUTPUT_DIR,
                force_refresh=args.force
            )
            print(f"\n[OK] {args.ticker}: {len(df)} rows")
        except Exception as e:
            print(f"\n[FAIL] {args.ticker}: {e}")
            sys.exit(1)
    else:
        # Bulk fetch all stocks
        results, failed = fetch_all_stocks_sequential(
            VN30_TICKERS,
            args.start,
            args.end,
            OUTPUT_DIR,
            force_refresh=args.force
        )

        if len(failed) == 0:
            print("\n[SUCCESS] All stocks fetched successfully!")
        else:
            print(f"\n[PARTIAL] {len(failed)} stocks failed - see retry instructions above")
            sys.exit(1)
