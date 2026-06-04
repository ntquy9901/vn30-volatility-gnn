"""
Filter VN30 stocks by Effective Sample Size (ESS).

ESS_h = N_raw / h (López de Prado 2018, Ch.7)
For h=1, ESS_h = N_raw (number of trading days before cutoff)

Script filters stocks where ESS_h > 3500 trading days.
Expected: likely ZERO stocks qualify (all VN30 have ~2,900 days < 3,500).
This is a diagnostic step to verify data sufficiency assumptions.

Output:
  - Console: ESS summary table
  - CSV: results/stocks_ess_h1_over_3500.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices
from gnn.build_graph import VN30_TICKERS
import yaml


# ──────────────────── CONFIG ──────────────────────
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config["data"]["prices_dir"])
DATA_START = pd.Timestamp(config["data"]["data_start"])
GLOBAL_TEST_START = pd.Timestamp(config["data"]["test_start"])  # 2026-01-01
LOOKBACK = 20
MAX_H = 1  # For ESS calculation: ESS_h=1 = N_raw / 1

ESS_THRESHOLD = 3500  # Filter: ESS_h > 3500

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def filter_stocks_by_ess():
    """
    Load price data, calculate ESS_h=1, filter stocks.
    """
    print("\n" + "=" * 70)
    print("  ESS FILTERING — VN30 Stocks with ESS_h > 3500 (h=1)")
    print("=" * 70)

    # Load close prices
    print(f"\n[Loading] Price data from {DATA_DIR}...")
    prices = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    print(f"   Loaded: {prices.shape[0]} dates x {prices.shape[1]} tickers")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Filter to data BEFORE global test start (2026-01-01)
    prices_train_val = prices[prices.index < GLOBAL_TEST_START]
    print(f"\n   Data before {GLOBAL_TEST_START.date()}: {prices_train_val.shape[0]} dates")

    # Calculate ESS for each stock
    results = []
    for ticker in VN30_TICKERS:
        if ticker not in prices_train_val.columns:
            print(f"   [WARNING] {ticker}: NOT FOUND in data")
            continue

        # Count non-NaN samples (trading days available)
        ticker_data = prices_train_val[ticker].dropna()
        n_raw = len(ticker_data)

        # ESS_h=1 = N_raw / 1
        ess_h1 = n_raw / MAX_H

        # Check if qualifies
        qualifies = "YES" if ess_h1 > ESS_THRESHOLD else "NO"

        results.append({
            "ticker": ticker,
            "n_raw": n_raw,
            "ess_h1": int(ess_h1),
            "qualifies": qualifies,
        })

    results_df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "-" * 70)
    print("  ESS Summary Table (h=1)")
    print("-" * 70)
    print(results_df.to_string(index=False))
    print("-" * 70)

    # Statistics
    n_qualify = (results_df["qualifies"] == "YES").sum()
    n_total = len(results_df)
    print(f"\n  Stocks with ESS_h > {ESS_THRESHOLD}: {n_qualify}/{n_total}")
    print(f"  Range ESS_h: [{results_df['ess_h1'].min()}, {results_df['ess_h1'].max()}]")

    # Save filtered list
    output_csv = RESULTS_DIR / "stocks_ess_h1_over_3500.csv"
    filtered_tickers = results_df[results_df["qualifies"] == "YES"].copy()
    filtered_tickers.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv}")
    print(f"     ({len(filtered_tickers)} stocks qualify)")

    if len(filtered_tickers) == 0:
        print("\n  DIAGNOSTIC NOTE:")
        print(f"      All VN30 stocks have ESS_h < {ESS_THRESHOLD}.")
        print(f"      This confirms data sufficiency constraint:")
        print(f"      - Dataset: ~{results_df['n_raw'].mean():.0f} trading days (avg)")
        print(f"      - Threshold: {ESS_THRESHOLD} trading days")
        print(f"      - Gap: All stocks below threshold")
        print(f"      Phase 2 (training) will have empty dataset by design.")

    print("\n" + "=" * 70)
    print("  END FILTERING")
    print("=" * 70 + "\n")

    return filtered_tickers


if __name__ == "__main__":
    filtered = filter_stocks_by_ess()
