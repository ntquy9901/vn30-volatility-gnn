"""
Filter VN30 stocks by ESS for h=20.

ESS_h20 = N_raw / 20  (Lopez de Prado 2018, Ch.7)
Threshold: ESS_h20 >= 100

Expected: most/all 30 stocks qualify (N_raw > 2000 for almost all VN30).

Output:
  Console: ESS summary table
  CSV: results/stocks_ess_h20_over100.csv

Usage:
  python scripts/eda/filter_stocks_by_ess_h20.py
"""
import sys
import pandas as pd
from pathlib import Path
import yaml

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices
from gnn.build_graph import VN30_TICKERS

# ─────────────────────────── CONFIG ───────────────────────────────────────────
CONFIG_PATH  = Path(__file__).parent.parent.parent / "config.yaml"
RESULTS_DIR  = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR          = Path(__file__).parent.parent.parent / _cfg["data"]["prices_dir"]
GLOBAL_TEST_START = pd.Timestamp(_cfg["data"]["test_start"])

MAX_H         = 20
ESS_THRESHOLD = 100


def main():
    print("\n" + "=" * 65)
    print("  ESS FILTER -- VN30 stocks with ESS_h20 >= 100")
    print("=" * 65)

    print(f"\n[Loading] {DATA_DIR} ...")
    prices = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    print(f"  Loaded: {prices.shape[0]} dates x {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")

    pre_test = prices[prices.index < GLOBAL_TEST_START]
    print(f"  Pre-test dates (before {GLOBAL_TEST_START.date()}): {len(pre_test)}\n")

    rows = []
    for ticker in VN30_TICKERS:
        if ticker not in pre_test.columns:
            print(f"  [WARN] {ticker}: not in data")
            continue
        n_raw    = int(pre_test[ticker].dropna().__len__())
        ess_h20  = n_raw // MAX_H
        qualifies = "YES" if ess_h20 >= ESS_THRESHOLD else "NO"
        rows.append(dict(ticker=ticker, n_raw=n_raw, ess_h20=ess_h20, qualifies=qualifies))

    df = pd.DataFrame(rows)

    print("-" * 45)
    print("  ESS Summary (h=20, MAX_H=20)")
    print("-" * 45)
    print(df.to_string(index=False))
    print("-" * 45)

    n_yes = (df["qualifies"] == "YES").sum()
    print(f"\n  Stocks with ESS_h20 >= {ESS_THRESHOLD}: {n_yes}/{len(df)}")
    print(f"  ESS_h20 range: [{df['ess_h20'].min()}, {df['ess_h20'].max()}]")

    out_csv = RESULTS_DIR / "stocks_ess_h20_over100.csv"
    qualified = df[df["qualifies"] == "YES"].copy()
    qualified.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    print(f"  ({len(qualified)} stocks qualify)")

    if (df["qualifies"] == "NO").any():
        excluded = df[df["qualifies"] == "NO"]["ticker"].tolist()
        print(f"  Excluded (ESS_h20 < {ESS_THRESHOLD}): {excluded}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
