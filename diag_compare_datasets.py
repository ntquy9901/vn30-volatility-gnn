"""
So sánh 2 dataset: data/raw/prices vs data/raw_20260517/prices
Kiểm tra: date range, số ngày, overlap, giá trị OHLCV có nhất quán không.

Chạy: python diag_compare_datasets.py
Output: results/data_compare/
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

warnings.filterwarnings("ignore")
OUT_DIR = Path("results/data_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLD_DIR = Path("data/raw/prices")
NEW_DIR = Path("data/raw_20260517/prices")

# ── Load all files ────────────────────────────────────────────────────────────

def load_dir(d):
    frames = {}
    for f in sorted(d.glob("*_ohlcv.csv")):
        ticker = f.stem.replace("_ohlcv", "")
        df = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
        frames[ticker] = df
    return frames

print("Loading datasets...")
old = load_dir(OLD_DIR)
new = load_dir(NEW_DIR)

old_tickers = set(old.keys())
new_tickers = set(new.keys())

only_old  = sorted(old_tickers - new_tickers)
only_new  = sorted(new_tickers - old_tickers)
in_both   = sorted(old_tickers & new_tickers)

print(f"  OLD: {len(old_tickers)} stocks")
print(f"  NEW: {len(new_tickers)} stocks")
print(f"  Only in OLD:  {only_old}")
print(f"  Only in NEW:  {only_new}")
print(f"  In both:      {len(in_both)} stocks")

# ── Per-stock comparison ──────────────────────────────────────────────────────

records = []
for ticker in sorted(old_tickers | new_tickers):
    row = {"ticker": ticker}

    if ticker in old:
        df = old[ticker]
        row["old_start"] = df.index[0].date()
        row["old_end"]   = df.index[-1].date()
        row["old_days"]  = len(df)
    else:
        row["old_start"] = None
        row["old_end"]   = None
        row["old_days"]  = 0

    if ticker in new:
        df = new[ticker]
        row["new_start"] = df.index[0].date()
        row["new_end"]   = df.index[-1].date()
        row["new_days"]  = len(df)
    else:
        row["new_start"] = None
        row["new_end"]   = None
        row["new_days"]  = 0

    row["extra_days"] = row["new_days"] - row["old_days"]
    records.append(row)

df_cmp = pd.DataFrame(records).set_index("ticker")

# ── Print comparison table ────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("  DATASET COMPARISON — Per Stock")
print(f"{'='*80}")
print(f"{'Ticker':<8} {'OLD start':>12} {'OLD end':>12} {'OLD days':>9} "
      f"{'NEW start':>12} {'NEW end':>12} {'NEW days':>9} {'Extra':>8}")
print("-" * 80)
for t, row in df_cmp.iterrows():
    flag = "[NEW]" if row["old_days"] == 0 else \
           "[MISS]" if row["new_days"] == 0 else ""
    extra = f"+{row['extra_days']}" if row["extra_days"] > 0 else str(row["extra_days"])
    print(f"  {t:<6} "
          f"{str(row['old_start']):>12} {str(row['old_end']):>12} {row['old_days']:>9} "
          f"{str(row['new_start']):>12} {str(row['new_end']):>12} {row['new_days']:>9} "
          f"{extra:>8}  {flag}")

# ── Overlap consistency check ─────────────────────────────────────────────────

print(f"\n{'='*80}")
print("  OVERLAP CONSISTENCY CHECK (same dates, same close price?)")
print(f"{'='*80}")

mismatch_records = []
for ticker in in_both:
    df_o = old[ticker][["close"]].rename(columns={"close": "close_old"})
    df_n = new[ticker][["close"]].rename(columns={"close": "close_new"})
    merged = df_o.join(df_n, how="inner")
    if merged.empty:
        print(f"  {ticker}: NO overlapping dates!")
        continue
    diff = (merged["close_old"] - merged["close_new"]).abs()
    n_mismatch = (diff > 0.01).sum()
    pct_match  = 1 - n_mismatch / len(merged)
    mismatch_records.append({
        "ticker":      ticker,
        "overlap_days": len(merged),
        "mismatches":  n_mismatch,
        "pct_match":   pct_match,
        "max_diff":    diff.max(),
    })
    status = "[OK]" if n_mismatch == 0 else f"[WARN] {n_mismatch} mismatch days"
    print(f"  {ticker:<6}: overlap={len(merged):>4}d  match={pct_match:.1%}  "
          f"max_diff={diff.max():.4f}  {status}")

df_match = pd.DataFrame(mismatch_records)

# ── Chart 1: Days comparison bar chart ───────────────────────────────────────

all_tickers = sorted(old_tickers | new_tickers)
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(all_tickers))
w = 0.38

old_days = [df_cmp.loc[t, "old_days"] if t in df_cmp.index else 0 for t in all_tickers]
new_days = [df_cmp.loc[t, "new_days"] if t in df_cmp.index else 0 for t in all_tickers]

ax.bar(x - w/2, old_days, w, label="OLD (data/raw)",          color="#e74c3c", alpha=0.8)
ax.bar(x + w/2, new_days, w, label="NEW (raw_20260517)",       color="#2ecc71", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(all_tickers, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Trading Days")
ax.set_title("Chart 1 — Trading Days: OLD vs NEW dataset per stock",
             fontsize=13, fontweight="bold")
ax.legend()
ax.axhline(2500, color="blue", ls="--", lw=1, label="2500d reference")

# Annotate extra days
for i, t in enumerate(all_tickers):
    extra = new_days[i] - old_days[i]
    if extra > 0:
        ax.text(i + w/2, new_days[i] + 30, f"+{extra}",
                ha="center", fontsize=7, color="#27ae60")
    elif old_days[i] > 0 and new_days[i] == 0:
        ax.text(i - w/2, old_days[i] + 30, "MISS",
                ha="center", fontsize=7, color="red")

plt.tight_layout()
plt.savefig(OUT_DIR / "01_days_comparison.png")
plt.close()
print("\n[OK] Chart 1: Days comparison saved")

# ── Chart 2: Date range timeline ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 10))
all_t = sorted(old_tickers | new_tickers)

for i, t in enumerate(all_t):
    y_old = i * 2
    y_new = i * 2 + 0.6
    if t in old:
        df = old[t]
        ax.barh(y_old,
                (df.index[-1] - df.index[0]).days,
                left=mdates.date2num(df.index[0]),
                height=0.5, color="#e74c3c", alpha=0.75, label="OLD" if i == 0 else "")
    if t in new:
        df = new[t]
        ax.barh(y_new,
                (df.index[-1] - df.index[0]).days,
                left=mdates.date2num(df.index[0]),
                height=0.5, color="#2ecc71", alpha=0.75, label="NEW" if i == 0 else "")
    ax.text(mdates.date2num(pd.Timestamp("2026-06-01")), y_old + 0.3,
            t, va="center", fontsize=8)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.set_yticks([])
ax.set_title("Chart 2 — Date Range Timeline: OLD (red) vs NEW (green)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.axvline(mdates.date2num(pd.Timestamp("2020-01-01")),
           color="blue", ls="--", lw=1, label="COVID start")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_timeline_comparison.png")
plt.close()
print("[OK] Chart 2: Timeline comparison saved")

# ── Chart 3: Overlap consistency ──────────────────────────────────────────────

if not df_match.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_match = ["#2ecc71" if v == 1.0 else "#e74c3c"
                    for v in df_match["pct_match"]]
    axes[0].barh(df_match["ticker"], df_match["pct_match"] * 100,
                 color=colors_match, alpha=0.85)
    axes[0].axvline(100, color="green", ls="--", lw=1)
    axes[0].set_xlabel("% Matching close prices (|diff| < 0.01)")
    axes[0].set_title("Overlap Consistency per Stock")
    axes[0].set_xlim(90, 101)

    axes[1].barh(df_match["ticker"], df_match["overlap_days"],
                 color="#3498db", alpha=0.85)
    axes[1].set_xlabel("Overlapping Trading Days")
    axes[1].set_title("Overlap Duration per Stock")

    plt.suptitle("Chart 3 — Data Consistency in Overlapping Period",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_overlap_consistency.png")
    plt.close()
    print("[OK] Chart 3: Overlap consistency saved")

# ── Chart 4: Close price trace — OLD vs NEW for 4 stocks ─────────────────────

sample_stocks = [t for t in ["VCB", "HPG", "FPT", "MBB"] if t in in_both]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, ticker in enumerate(sample_stocks):
    ax = axes[i]
    df_o = old[ticker]["close"]
    df_n = new[ticker]["close"]
    ax.plot(df_n.index, df_n.values, lw=0.8, color="#2ecc71",
            alpha=0.9, label=f"NEW ({len(df_n)}d from {df_n.index[0].year})")
    ax.plot(df_o.index, df_o.values, lw=1.0, color="#e74c3c",
            alpha=0.8, ls="--", label=f"OLD ({len(df_o)}d from {df_o.index[0].year})")
    ax.axvline(df_o.index[0], color="#e74c3c", lw=1, ls=":", alpha=0.5,
               label="OLD start")
    ax.set_title(f"{ticker} — Close Price Trace")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.suptitle("Chart 4 — Close Price: OLD vs NEW (green=NEW longer history)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_price_trace.png")
plt.close()
print("[OK] Chart 4: Price trace saved")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  OLD dataset: {len(old_tickers)} stocks")
print(f"  NEW dataset: {len(new_tickers)} stocks")
print(f"  Stocks only in OLD: {only_old if only_old else 'None'}")
print(f"  Stocks only in NEW: {only_new}")
print(f"  Common stocks:      {len(in_both)}")

for_both = df_cmp[df_cmp["old_days"] > 0]
avg_extra = for_both["extra_days"].mean()
print(f"\n  Avg extra days (NEW vs OLD): +{avg_extra:.0f} days per stock")
print(f"  Max extra days: +{for_both['extra_days'].max():.0f} ({for_both['extra_days'].idxmax()})")

n_perfect = (df_match["pct_match"] == 1.0).sum() if not df_match.empty else 0
print(f"\n  Overlap consistency: {n_perfect}/{len(df_match)} stocks 100% match")
if not df_match.empty and (df_match["pct_match"] < 1.0).any():
    bad = df_match[df_match["pct_match"] < 1.0][["ticker","mismatches","max_diff"]]
    print(f"  Stocks with mismatch:\n{bad.to_string(index=False)}")

print(f"\n  RECOMMENDATION:")
print(f"  - NEW data has longer history -> better for LSTM (more ESS)")
print(f"  - New stocks: {only_new} -> consider adding to VN30 basket")
print(f"  - Verify mismatch stocks before switching datasets")
print(f"\n  Charts saved to: {OUT_DIR}/")
print(f"{'='*60}\n")
