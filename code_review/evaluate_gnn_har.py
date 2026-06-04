"""
Evaluate GNN+HAR results and generate comparison chart vs HAR-RV.

Loads results/gnn_har_results.csv (produced by gnn/train_gnn_har.py)
and lstm_har_results.csv for MIMO LSTM comparison.

Output:
  results/gnn_har_vs_harv_bar.png   -- avg R2 grouped bar: GNN+HAR vs HAR-RV vs LSTM MIMO

Usage:
  python scripts/eda/evaluate_gnn_har.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_root      = Path(__file__).parent.parent.parent
RESULTS    = _root / "results"
GNN_CSV    = RESULTS / "gnn_har_results.csv"
LSTM_CSV   = RESULTS / "lstm_har_results.csv"
HORIZONS   = [1, 5, 10, 20]


# ─────────────────────────── LOAD ─────────────────────────────────────────────
def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [WARN] {label} not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  {label}: {len(df)} stocks  <- {path.name}")
    return df


# ─────────────────────────── CONSOLE TABLE ────────────────────────────────────
def print_comparison(gnn_df: pd.DataFrame, lstm_df: pd.DataFrame | None):
    print(f"\n{'='*75}")
    print(f"  GNN+HAR vs HAR-RV vs LSTM HAR MIMO -- Avg R2 by Horizon")
    print(f"{'='*75}")
    hdr = f"  {'H':>4} | {'GNN+HAR':>10} | {'HAR-RV':>10} | {'LSTM MIMO':>10} | {'GNN-HAR delta':>14}"
    print(hdr)
    print(f"  {'-'*60}")
    for h in HORIZONS:
        gnn_col = f"gnn_h{h}_r2"
        har_col = f"har_h{h}_r2"
        lstm_col= f"lstm_h{h}_r2"

        gnn_avg  = gnn_df[gnn_col].mean()  if gnn_col  in gnn_df.columns else float("nan")
        har_avg  = gnn_df[har_col].mean()  if har_col  in gnn_df.columns else float("nan")
        lstm_avg = (lstm_df[lstm_col].mean() if (lstm_df is not None and
                    lstm_col in lstm_df.columns) else float("nan"))
        delta    = gnn_avg - har_avg
        flag     = "[+]" if delta > 0.01 else ("[-]" if delta < -0.01 else "[~]")

        print(f"  {h:>4} | {gnn_avg:>10.4f} | {har_avg:>10.4f} | "
              f"{lstm_avg:>10.4f} | {delta:>+10.4f}  {flag}")
    print(f"{'='*75}\n")


# ─────────────────────────── PER-STOCK TABLE ──────────────────────────────────
def print_per_stock(gnn_df: pd.DataFrame):
    print(f"\n  Per-stock GNN+HAR vs HAR-RV (avg R2 across 4 horizons):")
    print(f"  {'Ticker':<8} {'GNN h1':>8} {'GNN h5':>8} "
          f"{'GNN h10':>9} {'GNN h20':>9} {'HAR h20':>9}")
    print(f"  {'-'*56}")
    for _, row in gnn_df.iterrows():
        ticker = row.get("ticker", "")
        g1  = row.get("gnn_h1_r2",  float("nan"))
        g5  = row.get("gnn_h5_r2",  float("nan"))
        g10 = row.get("gnn_h10_r2", float("nan"))
        g20 = row.get("gnn_h20_r2", float("nan"))
        h20 = row.get("har_h20_r2", float("nan"))
        print(f"  {ticker:<8} {g1:>8.4f} {g5:>8.4f} {g10:>9.4f} {g20:>9.4f} {h20:>9.4f}")


# ─────────────────────────── BAR CHART ────────────────────────────────────────
def plot_bar(gnn_df: pd.DataFrame, lstm_df: pd.DataFrame | None, out_path: Path):
    models  = ["GNN+HAR", "HAR-RV", "LSTM MIMO"]
    colors  = {"GNN+HAR": "steelblue", "HAR-RV": "seagreen", "LSTM MIMO": "darkorange"}

    x   = np.arange(len(HORIZONS))
    w   = 0.25
    offsets = {"GNN+HAR": -w, "HAR-RV": 0, "LSTM MIMO": w}

    # Build avg R2 per model per horizon
    vals: dict[str, list] = {m: [] for m in models}
    for h in HORIZONS:
        gnn_col  = f"gnn_h{h}_r2"
        har_col  = f"har_h{h}_r2"
        lstm_col = f"lstm_h{h}_r2"

        vals["GNN+HAR"].append(
            gnn_df[gnn_col].mean() if gnn_col in gnn_df.columns else float("nan"))
        vals["HAR-RV"].append(
            gnn_df[har_col].mean() if har_col in gnn_df.columns else float("nan"))
        vals["LSTM MIMO"].append(
            lstm_df[lstm_col].mean() if (lstm_df is not None
                and lstm_col in lstm_df.columns) else float("nan"))

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        bars = ax.bar(x + offsets[m], vals[m], w, label=m,
                      color=colors[m], alpha=0.85)
        for bar, v in zip(bars, vals[m]):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("Avg R2 (test set, 2026-01-01 onwards)")
    ax.set_title(
        "GNN+HAR vs HAR-RV vs LSTM HAR MIMO -- Avg R2 by Horizon\n"
        "GNN+HAR above HAR-RV = cross-stock spillover adds value"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart saved: {out_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    print(f"\n{'='*75}")
    print(f"  GNN+HAR Evaluation  (Test set: 2026-01-01 onwards)")
    print(f"{'='*75}\n")

    gnn_df  = load_csv(GNN_CSV,  "GNN+HAR")
    lstm_df = load_csv(LSTM_CSV, "LSTM HAR MIMO")

    if gnn_df is None:
        print("\n  No GNN+HAR results. Run training first:")
        print("    python gnn/train_gnn_har.py\n")
        return

    print_comparison(gnn_df, lstm_df)
    print_per_stock(gnn_df)
    plot_bar(gnn_df, lstm_df, RESULTS / "gnn_har_vs_harv_bar.png")

    # Median R2 summary
    print(f"\n  Median R2 across stocks:")
    for h in HORIZONS:
        col = f"gnn_h{h}_r2"
        if col in gnn_df.columns:
            print(f"    GNN+HAR h={h}: med={gnn_df[col].median():+.4f}"
                  f"  avg={gnn_df[col].mean():+.4f}"
                  f"  > 0 stocks: {(gnn_df[col] > 0).sum()}/{len(gnn_df)}")

    print(f"\n{'='*75}")
    print(f"  DONE")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
