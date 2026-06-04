"""
3-way comparison: LSTM (abs_ret input) vs LSTM (HAR features input) vs HAR-RV.

Primary thesis output. Loads both LSTM result CSVs and produces:
  - Console: comparison table avg R2/MAE per horizon
  - results/lstm_comparison_r2.png    -- grouped bar chart, 4 panels (1 per horizon)
  - results/lstm_comparison_table.csv -- 3-model x 4-horizon R2/MAE/RMSE avg

Inputs (must exist):
  results/lstm_h20_results.csv    -- from evaluate_lstm_h20.py (column prefix: lstm_hX)
  results/lstm_har_results.csv    -- from evaluate_lstm_har_features.py (column prefix: lstm_hX)
  (HAR-RV metrics are embedded in both CSVs as har_hX columns)

Usage:
  python scripts/eda/compare_lstm_variants.py
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

HORIZONS = [1, 5, 10, 20]

_root    = Path(__file__).parent.parent.parent
RES_DIR  = _root / "results"

CSV_ABSRET  = RES_DIR / "lstm_h20_results.csv"       # LSTM abs_ret input
CSV_HAR     = RES_DIR / "lstm_har_results.csv"        # LSTM HAR features input
CSV_OUT     = RES_DIR / "lstm_comparison_table.csv"
PLOT_OUT    = RES_DIR / "lstm_comparison_r2.png"


# ─────────────────────────── LOAD + VALIDATE ──────────────────────────────────
def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [CSV_ABSRET, CSV_HAR]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"  Run the corresponding evaluation script first."
            )
    df_abs = pd.read_csv(CSV_ABSRET)
    df_har = pd.read_csv(CSV_HAR)
    print(f"  LSTM abs_ret : {len(df_abs)} stocks  ({CSV_ABSRET.name})")
    print(f"  LSTM HAR     : {len(df_har)} stocks  ({CSV_HAR.name})")
    return df_abs, df_har


# ─────────────────────────── AGGREGATE HELPERS ────────────────────────────────
def avg_metric(df: pd.DataFrame, col: str) -> float:
    """Return mean of column, NaN if column missing."""
    if col not in df.columns:
        return float("nan")
    return float(df[col].mean())


def build_comparison_table(
    df_abs: pd.DataFrame,
    df_har: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build 3-model x (4-horizon x 3-metric) summary table.
    Rows: LSTM_absret, LSTM_HAR, HAR_RV
    Columns: h1_r2, h1_mae, h1_rmse, h5_r2, ...
    """
    models = ["LSTM (abs_ret)", "LSTM (HAR feats)", "HAR-RV"]
    rows   = []

    for model in models:
        row = {"model": model}
        for h in HORIZONS:
            if model == "LSTM (abs_ret)":
                row[f"h{h}_r2"]   = avg_metric(df_abs, f"lstm_h{h}_r2")
                row[f"h{h}_mae"]  = avg_metric(df_abs, f"lstm_h{h}_mae")
                row[f"h{h}_rmse"] = avg_metric(df_abs, f"lstm_h{h}_rmse")
            elif model == "LSTM (HAR feats)":
                row[f"h{h}_r2"]   = avg_metric(df_har, f"lstm_h{h}_r2")
                row[f"h{h}_mae"]  = avg_metric(df_har, f"lstm_h{h}_mae")
                row[f"h{h}_rmse"] = avg_metric(df_har, f"lstm_h{h}_rmse")
            else:  # HAR-RV -- use har_hX columns from either CSV
                # prefer df_har (more recent run) but fall back to df_abs
                r2   = avg_metric(df_har, f"har_h{h}_r2")
                mae  = avg_metric(df_har, f"har_h{h}_mae")
                rmse = avg_metric(df_har, f"har_h{h}_rmse")
                if np.isnan(r2):
                    r2   = avg_metric(df_abs, f"har_h{h}_r2")
                    mae  = avg_metric(df_abs, f"har_h{h}_mae")
                    rmse = avg_metric(df_abs, f"har_h{h}_rmse")
                row[f"h{h}_r2"]   = r2
                row[f"h{h}_mae"]  = mae
                row[f"h{h}_rmse"] = rmse
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────── CONSOLE TABLE ────────────────────────────────────
def print_comparison(cmp: pd.DataFrame):
    """Print readable 3-model x 4-horizon R2 comparison table."""
    print(f"\n{'='*72}")
    print(f"  3-WAY COMPARISON: avg R2 per horizon")
    print(f"{'='*72}")
    header = f"  {'Model':<22}" + "".join(f"  h={h} R2  " for h in HORIZONS)
    print(header)
    print(f"  {'-'*22}" + "".join(f"  {'----------'}" for _ in HORIZONS))
    for _, r in cmp.iterrows():
        vals = "".join(f"  {r[f'h{h}_r2']:>+9.4f}" for h in HORIZONS)
        print(f"  {r['model']:<22}{vals}")
    print(f"{'='*72}")

    print(f"\n  avg MAE per horizon:")
    header2 = f"  {'Model':<22}" + "".join(f"  h={h} MAE " for h in HORIZONS)
    print(header2)
    print(f"  {'-'*22}" + "".join(f"  {'----------'}" for _ in HORIZONS))
    for _, r in cmp.iterrows():
        vals = "".join(f"  {r[f'h{h}_mae']:>10.5f}" for h in HORIZONS)
        print(f"  {r['model']:<22}{vals}")
    print(f"{'='*72}\n")


# ─────────────────────────── PLOT ─────────────────────────────────────────────
def plot_r2_comparison(cmp: pd.DataFrame, out_path: Path):
    """
    4-panel grouped bar chart (1 panel per horizon).
    3 bars per horizon: LSTM abs_ret, LSTM HAR, HAR-RV.
    """
    colors = {
        "LSTM (abs_ret)":  "#2196F3",
        "LSTM (HAR feats)": "#4CAF50",
        "HAR-RV":           "#FF9800",
    }
    model_labels = cmp["model"].tolist()
    x = np.arange(len(model_labels))
    bar_w = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, h in enumerate(HORIZONS):
        ax     = axes[idx // 2, idx % 2]
        r2_vals = [float(cmp.loc[cmp["model"] == m, f"h{h}_r2"].values[0])
                   for m in model_labels]
        bar_colors = [colors[m] for m in model_labels]

        bars = ax.bar(x, r2_vals, width=bar_w, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        # value labels on bars
        for bar, val in zip(bars, r2_vals):
            ypos = bar.get_height() if val >= 0 else bar.get_height() - 0.02
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    ypos + 0.01 * np.sign(val),
                    f"{val:+.3f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", lw=0.9, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylabel("Avg R2 (test set)")
        ax.set_title(f"h={h}: avg R2 across all stocks")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(min(min(r2_vals) - 0.15, -0.1), max(max(r2_vals) + 0.12, 0.1))

    plt.suptitle(
        "LSTM Variant Comparison: abs_ret input vs HAR-features input vs HAR-RV\n"
        "Thesis ablation -- avg R2 on test set (2026-01-01 onwards)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved: {out_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print(f"  LSTM Variant Comparison (3-way)")
    print(f"  LSTM (abs_ret) vs LSTM (HAR feats) vs HAR-RV")
    print(f"  Horizons: {HORIZONS}  |  Primary metric: h=20 R2")
    print(f"{'='*65}\n")

    df_abs, df_har = load_results()

    # ── common tickers (intersection) ────────────────────────────────────────
    tickers_abs = set(df_abs["ticker"])
    tickers_har = set(df_har["ticker"])
    common      = tickers_abs & tickers_har

    if common != tickers_abs or common != tickers_har:
        only_abs = tickers_abs - common
        only_har = tickers_har - common
        if only_abs:
            print(f"  [WARN] In abs_ret only: {sorted(only_abs)}")
        if only_har:
            print(f"  [WARN] In HAR only:     {sorted(only_har)}")
        print(f"  Using {len(common)} common stocks for comparison")

    df_abs = df_abs[df_abs["ticker"].isin(common)].copy()
    df_har = df_har[df_har["ticker"].isin(common)].copy()

    cmp = build_comparison_table(df_abs, df_har)
    print_comparison(cmp)

    # ── save comparison CSV ───────────────────────────────────────────────────
    cmp.to_csv(CSV_OUT, index=False)
    print(f"  Comparison table saved: {CSV_OUT}")

    # ── per-stock delta: LSTM-HAR vs LSTM-abs at h=20 ────────────────────────
    merged = df_abs[["ticker", "lstm_h20_r2"]].rename(
        columns={"lstm_h20_r2": "r2_absret"}
    ).merge(
        df_har[["ticker", "lstm_h20_r2"]].rename(
            columns={"lstm_h20_r2": "r2_har"}
        ),
        on="ticker", how="inner"
    )
    if len(merged) > 0:
        delta = merged["r2_har"] - merged["r2_absret"]
        print(f"\n  Per-stock delta (LSTM-HAR minus LSTM-abs) at h=20:")
        print(f"    avg delta : {delta.mean():+.4f}")
        print(f"    max delta : {delta.max():+.4f}  (most improved)")
        print(f"    min delta : {delta.min():+.4f}  (least improved)")
        wins = int((delta > 0).sum())
        print(f"    LSTM-HAR improved: {wins}/{len(merged)} stocks")

    # ── key thesis metrics ────────────────────────────────────────────────────
    print(f"\n  KEY THESIS METRICS (h=20 R2):")
    for _, r in cmp.iterrows():
        print(f"    {r['model']:<22}: {r['h20_r2']:+.4f}")

    plot_r2_comparison(cmp, PLOT_OUT)

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"  Table: {CSV_OUT}")
    print(f"  Plot:  {PLOT_OUT}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
