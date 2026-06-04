"""
Master comparison chart: all models x all horizons.

Reads available results CSVs and produces:
  results/master_comparison.png  -- bar chart, all models x h=1,5,10,20
  results/master_comparison.csv  -- summary table

Sources:
  GNN+HAR SISO + HAR-RV:  results/gnn_har_siso_results.csv  (long, 120 rows)
  LSTM MIMO:               results/lstm_har_results.csv
  LSTM Pooled h5:          results/lstm_siso_pooled_h5_results.csv

Usage:
  python scripts/eda/plot_master_comparison.py
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

_root    = Path(__file__).parent.parent.parent
RESULTS  = _root / "results"
HORIZONS = [1, 5, 10, 20]


def _load(path, hint=""):
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        print(f"  [OK] {p.name}  ({len(df)} rows)")
        return df
    msg = f"  [--] {p.name}: not found"
    if hint:
        msg += f"  -- run: {hint}"
    print(msg)
    return None


def main():
    print(f"\n{'='*70}")
    print(f"  Master Comparison -- All Models x All Horizons")
    print(f"  Test period: 2026-01-01 onwards")
    print(f"{'='*70}\n")

    models_data = {}   # model_name -> {h: avg_r2}

    # ── GNN+HAR SISO + HAR-RV (long format: ticker, h, gnn_r2, har_r2) ────────
    gnn_siso = _load(RESULTS / "gnn_har_siso_results.csv", "python gnn/train_gnn_har.py")
    if gnn_siso is not None:
        har_r2 = {}
        gnn_r2 = {}
        for h in HORIZONS:
            sub = gnn_siso[gnn_siso["h"] == h]
            if len(sub) > 0:
                har_r2[h] = float(sub["har_r2"].mean())
                gnn_r2[h] = float(sub["gnn_r2"].mean())
        models_data["HAR-RV"] = har_r2
        models_data["GNN+HAR SISO"] = gnn_r2

    # ── LSTM MIMO (wide format: one row per stock, lstm_h{h}_r2 per horizon) ──
    lstm = _load(RESULTS / "lstm_har_results.csv",
                 "python scripts/eda/evaluate_lstm_har_features.py")
    if lstm is not None:
        lstm_d = {}
        for h in HORIZONS:
            col = f"lstm_h{h}_r2"
            if col in lstm.columns:
                lstm_d[h] = float(lstm[col].mean())
        if lstm_d:
            models_data["LSTM MIMO"] = lstm_d

    # ── LSTM Pooled SISO h=5 only ──────────────────────────────────────────────
    pooled_h5 = _load(RESULTS / "lstm_siso_pooled_h5_results.csv")
    if pooled_h5 is not None and "lstm_pooled_h5_r2" in pooled_h5.columns:
        models_data["LSTM Pooled"] = {5: float(pooled_h5["lstm_pooled_h5_r2"].mean())}

    # ── Console table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Avg R2 by Model x Horizon")
    print(f"{'='*70}")
    header = f"  {'Model':<20}" + "".join(f"  {'h='+str(h):>8}" for h in HORIZONS)
    print(header)
    print("  " + "-" * 58)
    for m, vals in models_data.items():
        row = f"  {m:<20}"
        for h in HORIZONS:
            v = vals.get(h, float("nan"))
            row += f"  {v:>+8.4f}" if not np.isnan(v) else f"  {'---':>8}"
        print(row)
    print(f"{'='*70}\n")

    if not models_data:
        print("  [WARN] No results loaded. Run training scripts first.")
        return

    # ── Stock-count wins (GNN+HAR SISO only) ──────────────────────────────────
    if gnn_siso is not None:
        print("  GNN+HAR SISO beats HAR-RV (per stock):")
        for h in HORIZONS:
            sub = gnn_siso[gnn_siso["h"] == h]
            if len(sub) > 0:
                wins = int((sub["delta_r2"] > 0).sum())
                print(f"    h={h}: {wins}/{len(sub)} stocks")
        print()

    # ── Bar chart ──────────────────────────────────────────────────────────────
    n_models = len(models_data)
    x        = np.arange(len(HORIZONS))
    w        = 0.8 / n_models
    offsets  = np.linspace(-(n_models-1)*w/2, (n_models-1)*w/2, n_models)
    palette  = ["seagreen", "steelblue", "darkorange", "crimson", "purple"]
    model_list = list(models_data.keys())

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, m in enumerate(model_list):
        vals = [models_data[m].get(h, float("nan")) for h in HORIZONS]
        bars = ax.bar(x + offsets[i], vals, w, label=m,
                      color=palette[i % len(palette)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ypos = bar.get_height()
                va   = "bottom" if ypos >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        ypos + (0.01 if ypos >= 0 else -0.01),
                        f"{v:.3f}", ha="center", va=va, fontsize=7)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("Avg R2 across 30 VN30 stocks (test 2026-01-01+)")
    ax.set_title(
        "VN30 Realized Volatility Forecasting -- All Models x All Horizons\n"
        "R2 > 0: beats naive mean; HAR-RV = OLS baseline"
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out_png = RESULTS / "master_comparison.png"
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Chart -> {out_png}")

    # ── CSV summary ────────────────────────────────────────────────────────────
    rows = []
    for m, vals in models_data.items():
        row = {"model": m}
        for h in HORIZONS:
            row[f"h{h}_r2"] = vals.get(h, float("nan"))
        rows.append(row)
    df_out = pd.DataFrame(rows)
    out_csv = RESULTS / "master_comparison.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  Table -> {out_csv}")

    print(f"\n  DONE\n")


if __name__ == "__main__":
    main()
