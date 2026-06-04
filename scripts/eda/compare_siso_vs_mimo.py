"""
Compare LSTM SISO (h=1, h=5, h=10) vs LSTM MIMO vs HAR-RV.

Tests MIMO gradient interference hypothesis:
  SISO vs MIMO gap -> interference
  SISO vs HAR-RV gap -> ESS bottleneck

Sources:
  SISO:    results/lstm_siso_h{H}_results.csv  (lstm_h{H}_r2, har_h{H}_r2)
  MIMO:    results/lstm_har_results.csv         (lstm_h{H}_r2, har_h{H}_r2)

Output:
  results/siso_vs_mimo_comparison.csv
  results/siso_vs_mimo_scatter.png    (3-panel scatter per horizon)
  results/siso_vs_mimo_bar.png        (avg R2 grouped bar)

Usage:
  python scripts/eda/compare_siso_vs_mimo.py
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
HORIZONS   = [1, 5, 10]   # SISO horizons only (not 20)
MIMO_CSV   = RESULTS / "lstm_har_results.csv"


# ─────────────────────────── LOAD DATA ────────────────────────────────────────
def load_mimo() -> pd.DataFrame | None:
    if not MIMO_CSV.exists():
        print(f"  [WARN] MIMO results not found: {MIMO_CSV}")
        print(f"  Run first: python scripts/eda/evaluate_lstm_har_features.py")
        return None
    df = pd.read_csv(MIMO_CSV)
    print(f"  MIMO: {len(df)} stocks  <- {MIMO_CSV.name}")
    return df


def load_siso(h: int) -> pd.DataFrame | None:
    path = RESULTS / f"lstm_siso_h{h}_results.csv"
    if not path.exists():
        print(f"  [WARN] SISO h={h} results not found: {path}")
        print(f"  Run first: python baselines/train_lstm_siso.py --horizon {h} --all")
        return None
    df = pd.read_csv(path)
    print(f"  SISO h={h}: {len(df)} stocks  <- {path.name}")
    return df


# ─────────────────────────── CONSOLE TABLE ────────────────────────────────────
def print_summary(summary: list[dict]):
    """Print avg R2 and MAE comparison per horizon."""
    print(f"\n{'='*70}")
    print(f"  SISO vs MIMO vs HAR-RV -- Avg R2 on test set ({len(summary)} horizons)")
    print(f"{'='*70}")
    hdr = f"  {'H':>4} | {'Model':<14} | {'Avg R2':>8} | {'Avg MAE':>9} | {'N stocks':>8}"
    print(hdr)
    print(f"  {'-'*60}")
    for row in summary:
        print(f"  {row['h']:>4} | {row['model']:<14} | "
              f"{row['avg_r2']:>8.4f} | {row['avg_mae']:>9.5f} | "
              f"{row['n']:>8}")
    print(f"{'='*70}\n")


# ─────────────────────────── SCATTER PLOT ─────────────────────────────────────
def plot_scatter(merged_h: dict[int, pd.DataFrame], out_path: Path):
    """3-panel scatter: SISO R2 vs MIMO R2 per horizon, diagonal = no improvement."""
    avail_h = [h for h in HORIZONS if h in merged_h]
    if not avail_h:
        print("  [WARN] No data for scatter plot")
        return

    n_panels = len(avail_h)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, h in zip(axes, avail_h):
        df = merged_h[h]
        xv = df["mimo_r2"].values.astype(float)
        yv = df["siso_r2"].values.astype(float)
        mask = ~(np.isnan(xv) | np.isnan(yv))
        xv, yv = xv[mask], yv[mask]
        lbls = df["ticker"].values[mask]

        ax.scatter(xv, yv, color="steelblue", s=60, zorder=5)
        for xi, yi, lbl in zip(xv, yv, lbls):
            ax.annotate(lbl, (xi, yi), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", color="gray")

        lo = min(np.nanmin(xv), np.nanmin(yv)) - 0.05
        hi = max(np.nanmax(xv), np.nanmax(yv)) + 0.05
        ax.plot([lo, hi], [lo, hi], color="red", lw=1.0, ls="--", label="SISO=MIMO")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.axvline(0, color="black", lw=0.5, ls=":")

        wins = int(np.sum(yv > xv))
        ax.set_xlabel(f"MIMO h={h} R2")
        ax.set_ylabel(f"SISO h={h} R2")
        ax.set_title(f"h={h}: SISO vs MIMO  ({wins}/{len(xv)} SISO wins)\n"
                     f"above diagonal = SISO better")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "SISO vs MIMO -- R2 per Stock and Horizon\n"
        "SISO better above diagonal; below = MIMO interference NOT the cause",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved: {out_path}")


# ─────────────────────────── BAR PLOT ─────────────────────────────────────────
def plot_bar(summary: list[dict], out_path: Path):
    """Grouped bar: avg R2 for SISO, MIMO, HAR-RV per horizon."""
    avail_h = sorted(set(r["h"] for r in summary))
    models   = ["SISO", "MIMO", "HAR-RV"]
    colors   = {"SISO": "steelblue", "MIMO": "darkorange", "HAR-RV": "seagreen"}

    x   = np.arange(len(avail_h))
    w   = 0.25
    offsets = {"SISO": -w, "MIMO": 0, "HAR-RV": w}

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        vals = []
        for h in avail_h:
            row = next((r for r in summary if r["h"] == h and r["model"] == m), None)
            vals.append(row["avg_r2"] if row else float("nan"))
        ax.bar(x + offsets[m], vals, w, label=m,
               color=colors[m], alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in avail_h])
    ax.set_ylabel("Avg R2")
    ax.set_title("SISO vs MIMO vs HAR-RV -- Avg R2 by Horizon\n"
                 "(SISO vs MIMO gap = MIMO interference; both vs HAR-RV gap = ESS bottleneck)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Bar plot saved: {out_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    print(f"\n{'='*70}")
    print(f"  SISO vs MIMO Comparison  (Ablation: MIMO interference hypothesis)")
    print(f"  Horizons: {HORIZONS}  |  Test set: 2026-01-01 onwards")
    print(f"{'='*70}\n")

    mimo_df = load_mimo()

    summary:  list[dict]             = []
    merged_h: dict[int, pd.DataFrame] = {}
    all_rows: list[dict]             = []

    for h in HORIZONS:
        siso_df = load_siso(h)
        if siso_df is None:
            continue

        # SISO metrics
        siso_r2_col  = f"lstm_h{h}_r2"
        siso_mae_col = f"lstm_h{h}_mae"
        if siso_r2_col not in siso_df.columns:
            print(f"  [WARN] Column {siso_r2_col} not in SISO results -- skip h={h}")
            continue

        siso_vals = siso_df[["ticker", siso_r2_col, siso_mae_col]].copy()
        siso_vals.columns = ["ticker", "siso_r2", "siso_mae"]

        summary.append({
            "h": h, "model": "SISO",
            "avg_r2":  siso_vals["siso_r2"].mean(),
            "avg_mae": siso_vals["siso_mae"].mean(),
            "n":       len(siso_vals),
        })

        # MIMO metrics
        if mimo_df is not None:
            mimo_r2_col  = f"lstm_h{h}_r2"
            mimo_mae_col = f"lstm_h{h}_mae"
            har_r2_col   = f"har_h{h}_r2"
            har_mae_col  = f"har_h{h}_mae"

            if mimo_r2_col in mimo_df.columns:
                summary.append({
                    "h": h, "model": "MIMO",
                    "avg_r2":  mimo_df[mimo_r2_col].mean(),
                    "avg_mae": mimo_df[mimo_mae_col].mean() if mimo_mae_col in mimo_df.columns else float("nan"),
                    "n":       len(mimo_df),
                })

            if har_r2_col in mimo_df.columns:
                summary.append({
                    "h": h, "model": "HAR-RV",
                    "avg_r2":  mimo_df[har_r2_col].mean(),
                    "avg_mae": mimo_df[har_mae_col].mean() if har_mae_col in mimo_df.columns else float("nan"),
                    "n":       len(mimo_df),
                })

            # merged for scatter
            mimo_slice = mimo_df[["ticker", mimo_r2_col]].copy() if mimo_r2_col in mimo_df.columns else None
            if mimo_slice is not None:
                mimo_slice.columns = ["ticker", "mimo_r2"]
                merged = siso_vals.merge(mimo_slice, on="ticker", how="inner")
                merged_h[h] = merged
                print(f"  h={h}: {len(merged)} common stocks for scatter")

        # per-stock comparison rows
        har_r2_col_siso = f"har_h{h}_r2"
        for _, row_s in siso_vals.iterrows():
            r = {
                "ticker":   row_s["ticker"],
                "h":        h,
                "siso_r2":  row_s["siso_r2"],
                "siso_mae": row_s["siso_mae"],
            }
            if mimo_df is not None and f"lstm_h{h}_r2" in mimo_df.columns:
                mimo_row = mimo_df[mimo_df["ticker"] == row_s["ticker"]]
                if len(mimo_row) > 0:
                    r["mimo_r2"]  = float(mimo_row[f"lstm_h{h}_r2"].iloc[0])
                    r["mimo_mae"] = float(mimo_row[f"lstm_h{h}_mae"].iloc[0]) if f"lstm_h{h}_mae" in mimo_row.columns else float("nan")
                    if f"har_h{h}_r2" in mimo_row.columns:
                        r["har_r2"]  = float(mimo_row[f"har_h{h}_r2"].iloc[0])
                        r["har_mae"] = float(mimo_row[f"har_h{h}_mae"].iloc[0]) if f"har_h{h}_mae" in mimo_row.columns else float("nan")
            if har_r2_col_siso in siso_df.columns:
                siso_row = siso_df[siso_df["ticker"] == row_s["ticker"]]
                if len(siso_row) > 0 and "har_r2" not in r:
                    r["har_r2"] = float(siso_row[har_r2_col_siso].iloc[0])
            all_rows.append(r)

    if not summary:
        print("\n  No results available. Run SISO training first:\n"
              "    python baselines/train_lstm_siso.py --horizon 1 --all\n"
              "    python baselines/train_lstm_siso.py --horizon 5 --all\n"
              "    python baselines/train_lstm_siso.py --horizon 10 --all\n")
        return

    # console summary table
    print_summary(summary)

    # SISO improvement vs MIMO per horizon
    print(f"  SISO improvement over MIMO (avg R2 delta):")
    for h in HORIZONS:
        siso_row = next((r for r in summary if r["h"] == h and r["model"] == "SISO"),  None)
        mimo_row = next((r for r in summary if r["h"] == h and r["model"] == "MIMO"),  None)
        har_row  = next((r for r in summary if r["h"] == h and r["model"] == "HAR-RV"), None)
        if siso_row and mimo_row:
            delta = siso_row["avg_r2"] - mimo_row["avg_r2"]
            flag  = "[+] SISO better" if delta > 0.01 else ("[-] MIMO better" if delta < -0.01 else "[~] negligible")
            print(f"    h={h}: SISO={siso_row['avg_r2']:+.4f}  "
                  f"MIMO={mimo_row['avg_r2']:+.4f}  "
                  f"delta={delta:+.4f}  {flag}")
            if har_row:
                gap = har_row["avg_r2"] - siso_row["avg_r2"]
                print(f"         HAR-RV={har_row['avg_r2']:+.4f}  gap(HAR-SISO)={gap:+.4f}  <- ESS bottleneck")

    print()

    # save CSV
    if all_rows:
        out_csv = RESULTS / "siso_vs_mimo_comparison.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"  Comparison CSV saved: {out_csv}")

    # plots
    if merged_h:
        plot_scatter(merged_h, RESULTS / "siso_vs_mimo_scatter.png")
    if summary:
        plot_bar(summary, RESULTS / "siso_vs_mimo_bar.png")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
