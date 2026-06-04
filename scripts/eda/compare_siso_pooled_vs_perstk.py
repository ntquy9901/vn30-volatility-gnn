"""
Compare LSTM SISO Pooled vs Per-stock SISO vs MIMO vs HAR-RV.

Tests:
  Pooled >> Per-stock SISO  -> ESS bottleneck; cross-stock transfer works
  Pooled ~= Per-stock SISO  -> Distribution heterogeneity blocks transfer
  Pooled <  Per-stock SISO  -> Cross-stock noise hurts

Sources:
  Pooled:      results/lstm_siso_pooled_h{H}_results.csv  (lstm_pooled_h{H}_r2, har_h{H}_r2)
  Per-stock:   results/lstm_siso_h{H}_results.csv          (lstm_h{H}_r2, har_h{H}_r2)
  MIMO:        results/lstm_har_results.csv                 (lstm_h{H}_r2, har_h{H}_r2)

Output:
  results/siso_pooled_comparison.csv
  results/siso_pooled_bar.png    (avg R2 grouped bar: 4 models x 3 horizons)

Usage:
  python scripts/eda/compare_siso_pooled_vs_perstk.py
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
HORIZONS   = [1, 5, 10]
MIMO_CSV   = RESULTS / "lstm_har_results.csv"


# ─────────────────────────── LOAD DATA ────────────────────────────────────────
def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [WARN] {label} not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  {label}: {len(df)} stocks  <- {path.name}")
    return df


# ─────────────────────────── CONSOLE TABLE ────────────────────────────────────
def print_summary(summary: list[dict]):
    print(f"\n{'='*75}")
    print(f"  SISO Pooled vs Per-stock vs MIMO vs HAR-RV  -- Avg R2")
    print(f"{'='*75}")
    hdr = f"  {'H':>4} | {'Model':<18} | {'Avg R2':>8} | {'Med R2':>8} | {'Avg MAE':>9} | {'N':>5}"
    print(hdr)
    print(f"  {'-'*65}")
    for row in summary:
        med_str = f"{row['med_r2']:>8.4f}" if not np.isnan(row.get("med_r2", float("nan"))) else f"{'--':>8}"
        print(f"  {row['h']:>4} | {row['model']:<18} | "
              f"{row['avg_r2']:>8.4f} | {med_str} | "
              f"{row['avg_mae']:>9.5f} | {row['n']:>5}")
    print(f"{'='*75}\n")


# ─────────────────────────── BAR PLOT ─────────────────────────────────────────
def plot_bar(summary: list[dict], out_path: Path):
    avail_h = sorted(set(r["h"] for r in summary))
    models  = ["Pooled SISO", "Per-stock SISO", "MIMO", "HAR-RV"]
    colors  = {
        "Pooled SISO":    "steelblue",
        "Per-stock SISO": "cornflowerblue",
        "MIMO":           "darkorange",
        "HAR-RV":         "seagreen",
    }

    x   = np.arange(len(avail_h))
    w   = 0.20
    offsets = {
        "Pooled SISO":    -1.5 * w,
        "Per-stock SISO": -0.5 * w,
        "MIMO":            0.5 * w,
        "HAR-RV":          1.5 * w,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        vals = []
        for h in avail_h:
            row = next((r for r in summary if r["h"] == h and r["model"] == m), None)
            vals.append(row["avg_r2"] if row else float("nan"))
        bars = ax.bar(x + offsets[m], vals, w, label=m,
                      color=colors[m], alpha=0.85)
        # value labels
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in avail_h])
    ax.set_ylabel("Avg R2 (test set)")
    ax.set_title(
        "SISO Pooled vs Per-stock SISO vs MIMO vs HAR-RV -- Avg R2\n"
        "Pooled vs Per-stock gap = ESS bottleneck effect; "
        "all LSTM vs HAR-RV = structural/information gap"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Bar plot saved: {out_path}")


# ─────────────────────────── SCATTER PLOT (OPTIONAL) ─────────────────────────
def plot_scatter_pooled_vs_perstk(merged_h: dict, out_path: Path):
    avail_h = [h for h in HORIZONS if h in merged_h]
    if not avail_h:
        return
    n_panels = len(avail_h)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, h in zip(axes, avail_h):
        df = merged_h[h]
        xv = df["perstk_r2"].values.astype(float)
        yv = df["pooled_r2"].values.astype(float)
        mask = ~(np.isnan(xv) | np.isnan(yv))
        xv, yv = xv[mask], yv[mask]
        lbls = df["ticker"].values[mask]

        ax.scatter(xv, yv, color="steelblue", s=60, zorder=5)
        for xi, yi, lbl in zip(xv, yv, lbls):
            ax.annotate(lbl, (xi, yi), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", color="gray")

        lo = min(np.nanmin(xv), np.nanmin(yv)) - 0.05
        hi = max(np.nanmax(xv), np.nanmax(yv)) + 0.05
        ax.plot([lo, hi], [lo, hi], color="red", lw=1.0, ls="--", label="Pooled=Per-stock")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.axvline(0, color="black", lw=0.5, ls=":")

        wins = int(np.sum(yv > xv))
        ax.set_xlabel(f"Per-stock SISO h={h} R2")
        ax.set_ylabel(f"Pooled SISO h={h} R2")
        ax.set_title(f"h={h}: Pooled vs Per-stock  ({wins}/{len(xv)} Pooled wins)\n"
                     "above diagonal = pooled better")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Pooled vs Per-stock SISO -- R2 per Stock and Horizon\n"
        "above diagonal = pooled ESS helps; below = heterogeneity blocks transfer",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved: {out_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    print(f"\n{'='*75}")
    print(f"  SISO Pooled vs Per-stock SISO vs MIMO vs HAR-RV  (ESS bottleneck test)")
    print(f"  Horizons: {HORIZONS}  |  Test set: 2026-01-01 onwards")
    print(f"{'='*75}\n")

    mimo_df = load_csv(MIMO_CSV, "MIMO")

    summary:  list[dict]              = []
    merged_h: dict[int, pd.DataFrame] = {}
    all_rows: list[dict]              = []

    for h in HORIZONS:
        pooled_df  = load_csv(RESULTS / f"lstm_siso_pooled_h{h}_results.csv",
                              f"Pooled h={h}")
        perstk_df  = load_csv(RESULTS / f"lstm_siso_h{h}_results.csv",
                              f"Per-stock h={h}")

        if pooled_df is not None:
            col_r2  = f"lstm_pooled_h{h}_r2"
            col_mae = f"lstm_pooled_h{h}_mae"
            if col_r2 in pooled_df.columns:
                summary.append({
                    "h": h, "model": "Pooled SISO",
                    "avg_r2":  pooled_df[col_r2].mean(),
                    "med_r2":  pooled_df[col_r2].median(),
                    "avg_mae": pooled_df[col_mae].mean() if col_mae in pooled_df.columns else float("nan"),
                    "n":       len(pooled_df),
                })

        if perstk_df is not None:
            col_r2  = f"lstm_h{h}_r2"
            col_mae = f"lstm_h{h}_mae"
            if col_r2 in perstk_df.columns:
                summary.append({
                    "h": h, "model": "Per-stock SISO",
                    "avg_r2":  perstk_df[col_r2].mean(),
                    "med_r2":  perstk_df[col_r2].median(),
                    "avg_mae": perstk_df[col_mae].mean() if col_mae in perstk_df.columns else float("nan"),
                    "n":       len(perstk_df),
                })

        if mimo_df is not None:
            col_r2  = f"lstm_h{h}_r2"
            col_mae = f"lstm_h{h}_mae"
            col_hr2 = f"har_h{h}_r2"
            col_hm  = f"har_h{h}_mae"
            if col_r2 in mimo_df.columns:
                summary.append({
                    "h": h, "model": "MIMO",
                    "avg_r2":  mimo_df[col_r2].mean(),
                    "med_r2":  mimo_df[col_r2].median(),
                    "avg_mae": mimo_df[col_mae].mean() if col_mae in mimo_df.columns else float("nan"),
                    "n":       len(mimo_df),
                })
            if col_hr2 in mimo_df.columns:
                summary.append({
                    "h": h, "model": "HAR-RV",
                    "avg_r2":  mimo_df[col_hr2].mean(),
                    "med_r2":  mimo_df[col_hr2].median(),
                    "avg_mae": mimo_df[col_hm].mean() if col_hm in mimo_df.columns else float("nan"),
                    "n":       len(mimo_df),
                })

        # scatter data: pooled vs per-stock
        if pooled_df is not None and perstk_df is not None:
            pc = f"lstm_pooled_h{h}_r2"
            sc = f"lstm_h{h}_r2"
            if pc in pooled_df.columns and sc in perstk_df.columns:
                merged = (
                    pooled_df[["ticker", pc]].rename(columns={pc: "pooled_r2"})
                    .merge(
                        perstk_df[["ticker", sc]].rename(columns={sc: "perstk_r2"}),
                        on="ticker", how="inner"
                    )
                )
                merged_h[h] = merged
                print(f"  h={h}: {len(merged)} common stocks for scatter")

        # per-stock comparison rows
        if pooled_df is not None:
            pcol = f"lstm_pooled_h{h}_r2"
            for _, rp in pooled_df.iterrows():
                r: dict = {
                    "ticker":     rp["ticker"],
                    "h":          h,
                    "pooled_r2":  rp.get(pcol, float("nan")),
                }
                if perstk_df is not None and f"lstm_h{h}_r2" in perstk_df.columns:
                    ps_row = perstk_df[perstk_df["ticker"] == rp["ticker"]]
                    if len(ps_row) > 0:
                        r["perstk_r2"] = float(ps_row[f"lstm_h{h}_r2"].iloc[0])
                if mimo_df is not None and f"har_h{h}_r2" in mimo_df.columns:
                    mo_row = mimo_df[mimo_df["ticker"] == rp["ticker"]]
                    if len(mo_row) > 0:
                        r["har_r2"]  = float(mo_row[f"har_h{h}_r2"].iloc[0])
                        r["mimo_r2"] = float(mo_row[f"lstm_h{h}_r2"].iloc[0]) if f"lstm_h{h}_r2" in mo_row.columns else float("nan")
                all_rows.append(r)

    if not summary:
        print("\n  No results available. Run pooled training first:\n"
              "    python baselines/train_lstm_siso_pooled.py --horizon 1\n"
              "    python baselines/train_lstm_siso_pooled.py --horizon 5\n"
              "    python baselines/train_lstm_siso_pooled.py --horizon 10\n")
        return

    print_summary(summary)

    # pooled improvement over per-stock
    print(f"  Pooled improvement over Per-stock SISO (avg R2 delta):")
    for h in HORIZONS:
        pooled_row  = next((r for r in summary if r["h"] == h and r["model"] == "Pooled SISO"),  None)
        perstk_row  = next((r for r in summary if r["h"] == h and r["model"] == "Per-stock SISO"), None)
        har_row     = next((r for r in summary if r["h"] == h and r["model"] == "HAR-RV"), None)
        if pooled_row and perstk_row:
            delta = pooled_row["avg_r2"] - perstk_row["avg_r2"]
            flag = (
                "[+] Pooled better (ESS helps)"
                if delta > 0.05 else
                ("[-] Per-stock better (noise hurts)"
                 if delta < -0.05 else "[~] negligible")
            )
            print(f"    h={h}: Pooled={pooled_row['avg_r2']:+.4f}  "
                  f"Per-stock={perstk_row['avg_r2']:+.4f}  "
                  f"delta={delta:+.4f}  {flag}")
            if har_row:
                gap = har_row["avg_r2"] - pooled_row["avg_r2"]
                print(f"         HAR-RV={har_row['avg_r2']:+.4f}  "
                      f"gap(HAR-Pooled)={gap:+.4f}  <- structural gap")
    print()

    # save CSV
    if all_rows:
        out_csv = RESULTS / "siso_pooled_comparison.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"  Comparison CSV saved: {out_csv}")

    # plots
    if merged_h:
        plot_scatter_pooled_vs_perstk(merged_h, RESULTS / "siso_pooled_scatter.png")
    if summary:
        plot_bar(summary, RESULTS / "siso_pooled_bar.png")

    print(f"\n{'='*75}")
    print(f"  DONE")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
