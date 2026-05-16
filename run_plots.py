"""
Visualizations and per-stock analysis.

Generates:
  1. Bar chart: MAE / RMSE / QLIKE by model
  2. R2 heatmap: 30 stocks x 5 models
  3. Time series: predicted vs true RV for selected stocks
  4. Per-stock metrics table (CSV + console)

Run after: run_evaluation.py
"""
import os, sys, warnings, yaml
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS

RESULTS    = "results"
TEST_START = pd.Timestamp(cfg["data"]["test_start"])
PRICES_DIR = cfg["data"]["prices_dir"]
HORIZON    = cfg["model"]["horizon"]

MODEL_NAMES = ["GNN (ours)", "MLP+Moirai2", "GARCH(1,1)", "HAR-RV", "LSTM"]
COLORS      = ["#2563EB", "#7C3AED", "#F59E0B", "#10B981", "#EF4444"]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
close   = load_close_prices(PRICES_DIR, tickers=VN30_TICKERS)
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

test_dates = rv_all.index[
    (rv_all.index >= TEST_START) & (~rv_all.isna().all(axis=1))
]

pred_files = {
    "GNN (ours)":  "pred_gnn.csv",
    "MLP+Moirai2": "pred_mlp.csv",
    "GARCH(1,1)":  "pred_garch.csv",
    "HAR-RV":      "pred_har.csv",
    "LSTM":        "pred_lstm.csv",
}

preds: dict[str, pd.DataFrame] = {}
for name, fname in pred_files.items():
    preds[name] = pd.read_csv(f"{RESULTS}/{fname}", index_col=0, parse_dates=True)

rv_test = rv_all.loc[rv_all.index.isin(test_dates)]

# ── Per-stock metrics ──────────────────────────────────────────────────────────
print("Computing per-stock metrics...")

from evaluation.metrics import mae, rmse, r2_score, qlike, pearson_r

rows_per_stock = []
for model_name in MODEL_NAMES:
    pdf = preds[model_name]
    for tk in VN30_TICKERS:
        if tk not in pdf.columns or tk not in rv_test.columns:
            continue
        common = pdf.index.intersection(rv_test.index)
        yt = rv_test.loc[common, tk].values
        yp = pdf.loc[common, tk].values
        v  = ~(np.isnan(yt) | np.isnan(yp))
        if v.sum() < 5:
            continue
        rows_per_stock.append({
            "Model":  model_name,
            "Ticker": tk,
            "MAE":    mae(yt[v], yp[v]),
            "RMSE":   rmse(yt[v], yp[v]),
            "R2":     r2_score(yt[v], yp[v]),
            "QLIKE":  qlike(yt[v], yp[v]),
            "Pearson_r": pearson_r(yt[v], yp[v]),
            "N":      int(v.sum()),
        })

per_stock_df = pd.DataFrame(rows_per_stock)
per_stock_df.to_csv(f"{RESULTS}/per_stock_metrics.csv", index=False)
print(f"Saved: {RESULTS}/per_stock_metrics.csv  ({len(per_stock_df)} rows)")

# Print summary per stock (GNN vs HAR-RV)
print("\n=== Per-stock R2: GNN vs HAR-RV ===")
gnn_ps  = per_stock_df[per_stock_df.Model == "GNN (ours)"].set_index("Ticker")
har_ps  = per_stock_df[per_stock_df.Model == "HAR-RV"].set_index("Ticker")
mlp_ps  = per_stock_df[per_stock_df.Model == "MLP+Moirai2"].set_index("Ticker")
compare = pd.DataFrame({
    "GNN_MAE":  gnn_ps["MAE"],
    "MLP_MAE":  mlp_ps["MAE"],
    "HAR_MAE":  har_ps["MAE"],
    "GNN_R2":   gnn_ps["R2"],
    "HAR_R2":   har_ps["R2"],
}).round(4)
print(compare.to_string())


# ── 1. Bar chart ───────────────────────────────────────────────────────────────
print("\nPlotting bar chart...")

summary = pd.read_csv(f"{RESULTS}/model_comparison.csv", index_col=0)
metrics_plot = ["MAE", "RMSE", "QLIKE"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison - Pooled VN30 Test Set", fontsize=14, fontweight="bold")

for ax, metric in zip(axes, metrics_plot):
    vals  = [summary.loc[m, metric] if m in summary.index else np.nan for m in MODEL_NAMES]
    bars  = ax.bar(range(len(MODEL_NAMES)), vals, color=COLORS, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_title(metric, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)
    best_idx = int(np.nanargmin(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig(f"{RESULTS}/plot_bar_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS}/plot_bar_metrics.png")


# ── 2. R2 heatmap ─────────────────────────────────────────────────────────────
print("Plotting R2 heatmap...")

r2_mat = pd.DataFrame(index=VN30_TICKERS, columns=MODEL_NAMES, dtype=float)
for model_name in MODEL_NAMES:
    for tk in VN30_TICKERS:
        row = per_stock_df[(per_stock_df.Model == model_name) & (per_stock_df.Ticker == tk)]
        if len(row) > 0:
            r2_mat.loc[tk, model_name] = float(row.iloc[0]["R2"])

r2_mat = r2_mat.astype(float)

fig, ax = plt.subplots(figsize=(10, 12))
data = r2_mat.values
vmin = max(data[~np.isnan(data)].min(), -1.0)
vmax = 1.0
im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(MODEL_NAMES)))
ax.set_xticklabels(MODEL_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(VN30_TICKERS)))
ax.set_yticklabels(VN30_TICKERS, fontsize=8)
ax.set_title("R² per Stock x Model (test set)", fontweight="bold", fontsize=13)
plt.colorbar(im, ax=ax, label="R²", fraction=0.03, pad=0.04)

for i in range(len(VN30_TICKERS)):
    for j in range(len(MODEL_NAMES)):
        val = data[i, j]
        if not np.isnan(val):
            color = "black" if abs(val) < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)

plt.tight_layout()
plt.savefig(f"{RESULTS}/plot_r2_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS}/plot_r2_heatmap.png")


# ── 3. Time series plot ────────────────────────────────────────────────────────
print("Plotting time series...")

FOCUS_TICKERS = ["VCB", "HPG", "FPT"]
fig, axes = plt.subplots(len(FOCUS_TICKERS), 1, figsize=(14, 10), sharex=False)
fig.suptitle("Predicted vs True RV - Test Set", fontsize=13, fontweight="bold")

LINE_STYLES = {
    "True RV":    dict(color="black", lw=1.8, ls="-", zorder=5),
    "GNN (ours)": dict(color="#2563EB", lw=1.2, ls="-"),
    "MLP+Moirai2": dict(color="#7C3AED", lw=1.0, ls="--"),
    "HAR-RV":     dict(color="#10B981", lw=1.0, ls="-."),
    "LSTM":       dict(color="#EF4444", lw=1.0, ls=":"),
}

for ax, tk in zip(axes, FOCUS_TICKERS):
    common = rv_test.index
    for model_name, pdf in preds.items():
        if model_name not in LINE_STYLES:
            continue
        c = pdf[tk].reindex(common) if tk in pdf.columns else pd.Series(np.nan, index=common)
        style = LINE_STYLES[model_name]
        ax.plot(common, c.values, label=model_name, **style)
    true_rv = rv_test[tk].reindex(common)
    ax.plot(common, true_rv.values, label="True RV", **LINE_STYLES["True RV"])
    ax.set_title(f"{tk}", fontweight="bold", fontsize=11)
    ax.set_ylabel("RV (daily std)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/plot_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS}/plot_timeseries.png")


# ── 4. Per-stock rank table ────────────────────────────────────────────────────
print("\n=== Best model per stock (by MAE) ===")
best = (
    per_stock_df.groupby("Ticker")
    .apply(lambda g: g.loc[g["MAE"].idxmin(), "Model"])
    .reset_index()
)
best.columns = ["Ticker", "Best_Model"]
counts = best.Best_Model.value_counts()
print(counts.to_string())
print()
print(best.to_string(index=False))

best.to_csv(f"{RESULTS}/best_model_per_stock.csv", index=False)
print(f"\nSaved: {RESULTS}/best_model_per_stock.csv")
print("\nAll plots and tables done.")
