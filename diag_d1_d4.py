"""
D1 + D4 Diagnostics (quick, no ML needed)

D1 — Scatter plot: predicted vs true RV (all models)
     Reveals bias structure: flat cloud = constant output, diagonal = tracking.

D4 — Train loss trajectory: loss over walk-forward windows
     Reveals whether GNN is actually learning or plateauing immediately.
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
HORIZON    = cfg["model"]["horizon"]

print("Loading data...")
close   = load_close_prices(cfg["data"]["prices_dir"], tickers=VN30_TICKERS)
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

test_dates = rv_all.index[
    (rv_all.index >= TEST_START) & (~rv_all.isna().all(axis=1))
]
rv_test = rv_all.loc[rv_all.index.isin(test_dates)]

pred_files = {
    "GNN (ours)":  "pred_gnn.csv",
    "MLP+Moirai2": "pred_mlp.csv",
    "GARCH(1,1)":  "pred_garch.csv",
    "HAR-RV":      "pred_har.csv",
    "LSTM":        "pred_lstm.csv",
}
COLORS = {
    "GNN (ours)":  "#2563EB",
    "MLP+Moirai2": "#7C3AED",
    "GARCH(1,1)":  "#F59E0B",
    "HAR-RV":      "#10B981",
    "LSTM":        "#EF4444",
}

preds: dict[str, pd.DataFrame] = {}
for name, fname in pred_files.items():
    path = f"{RESULTS}/{fname}"
    if os.path.exists(path):
        preds[name] = pd.read_csv(path, index_col=0, parse_dates=True)


# ── D1: Scatter plots ──────────────────────────────────────────────────────────
print("D1: Plotting scatter (pred vs true RV)...")

n_models = len(preds)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4.5))
fig.suptitle("D1: Predicted vs True RV (test set, all stocks pooled)", fontsize=13, fontweight="bold")

for ax, (name, pdf) in zip(axes, preds.items()):
    common = pdf.index.intersection(rv_test.index)
    yt = rv_test.loc[common].values.ravel()
    yp = pdf.loc[common].values.ravel()
    v  = ~(np.isnan(yt) | np.isnan(yp))
    yt_v, yp_v = yt[v], yp[v]

    # Subsample for readability
    if len(yt_v) > 3000:
        idx = np.random.choice(len(yt_v), 3000, replace=False)
        yt_v, yp_v = yt_v[idx], yp_v[idx]

    ax.scatter(yt_v, yp_v, alpha=0.25, s=4, color=COLORS.get(name, "gray"))
    lim_max = max(yt_v.max(), yp_v.max()) * 1.05
    lim_min = min(yt_v.min(), yp_v.min())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="y=x")

    # Regression line
    try:
        z = np.polyfit(yt_v, yp_v, 1)
        p = np.poly1d(z)
        xs = np.linspace(lim_min, lim_max, 100)
        ax.plot(xs, p(xs), "k-", lw=1.0, alpha=0.7,
                label=f"fit: y={z[0]:.2f}x+{z[1]:.4f}")
    except Exception:
        pass

    # R2
    ss_res = np.sum((yt_v - yp_v)**2)
    ss_tot = np.sum((yt_v - yt_v.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    ax.set_title(f"{name}\nR²={r2:.3f}", fontweight="bold", fontsize=9)
    ax.set_xlabel("True RV", fontsize=8)
    ax.set_ylabel("Predicted RV", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/diag_d1_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS}/diag_d1_scatter.png")


# ── D1b: Prediction distribution histogram ────────────────────────────────────
print("D1b: Prediction distribution histogram...")
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
fig.suptitle("D1b: Distribution of Predictions vs True RV", fontsize=13, fontweight="bold")

# Compute global range for shared bins
all_vals = []
for name, pdf in preds.items():
    common = pdf.index.intersection(rv_test.index)
    yp = pdf.loc[common].values.ravel()
    all_vals.extend(yp[~np.isnan(yp)].tolist())
yt_flat = rv_test.values.ravel()
all_vals.extend(yt_flat[~np.isnan(yt_flat)].tolist())
lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
bins = np.linspace(lo, hi, 50)

for ax, (name, pdf) in zip(axes, preds.items()):
    common = pdf.index.intersection(rv_test.index)
    yt = rv_test.loc[common].values.ravel()
    yp = pdf.loc[common].values.ravel()
    v  = ~(np.isnan(yt) | np.isnan(yp))

    ax.hist(yt[v], bins=bins, alpha=0.5, color="black", label="True RV", density=True)
    ax.hist(yp[v], bins=bins, alpha=0.6, color=COLORS.get(name, "gray"),
            label="Predicted", density=True)
    ax.set_title(name, fontweight="bold", fontsize=9)
    ax.set_xlabel("RV value", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/diag_d1b_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS}/diag_d1b_histograms.png")


# ── D4: Train loss trajectory ──────────────────────────────────────────────────
print("D4: Plotting train loss trajectory...")

loss_path = f"{RESULTS}/train_window_metrics.csv"
if not os.path.exists(loss_path):
    print(f"  NOT FOUND: {loss_path} — skipping D4")
else:
    metrics = pd.read_csv(loss_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("D4: GNN Training Loss Trajectory (walk-forward windows)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(metrics["window"], metrics["train_loss"], color="#2563EB", lw=1.3)
    ax.set_xlabel("Walk-forward window", fontsize=11)
    ax.set_ylabel("Train MSE loss (log-RV scale)", fontsize=11)
    ax.set_title("Loss per window", fontweight="bold")
    ax.grid(alpha=0.3)

    # Rolling average
    roll = metrics["train_loss"].rolling(5, min_periods=1).mean()
    ax.plot(metrics["window"], roll, "r--", lw=1.5, label="5-window rolling avg")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(metrics["window"], metrics["train_loss"].cummin(), color="#10B981", lw=1.5)
    ax.set_xlabel("Walk-forward window", fontsize=11)
    ax.set_ylabel("Cumulative best loss", fontsize=11)
    ax.set_title("Best loss achieved so far", fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/diag_d4_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULTS}/diag_d4_train_loss.png")

    # Summary stats
    print(f"\n  Loss trajectory stats:")
    print(f"    Window count : {len(metrics)}")
    print(f"    Initial loss : {metrics['train_loss'].iloc[0]:.6f}")
    print(f"    Final loss   : {metrics['train_loss'].iloc[-1]:.6f}")
    print(f"    Best loss    : {metrics['train_loss'].min():.6f}  (window {metrics['train_loss'].idxmin()+1})")
    print(f"    Improvement  : {(1 - metrics['train_loss'].min()/metrics['train_loss'].iloc[0])*100:.1f}%")

print("\nD1 + D4 diagnostics done.")
