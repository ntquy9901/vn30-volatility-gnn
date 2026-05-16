"""
Compare 4 pooling strategies for Moirai2 embeddings vs Realized Volatility.

Pooling options: 'last', 'last_context', 'mean_context', 'mean'
Metric: median |Pearson r| across all 384 dims with RV.

Run before retraining to pick the best pooling strategy.
"""
import os, sys, warnings, yaml, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from src.embed_extractor import Moirai2Embedder, _POOLING_OPTIONS
from gnn.build_graph import VN30_TICKERS, ALL_NODES
from gnn.train import extract_embeddings

RESULTS     = "results"
HORIZON     = cfg["model"]["horizon"]
CONTEXT_LEN = cfg["model"]["context_length"]
SAMPLE_N    = 6    # test dates to sample — keeps runtime ~1 min

print("Loading data...")
close   = load_close_prices(cfg["data"]["prices_dir"], tickers=VN30_TICKERS + ["VNINDEX"])
close   = close[close.index >= pd.Timestamp(cfg["data"].get("data_start", "2014-06-27"))]
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

test_start = pd.Timestamp(cfg["data"]["test_start"])
test_dates = rv_all.index[
    (rv_all.index >= test_start) & (~rv_all.isna().all(axis=1))
]
step         = max(1, len(test_dates) // SAMPLE_N)
sample_dates = test_dates[::step][:SAMPLE_N]

POOLING_OPTS = ["last", "last_context", "mean_context", "mean"]
results: dict[str, dict] = {}

for pooling in POOLING_OPTS:
    print(f"\n--- Pooling: '{pooling}' ---")
    embedder = Moirai2Embedder(
        size="small",
        context_length=CONTEXT_LEN,
        patch_size=cfg["model"].get("patch_size", 32),
        pooling=pooling,
    )

    all_embeds, all_rv = [], []
    t0 = time.time()
    for di, date in enumerate(sample_dates):
        emb = extract_embeddings(embedder, log_ret, date, CONTEXT_LEN)  # (31, 384)
        for i, tk in enumerate(VN30_TICKERS):
            rv_val = rv_all.at[date, tk] if (date in rv_all.index and tk in rv_all.columns) else np.nan
            if not np.isnan(rv_val):
                all_embeds.append(emb[i + 1].numpy())
                all_rv.append(rv_val)
        print(f"  date {di+1}/{len(sample_dates)}: {date.date()}  {time.time()-t0:.1f}s")

    E   = np.array(all_embeds, dtype=np.float32)   # (N, 384)
    rv  = np.array(all_rv,     dtype=np.float32)

    corrs     = np.array([np.corrcoef(E[:, i], rv)[0, 1] for i in range(E.shape[1])])
    abs_corrs = np.abs(corrs)

    results[pooling] = {
        "corrs":         corrs,
        "abs_corrs":     abs_corrs,
        "mean_abs":      float(abs_corrs.mean()),
        "median_abs":    float(np.median(abs_corrs)),
        "max_abs":       float(abs_corrs.max()),
        "n_above_02":    int((abs_corrs > 0.2).sum()),
        "n_above_03":    int((abs_corrs > 0.3).sum()),
        "embed_norm":    float(np.linalg.norm(E, axis=1).mean()),
    }

    r = results[pooling]
    print(f"  Mean |corr|   : {r['mean_abs']:.4f}")
    print(f"  Median |corr| : {r['median_abs']:.4f}")
    print(f"  Max |corr|    : {r['max_abs']:.4f}")
    print(f"  Dims > 0.2    : {r['n_above_02']}/384")
    print(f"  Dims > 0.3    : {r['n_above_03']}/384")


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("POOLING STRATEGY COMPARISON (embedding-RV correlation)")
print("=" * 60)
print(f"{'Pooling':<15} {'Mean|r|':>9} {'Median|r|':>11} {'Max|r|':>8} {'>0.2':>6} {'>0.3':>6}")
print("-" * 60)
for p in POOLING_OPTS:
    r = results[p]
    marker = " <-- BEST" if r["median_abs"] == max(results[p2]["median_abs"] for p2 in POOLING_OPTS) else ""
    print(f"{p:<15} {r['mean_abs']:>9.4f} {r['median_abs']:>11.4f} {r['max_abs']:>8.4f} "
          f"{r['n_above_02']:>6} {r['n_above_03']:>6}{marker}")
print("=" * 60)

best_pooling = max(POOLING_OPTS, key=lambda p: results[p]["median_abs"])
print(f"\nRecommended pooling: '{best_pooling}'")


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(POOLING_OPTS), figsize=(16, 4), sharey=True)
fig.suptitle("Moirai2 Pooling Strategy: |corr| with RV across 384 dims", fontsize=13, fontweight="bold")

colors = {"last": "#2563EB", "last_context": "#7C3AED", "mean_context": "#10B981", "mean": "#F59E0B"}
for ax, p in zip(axes, POOLING_OPTS):
    ac = results[p]["abs_corrs"]
    ax.hist(ac, bins=30, color=colors[p], edgecolor="white", lw=0.5)
    ax.axvline(np.median(ac), color="red", ls="--", lw=1.5,
               label=f"median={np.median(ac):.3f}")
    ax.set_title(f"'{p}'\nmean={ac.mean():.3f}", fontweight="bold", fontsize=9)
    ax.set_xlabel("|corr|", fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/diag_pooling_compare.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS}/diag_pooling_compare.png")
