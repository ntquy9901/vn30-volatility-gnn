"""
D2 Diagnostic: Moirai2 Embedding - RV Correlation Audit

Tests H1: "Moirai2 embeddings don't carry volatility signal."

Method:
  - Extract Moirai2 embeddings for a sample of test dates
  - Compute corr(embedding_dim_i, RV_stock) for all 384 dims
  - Plot histogram of |corr| values
  - Show top-20 dims with highest correlation

Interpretation:
  - If median |corr| ≈ 0  → embeddings carry no volatility signal (H1 confirmed)
  - If median |corr| > 0.3 → embeddings do carry signal, problem is elsewhere
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
from src.embed_extractor import Moirai2Embedder
from gnn.build_graph import VN30_TICKERS, ALL_NODES
from gnn.train import extract_embeddings

RESULTS     = "results"
HORIZON     = cfg["model"]["horizon"]
CONTEXT_LEN = cfg["model"]["context_length"]
SAMPLE_N    = 10   # number of test dates to sample (each takes ~5-10s on CPU)

print("Loading data...")
close   = load_close_prices(cfg["data"]["prices_dir"], tickers=VN30_TICKERS + ["VNINDEX"])
close   = close[close.index >= pd.Timestamp(cfg["data"].get("data_start", "2014-06-27"))]
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

test_start = pd.Timestamp(cfg["data"]["test_start"])
test_dates = rv_all.index[
    (rv_all.index >= test_start) & (~rv_all.isna().all(axis=1))
]

# Evenly sample SAMPLE_N dates from test set
step       = max(1, len(test_dates) // SAMPLE_N)
sample_dates = test_dates[::step][:SAMPLE_N]
print(f"Sampling {len(sample_dates)} test dates for embedding extraction...")

embedder = Moirai2Embedder(
    size="small",
    context_length=CONTEXT_LEN,
    patch_size=cfg["model"].get("patch_size", 32),
)

all_embeds: list[np.ndarray] = []
all_rv:     list[float]      = []

t0 = time.time()
for di, date in enumerate(sample_dates):
    emb = extract_embeddings(embedder, log_ret, date, CONTEXT_LEN)  # (31, 384)
    for i, tk in enumerate(VN30_TICKERS):
        rv_val = rv_all.loc[date, tk] if (date in rv_all.index and tk in rv_all.columns) else np.nan
        if not np.isnan(rv_val):
            all_embeds.append(emb[i + 1].numpy())   # node i+1 = VN30 stock i
            all_rv.append(rv_val)
    print(f"  date {di+1}/{len(sample_dates)}: {date.date()}  elapsed={time.time()-t0:.1f}s")

E   = np.array(all_embeds, dtype=np.float32)   # (N, 384)
rv  = np.array(all_rv,     dtype=np.float32)   # (N,)
N, D = E.shape
print(f"\nEmbedding matrix: {E.shape}  (N_samples={N}, D_model={D})")
print(f"RV stats: mean={rv.mean():.5f}  std={rv.std():.5f}  min={rv.min():.5f}  max={rv.max():.5f}")

# ── Correlation of each embedding dim with RV ──────────────────────────────────
print("\nComputing correlation of each embedding dim with RV...")
corrs = np.array([
    float(np.corrcoef(E[:, i], rv)[0, 1])
    for i in range(D)
])
abs_corrs = np.abs(corrs)

print(f"\nEmbedding-RV correlation stats (384 dims):")
print(f"  Mean |corr|   : {abs_corrs.mean():.4f}")
print(f"  Median |corr| : {np.median(abs_corrs):.4f}")
print(f"  Max |corr|    : {abs_corrs.max():.4f}")
print(f"  Dims with |corr| > 0.3 : {(abs_corrs > 0.3).sum()}")
print(f"  Dims with |corr| > 0.2 : {(abs_corrs > 0.2).sum()}")
print(f"  Dims with |corr| > 0.1 : {(abs_corrs > 0.1).sum()}")

top20_idx = np.argsort(abs_corrs)[::-1][:20]
print(f"\nTop-20 dims by |corr| with RV:")
for rank, idx in enumerate(top20_idx):
    print(f"  #{rank+1:2d}  dim={idx:3d}  corr={corrs[idx]:+.4f}  |corr|={abs_corrs[idx]:.4f}")

# Save correlation array
np.save(f"{RESULTS}/diag_embed_corrs.npy", corrs)
pd.DataFrame({"dim": range(D), "corr": corrs, "abs_corr": abs_corrs}).to_csv(
    f"{RESULTS}/diag_embed_corrs.csv", index=False
)

# ── Plot: histogram of |corr| ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("D2: Moirai2 Embedding Dimensions vs Realized Volatility", fontsize=13, fontweight="bold")

# Histogram
ax = axes[0]
ax.hist(abs_corrs, bins=40, color="#2563EB", edgecolor="white", linewidth=0.5)
ax.axvline(abs_corrs.mean(),   color="red",    ls="--", lw=1.5, label=f"Mean={abs_corrs.mean():.3f}")
ax.axvline(np.median(abs_corrs), color="orange", ls="-.", lw=1.5, label=f"Median={np.median(abs_corrs):.3f}")
ax.set_xlabel("|Pearson r| (embedding dim, RV)", fontsize=11)
ax.set_ylabel("Count of embedding dimensions", fontsize=11)
ax.set_title("Distribution of |corr| across 384 dims", fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Sorted correlation plot
ax = axes[1]
sorted_abs = np.sort(abs_corrs)[::-1]
ax.plot(range(D), sorted_abs, color="#2563EB", lw=1.2)
ax.axhline(0.3, color="red",    ls="--", lw=1.2, label="|corr|=0.3 threshold")
ax.axhline(0.2, color="orange", ls="-.", lw=1.2, label="|corr|=0.2 threshold")
ax.fill_between(range(D), sorted_abs, alpha=0.15, color="#2563EB")
ax.set_xlabel("Embedding dimension (sorted by |corr|)", fontsize=11)
ax.set_ylabel("|Pearson r| with RV", fontsize=11)
ax.set_title("Sorted embedding-RV correlations", fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/diag_d2_embed_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {RESULTS}/diag_d2_embed_corr.png")

# ── Interpretation ─────────────────────────────────────────────────────────────
median_corr = np.median(abs_corrs)
max_corr    = abs_corrs.max()
n_above_02  = (abs_corrs > 0.2).sum()

print("\n" + "=" * 60)
print("D2 INTERPRETATION")
print("=" * 60)
if median_corr < 0.1:
    print(f"  Median |corr|={median_corr:.4f} << 0.10")
    print("  => H1 CONFIRMED: Moirai2 embeddings carry minimal RV signal.")
    print("     Most dims are near-orthogonal to realized volatility.")
elif median_corr < 0.2:
    print(f"  Median |corr|={median_corr:.4f} -- moderate signal in embeddings.")
    print("  => H1 PARTIALLY confirmed: some dims carry RV signal but weak overall.")
else:
    print(f"  Median |corr|={median_corr:.4f} > 0.20")
    print("  => H1 NOT confirmed: embeddings carry meaningful RV signal.")
    print("     The problem may lie elsewhere (architecture, training, data).")

print(f"\n  {n_above_02}/{D} dims have |corr| > 0.2 ({100*n_above_02/D:.1f}%)")
print(f"  Max single-dim correlation: {max_corr:.4f}  (dim {top20_idx[0]})")
print("=" * 60)
