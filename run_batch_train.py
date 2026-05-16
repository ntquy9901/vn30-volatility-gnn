"""
Batch Training Experiment — tests the training-regime hypothesis (H2).

Walk-forward: 30 samples/window (30 stocks × 1 date) → starved gradient.
Batch:        all historical (embedding, RV) pairs at once → same data volume as HAR-OLS.

Steps
-----
1. Extract + cache Moirai2 embeddings for every training window → disk
2. Train Batch MLP  on all pairs (mini-batch SGD, time-ordered val split)
3. Train Batch GNN  on all graph snapshots (DGL dgl.batch mini-batching)
4. Inference on 2026 test set for both
5. Compare: walk-forward GNN/MLP vs Batch GNN/MLP vs HAR-RV vs LSTM

Expected outcome
----------------
  If batch >> walk-forward  → H2 confirmed: training regime was the bottleneck
  If batch ≈ walk-forward   → regime not the issue; feature quality is the ceiling
"""
import os, sys, warnings, yaml, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.volatility_labels import (
    load_close_prices, compute_log_returns, compute_rv, get_rv_node_features
)
from src.embed_extractor import Moirai2Embedder
from gnn.build_graph import build_graph, VN30_TICKERS, ALL_NODES
from gnn.model import VolatilityGNN
from gnn.train import walk_forward_dates, extract_embeddings
from baselines.mlp_baseline import VolatilityMLP
from evaluation.metrics import compare_models

import dgl

RESULTS      = Path("results")
CACHE_DIR    = RESULTS / "embed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE       = torch.device("cpu")
HORIZON      = cfg["model"]["horizon"]
CONTEXT_LEN  = cfg["model"]["context_length"]
USE_LOG_RV   = cfg["model"].get("use_log_rv", True)
USE_LAGGED   = cfg["model"].get("use_lagged_rv", True)
POOLING      = cfg["model"].get("pooling", "last_context")
GNN_HIDDEN   = cfg["model"]["gnn_hidden"]
MLP_HIDDEN   = cfg["model"]["mlp_hidden"]
DROPOUT      = cfg["model"]["dropout"]
LR           = cfg["training"]["lr"]
WD           = cfg["training"].get("weight_decay", 1e-4)
GRAD_CLIP    = cfg["training"].get("grad_clip", 1.0)
EPOCHS       = 300
PATIENCE     = 40
BATCH_SIZE   = 32
VAL_FRAC     = 0.20       # last VAL_FRAC windows → val set (time-ordered)
IN_DIM       = Moirai2Embedder.D_MODEL + (3 if USE_LAGGED else 0)   # 384 or 387

import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading prices / RV …")
close   = load_close_prices(cfg["data"]["prices_dir"], tickers=VN30_TICKERS + ["VNINDEX"])
close   = close[close.index >= pd.Timestamp(cfg["data"].get("data_start", "2014-06-27"))]
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

train_end  = pd.Timestamp(cfg["data"]["train_end"])
test_start = pd.Timestamp(cfg["data"]["test_start"])
train_dates = pd.DatetimeIndex(log_ret.index)
window_ends = walk_forward_dates(train_dates, train_end, CONTEXT_LEN,
                                 cfg["model"]["stride"])
print(f"Walk-forward windows (training): {len(window_ends)}")

# ── Step 1: Cache embeddings ───────────────────────────────────────────────────
print("\n[Step 1] Extracting / loading cached embeddings …")
embedder = Moirai2Embedder(
    size="small", context_length=CONTEXT_LEN,
    patch_size=cfg["model"].get("patch_size", 32), pooling=POOLING,
)

cache_index_path = CACHE_DIR / "index.csv"
if cache_index_path.exists():
    idx_df = pd.read_csv(cache_index_path, parse_dates=["window_end"])
    cached_dates = set(idx_df["window_end"].tolist())
else:
    idx_df = pd.DataFrame(columns=["window_idx", "window_end", "cache_file"])
    cached_dates = set()

new_rows = []
t0 = time.time()
for w_idx, wend in enumerate(window_ends):
    fname = f"w{w_idx:03d}.npz"
    fpath = CACHE_DIR / fname
    if wend in cached_dates and fpath.exists():
        continue
    feats = extract_embeddings(embedder, log_ret, wend, CONTEXT_LEN)   # (31, 384)
    if USE_LAGGED:
        rv_f = get_rv_node_features(log_ret, wend, ALL_NODES, HORIZON)
        feats = torch.cat([feats, rv_f], dim=-1)                        # (31, 387)
    np.savez_compressed(fpath, embed=feats.numpy(), date=str(wend.date()))
    new_rows.append({"window_idx": w_idx, "window_end": wend, "cache_file": fname})
    if (w_idx + 1) % 20 == 0 or w_idx == len(window_ends) - 1:
        print(f"  cached {w_idx+1}/{len(window_ends)}  {time.time()-t0:.1f}s")

if new_rows:
    new_df = pd.DataFrame(new_rows)
    idx_df  = pd.concat([idx_df, new_df], ignore_index=True)
    idx_df.to_csv(cache_index_path, index=False)
    print(f"  Saved {len(new_rows)} new cache files.")
else:
    print(f"  All {len(window_ends)} windows already cached.")

# ── Build flat dataset ─────────────────────────────────────────────────────────
print("\nBuilding flat (feature, label) dataset from cache …")
all_X, all_y, all_mask, all_graphs = [], [], [], []
valid_windows = []

for w_idx, wend in enumerate(window_ends):
    fpath = CACHE_DIR / f"w{w_idx:03d}.npz"
    if not fpath.exists():
        continue
    data = np.load(fpath)
    feats = torch.tensor(data["embed"], dtype=torch.float)       # (31, D)

    rv_at = rv_all.loc[wend] if wend in rv_all.index else None
    if rv_at is None or rv_at.isna().all():
        continue

    labels = np.full(31, np.nan, dtype=np.float32)
    for i, tk in enumerate(VN30_TICKERS):
        v = rv_at.get(tk, np.nan)
        if not pd.isna(v):
            labels[i + 1] = float(v)

    valid = ~np.isnan(labels[1:])          # (30,) bool — VN30 stocks only
    if valid.sum() == 0:
        continue

    # Flat MLP dataset: one row per (stock, window)
    for i in range(30):
        if valid[i]:
            raw = float(labels[i + 1])
            tgt = np.log(max(raw, 1e-8)) if USE_LOG_RV else raw
            all_X.append(feats[i + 1].numpy())
            all_y.append(tgt)

    # GNN dataset: keep full graph + node features + labels tensor
    g  = build_graph(log_ret, end_date=wend,
                     corr_window=cfg["model"]["corr_window"],
                     corr_threshold=cfg["model"]["corr_threshold"])
    lbl_t = torch.tensor(labels, dtype=torch.float)
    if USE_LOG_RV:
        lbl_t[1:][torch.tensor(valid)] = torch.log(
            lbl_t[1:][torch.tensor(valid)].clamp(min=1e-8)
        )
    all_graphs.append((g, feats, lbl_t, torch.tensor(valid)))
    valid_windows.append(w_idx)

X_all = torch.tensor(np.array(all_X), dtype=torch.float)   # (N_samples, D)
y_all = torch.tensor(np.array(all_y), dtype=torch.float)   # (N_samples,)
print(f"Flat MLP dataset : {X_all.shape[0]} samples  (D={X_all.shape[1]})")
print(f"GNN graph dataset: {len(all_graphs)} graphs")

# Time-ordered train / val split
n_graphs = len(all_graphs)
n_val_g  = max(1, int(n_graphs * VAL_FRAC))
n_trn_g  = n_graphs - n_val_g

# For MLP, map samples back to their window to split time-ordered
# Simple approach: last VAL_FRAC of all_X rows (they were appended in time order)
n_samples = X_all.shape[0]
n_val_s   = max(1, int(n_samples * VAL_FRAC))
X_trn, X_val = X_all[:-n_val_s], X_all[-n_val_s:]
y_trn, y_val = y_all[:-n_val_s], y_all[-n_val_s:]
graphs_trn   = all_graphs[:n_trn_g]
graphs_val   = all_graphs[n_trn_g:]
print(f"MLP  train={len(X_trn)}  val={len(X_val)}")
print(f"GNN  train={n_trn_g}     val={n_val_g}")


# ── Step 2: Train Batch MLP ────────────────────────────────────────────────────
print("\n[Step 2] Training Batch MLP …")

mlp = VolatilityMLP(in_dim=IN_DIM, hidden=GNN_HIDDEN,
                    mlp_hidden=MLP_HIDDEN, dropout=DROPOUT).to(DEVICE)
opt_mlp = optim.Adam(mlp.parameters(), lr=LR, weight_decay=WD)

trn_ds  = TensorDataset(X_trn, y_trn)
trn_ld  = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ds  = TensorDataset(X_val, y_val)
val_ld  = DataLoader(val_ds, batch_size=256, shuffle=False)

best_val, pat_ctr = float("inf"), 0
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    mlp.train()
    for xb, yb in trn_ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_mlp.zero_grad()
        pred = mlp(xb).squeeze(-1)
        loss = torch.mean((pred - yb) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), GRAD_CLIP)
        opt_mlp.step()

    mlp.eval()
    with torch.no_grad():
        val_loss = sum(
            torch.mean((mlp(xb.to(DEVICE)).squeeze(-1) - yb.to(DEVICE)) ** 2).item() * len(xb)
            for xb, yb in val_ld
        ) / len(X_val)

    if val_loss < best_val - 1e-6:
        best_val, pat_ctr = val_loss, 0
        torch.save(mlp.state_dict(), RESULTS / "best_batch_mlp.pt")
    else:
        pat_ctr += 1
        if pat_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}  val_loss={best_val:.6f}")
            break

    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}  val_loss={val_loss:.6f}  best={best_val:.6f}")

print(f"Batch MLP done in {time.time()-t0:.1f}s  best_val={best_val:.6f}")


# ── Step 3: Train Batch GNN ────────────────────────────────────────────────────
print("\n[Step 3] Training Batch GNN …")

gnn = VolatilityGNN(in_dim=IN_DIM, hidden=GNN_HIDDEN,
                    mlp_hidden=MLP_HIDDEN, dropout=DROPOUT).to(DEVICE)
opt_gnn = optim.Adam(gnn.parameters(), lr=LR, weight_decay=WD)

def gnn_batch_loss(model, batch_graphs):
    """Forward pass over a list of (GraphData, feats, labels, valid_mask) tuples."""
    total_loss, total_n = 0.0, 0
    for gdata, feats, lbl, valid in batch_graphs:
        g    = gdata.g.to(DEVICE)
        x    = feats.to(DEVICE)
        lbl  = lbl.to(DEVICE)
        pred = model(g, x).squeeze(-1)          # (31,)
        # loss_mask excludes node 0 (VNINDEX); vn30_valid excludes missing RV labels
        vn30_valid = torch.zeros(31, dtype=torch.bool, device=DEVICE)
        vn30_valid[1:] = valid.to(DEVICE)
        final_mask = gdata.loss_mask.to(DEVICE) & vn30_valid
        if final_mask.sum() == 0:
            continue
        loss = torch.mean((pred[final_mask] - lbl[final_mask]) ** 2)
        total_loss += loss * final_mask.sum().item()
        total_n    += final_mask.sum().item()
    return total_loss / max(total_n, 1)

best_val_gnn, pat_ctr_gnn = float("inf"), 0
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    gnn.train()
    # Shuffle training graphs
    perm = torch.randperm(len(graphs_trn)).tolist()
    # Process in mini-batches of graphs
    for start in range(0, len(graphs_trn), BATCH_SIZE):
        batch = [graphs_trn[i] for i in perm[start:start + BATCH_SIZE]]
        opt_gnn.zero_grad()
        loss = gnn_batch_loss(gnn, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), GRAD_CLIP)
        opt_gnn.step()

    gnn.eval()
    with torch.no_grad():
        val_loss_gnn = gnn_batch_loss(gnn, graphs_val).item()

    if val_loss_gnn < best_val_gnn - 1e-6:
        best_val_gnn, pat_ctr_gnn = val_loss_gnn, 0
        torch.save(gnn.state_dict(), RESULTS / "best_batch_gnn.pt")
    else:
        pat_ctr_gnn += 1
        if pat_ctr_gnn >= PATIENCE:
            print(f"  Early stop at epoch {epoch}  val_loss={best_val_gnn:.6f}")
            break

    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}  val_loss={val_loss_gnn:.6f}  best={best_val_gnn:.6f}")

print(f"Batch GNN done in {time.time()-t0:.1f}s  best_val={best_val_gnn:.6f}")


# ── Step 4: Inference on test set ─────────────────────────────────────────────
print("\n[Step 4] Inference on test set …")

mlp.load_state_dict(torch.load(RESULTS / "best_batch_mlp.pt", map_location=DEVICE))
mlp.eval()
gnn.load_state_dict(torch.load(RESULTS / "best_batch_gnn.pt", map_location=DEVICE))
gnn.eval()

test_dates = rv_all.index[
    (rv_all.index >= test_start) & (~rv_all.isna().all(axis=1))
]

batch_mlp_preds = {tk: {} for tk in VN30_TICKERS}
batch_gnn_preds = {tk: {} for tk in VN30_TICKERS}

t0 = time.time()
with torch.no_grad():
    for i, date in enumerate(test_dates):
        feats = extract_embeddings(embedder, log_ret, date, CONTEXT_LEN)
        if USE_LAGGED:
            rv_f  = get_rv_node_features(log_ret, date, ALL_NODES, HORIZON)
            feats = torch.cat([feats, rv_f], dim=-1)
        feats = feats.to(DEVICE)

        # MLP: each VN30 node independently
        pred_mlp = mlp(feats[1:]).cpu().numpy().ravel()    # (30,)
        for j, tk in enumerate(VN30_TICKERS):
            raw = float(pred_mlp[j])
            batch_mlp_preds[tk][date] = max(float(np.exp(raw)) if USE_LOG_RV else raw, 0.0)

        # GNN: full graph
        g_data = build_graph(log_ret, end_date=date,
                             corr_window=cfg["model"]["corr_window"],
                             corr_threshold=cfg["model"]["corr_threshold"])
        pred_gnn = gnn(g_data.g.to(DEVICE), feats).cpu().numpy().ravel()   # (31,)
        for j, tk in enumerate(VN30_TICKERS):
            raw = float(pred_gnn[j + 1])
            batch_gnn_preds[tk][date] = max(float(np.exp(raw)) if USE_LOG_RV else raw, 0.0)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_dates)}  {time.time()-t0:.1f}s")

batch_mlp_df = pd.DataFrame({t: pd.Series(batch_mlp_preds[t]) for t in VN30_TICKERS})
batch_gnn_df = pd.DataFrame({t: pd.Series(batch_gnn_preds[t]) for t in VN30_TICKERS})
batch_mlp_df.to_csv(RESULTS / "pred_batch_mlp.csv")
batch_gnn_df.to_csv(RESULTS / "pred_batch_gnn.csv")
print(f"Saved pred_batch_mlp.csv / pred_batch_gnn.csv  shape={batch_mlp_df.shape}")


# ── Step 5: Compare all models ─────────────────────────────────────────────────
print("\n[Step 5] Comparing all models …")

def pool(pred_df):
    common = pred_df.index.intersection(rv_all.index)
    yt = rv_all.loc[common].values.ravel()
    yp = pred_df.loc[common].values.ravel()
    v  = ~(np.isnan(yt) | np.isnan(yp))
    return yt[v], yp[v]

yt, yp_bmlp = pool(batch_mlp_df)
_,  yp_bgnn = pool(batch_gnn_df)

wf_gnn_df = pd.read_csv(RESULTS / "pred_gnn.csv",   index_col=0, parse_dates=True)
wf_mlp_df = pd.read_csv(RESULTS / "pred_mlp.csv",   index_col=0, parse_dates=True)
har_df    = pd.read_csv(RESULTS / "pred_har.csv",   index_col=0, parse_dates=True)
lstm_df   = pd.read_csv(RESULTS / "pred_lstm.csv",  index_col=0, parse_dates=True)

_, yp_wfgnn = pool(wf_gnn_df)
_, yp_wfmlp = pool(wf_mlp_df)
_, yp_har   = pool(har_df)
_, yp_lstm  = pool(lstm_df)

n = len(yt)
def align(a): return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), np.nan)])

results = compare_models(
    yt,
    {
        "Batch GNN  (H2 test)": yp_bgnn,
        "Batch MLP  (H2 test)": yp_bmlp,
        "WalkFwd GNN":           align(yp_wfgnn),
        "WalkFwd MLP":           align(yp_wfmlp),
        "HAR-RV (OLS)":          align(yp_har),
        "LSTM":                  align(yp_lstm),
    },
    dm_reference = "Batch MLP  (H2 test)",
    dm_loss      = cfg["evaluation"]["dm_loss"],
)

cols = ["MAE", "RMSE", "R2", "QLIKE", "Pearson_r", "DM_stat", "DM_pval"]
print("\n" + "=" * 74)
print("H2 RESULT: Batch vs Walk-Forward Training")
print("=" * 74)
print(results[cols].round(4).to_string())
print("=" * 74)

bmlp = results.loc["Batch MLP  (H2 test)", "MAE"]
bgnn = results.loc["Batch GNN  (H2 test)", "MAE"]
wmlp = results.loc["WalkFwd MLP",           "MAE"]
wgnn = results.loc["WalkFwd GNN",           "MAE"]
har  = results.loc["HAR-RV (OLS)",          "MAE"]

print(f"\nBatch vs Walk-forward:")
print(f"  MLP: {bmlp:.5f} vs {wmlp:.5f}  => {(wmlp-bmlp)/wmlp*100:+.1f}% (batch)")
print(f"  GNN: {bgnn:.5f} vs {wgnn:.5f}  => {(wgnn-bgnn)/wgnn*100:+.1f}% (batch)")
print(f"\nBatch vs HAR-RV:")
print(f"  MLP: {bmlp:.5f} vs HAR {har:.5f}  gap={bmlp/har:.1f}x")
print(f"  GNN: {bgnn:.5f} vs HAR {har:.5f}  gap={bgnn/har:.1f}x")

print()
if bmlp < wmlp * 0.90:
    print("=> H2 CONFIRMED: Batch training significantly improves MLP (>10% MAE drop).")
elif bmlp < wmlp:
    print("=> H2 PARTIAL: Batch training marginally improves MLP.")
else:
    print("=> H2 NOT confirmed: Batch training did not improve MLP.")

results.to_csv(RESULTS / "batch_vs_walkforward.csv")
print(f"\nSaved: results/batch_vs_walkforward.csv")
