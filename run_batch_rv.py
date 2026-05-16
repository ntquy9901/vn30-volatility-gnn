"""
Extended RV Features Experiment — batch training with 4 model variants.

Answers two questions:
  Q1. Do extended RV features (6-dim) improve over HAR-3 features (3-dim)?
  Q2. Does Moirai2 embedding add value ON TOP OF extended RV features?

Model variants (all trained with batch regime):
  A. MLP  (384 Moirai2 + 6 RV) — full feature set
  B. GNN  (384 Moirai2 + 6 RV) — full feature set with graph
  C. MLP  (6 RV only)           — no Moirai2; ablation for feature quality
  D. GNN  (6 RV only)           — no Moirai2; graph over pure RV features

Extended RV features (6-dim per node):
  log(RV_d)     — daily past-h RV
  log(RV_w)     — 5-day avg RV  (weekly)
  log(RV_m)     — 22-day avg RV (monthly)
  log(RV_q)     — 60-day avg RV (quarterly)
  corr_vnindex  — rolling 60-day corr with VNINDEX (systematic risk)
  jump_ratio    — max(RV_d - RV_w, 0) / RV_d  (jump component fraction)

Reuses embed_cache/ from run_batch_train.py (no Moirai2 re-extraction needed).
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
    load_close_prices, compute_log_returns, compute_rv, get_extended_rv_features
)
from src.embed_extractor import Moirai2Embedder
from gnn.build_graph import build_graph, VN30_TICKERS, ALL_NODES
from gnn.model import VolatilityGNN
from gnn.train import walk_forward_dates, extract_embeddings
from baselines.mlp_baseline import VolatilityMLP
from evaluation.metrics import compare_models

RESULTS     = Path("results")
CACHE_DIR   = RESULTS / "embed_cache"
DEVICE      = torch.device("cpu")
HORIZON     = cfg["model"]["horizon"]
CONTEXT_LEN = cfg["model"]["context_length"]
USE_LOG_RV  = cfg["model"].get("use_log_rv", True)
POOLING     = cfg["model"].get("pooling", "last_context")
GNN_HIDDEN  = cfg["model"]["gnn_hidden"]
MLP_HIDDEN  = cfg["model"]["mlp_hidden"]
DROPOUT     = cfg["model"]["dropout"]
LR          = cfg["training"]["lr"]
WD          = cfg["training"].get("weight_decay", 1e-4)
GRAD_CLIP   = cfg["training"].get("grad_clip", 1.0)
EPOCHS      = 300
PATIENCE    = 40
BATCH_SIZE  = 32
VAL_FRAC    = 0.20
N_RV        = 6     # extended RV feature dims
D_EMBED     = Moirai2Embedder.D_MODEL   # 384

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
print(f"Training windows: {len(window_ends)}")

# ── Build dataset (reuse cached Moirai2 embeddings, compute RV features fresh) ──
print("\nBuilding dataset from embed_cache + extended RV features …")
t0 = time.time()

# Four parallel datasets
X_full, X_rv_only = [], []   # for MLP (flat: one row per stock per window)
y_flat = []
graphs_full, graphs_rv = [], []   # for GNN (one graph per window)

for w_idx, wend in enumerate(window_ends):
    fpath = CACHE_DIR / f"w{w_idx:03d}.npz"
    if not fpath.exists():
        continue

    data       = np.load(fpath)
    embed_raw  = torch.tensor(data["embed"], dtype=torch.float)   # (31, 384+3) or (31,384)
    # Strip any existing lagged-RV cols — we recompute with extended 6-feat version
    embed_384  = embed_raw[:, :D_EMBED]                           # (31, 384)

    rv_feats = get_extended_rv_features(log_ret, wend, ALL_NODES, HORIZON)  # (31, 6)

    feats_full = torch.cat([embed_384, rv_feats], dim=-1)   # (31, 390)
    feats_rv   = rv_feats                                    # (31, 6)

    rv_at = rv_all.loc[wend] if wend in rv_all.index else None
    if rv_at is None or rv_at.isna().all():
        continue

    labels = np.full(31, np.nan, dtype=np.float32)
    for i, tk in enumerate(VN30_TICKERS):
        v = rv_at.get(tk, np.nan)
        if not pd.isna(v):
            labels[i + 1] = float(v)

    valid = ~np.isnan(labels[1:])    # (30,)
    if valid.sum() == 0:
        continue

    for i in range(30):
        if valid[i]:
            raw = float(labels[i + 1])
            tgt = np.log(max(raw, 1e-8)) if USE_LOG_RV else raw
            X_full.append(feats_full[i + 1].numpy())
            X_rv_only.append(feats_rv[i + 1].numpy())
            y_flat.append(tgt)

    g = build_graph(log_ret, end_date=wend,
                    corr_window=cfg["model"]["corr_window"],
                    corr_threshold=cfg["model"]["corr_threshold"])
    lbl_t = torch.tensor(labels, dtype=torch.float)
    if USE_LOG_RV:
        vv = torch.tensor(np.concatenate([[False], valid]), dtype=torch.bool)
        lbl_t[vv] = torch.log(lbl_t[vv].clamp(min=1e-8))

    valid_t = torch.tensor(valid, dtype=torch.bool)
    graphs_full.append((g, feats_full, lbl_t, valid_t))
    graphs_rv.append((g, feats_rv,   lbl_t, valid_t))

    if (w_idx + 1) % 30 == 0:
        print(f"  {w_idx+1}/{len(window_ends)}  {time.time()-t0:.1f}s")

X_full    = torch.tensor(np.array(X_full),    dtype=torch.float)
X_rv_only = torch.tensor(np.array(X_rv_only), dtype=torch.float)
y_all_t   = torch.tensor(np.array(y_flat),    dtype=torch.float)
n_samples = X_full.shape[0]

n_val = max(1, int(n_samples * VAL_FRAC))
n_val_g = max(1, int(len(graphs_full) * VAL_FRAC))

splits = {
    "full": (X_full[:-n_val], X_full[-n_val:],
             y_all_t[:-n_val], y_all_t[-n_val:],
             graphs_full[:-n_val_g], graphs_full[-n_val_g:]),
    "rv":   (X_rv_only[:-n_val], X_rv_only[-n_val:],
             y_all_t[:-n_val], y_all_t[-n_val:],
             graphs_rv[:-n_val_g], graphs_rv[-n_val_g:]),
}
print(f"Dataset: {n_samples} samples, {len(graphs_full)} graphs  "
      f"(train={n_samples-n_val}, val={n_val})")
print(f"Feature dims — full={X_full.shape[1]}  rv_only={X_rv_only.shape[1]}")


# ── Generic train helpers ──────────────────────────────────────────────────────
def train_mlp(X_trn, y_trn, X_val, y_val, in_dim, label):
    model = VolatilityMLP(in_dim=in_dim, hidden=GNN_HIDDEN,
                          mlp_hidden=MLP_HIDDEN, dropout=DROPOUT).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ld_t  = DataLoader(TensorDataset(X_trn, y_trn), batch_size=BATCH_SIZE, shuffle=True)

    best_val, pat = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in ld_t:
            opt.zero_grad()
            loss = torch.mean((model(xb.to(DEVICE)).squeeze(-1) - yb.to(DEVICE)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = torch.mean((model(X_val.to(DEVICE)).squeeze(-1) - y_val.to(DEVICE)) ** 2).item()

        if vl < best_val - 1e-6:
            best_val, pat = vl, 0
            torch.save(model.state_dict(), RESULTS / f"best_brv_{label}.pt")
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"    early stop epoch {epoch}  val={best_val:.5f}")
                break
    return model, best_val


def gnn_batch_loss(model, batch):
    total, n = 0.0, 0
    for gdata, feats, lbl, valid in batch:
        g    = gdata.g.to(DEVICE)
        x    = feats.to(DEVICE)
        lbl  = lbl.to(DEVICE)
        pred = model(g, x).squeeze(-1)
        vn30_valid = torch.zeros(31, dtype=torch.bool, device=DEVICE)
        vn30_valid[1:] = valid.to(DEVICE)
        mask = gdata.loss_mask.to(DEVICE) & vn30_valid
        if mask.sum() == 0:
            continue
        loss = torch.mean((pred[mask] - lbl[mask]) ** 2)
        total += loss * mask.sum().item()
        n     += mask.sum().item()
    return total / max(n, 1)


def train_gnn(graphs_trn, graphs_val, in_dim, label):
    model = VolatilityGNN(in_dim=in_dim, hidden=GNN_HIDDEN,
                          mlp_hidden=MLP_HIDDEN, dropout=DROPOUT).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_val, pat = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(graphs_trn)).tolist()
        for start in range(0, len(graphs_trn), BATCH_SIZE):
            batch = [graphs_trn[i] for i in perm[start:start + BATCH_SIZE]]
            opt.zero_grad()
            loss = gnn_batch_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = gnn_batch_loss(model, graphs_val).item()

        if vl < best_val - 1e-6:
            best_val, pat = vl, 0
            torch.save(model.state_dict(), RESULTS / f"best_brv_{label}.pt")
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"    early stop epoch {epoch}  val={best_val:.5f}")
                break
    return model, best_val


# ── Train all 4 variants ───────────────────────────────────────────────────────
variants = {
    "mlp_full": ("MLP  (Moirai2+RV6)", "mlp_full"),
    "gnn_full": ("GNN  (Moirai2+RV6)", "gnn_full"),
    "mlp_rv":   ("MLP  (RV6 only)",    "mlp_rv"),
    "gnn_rv":   ("GNN  (RV6 only)",    "gnn_rv"),
}

trained = {}
for key, (disp_name, label) in variants.items():
    feat_key = "full" if "full" in key else "rv"
    X_t, X_v, y_t, y_v, g_t, g_v = splits[feat_key]
    in_dim = X_t.shape[1]

    print(f"\n[Training] {disp_name}  in_dim={in_dim} …")
    t0 = time.time()
    if key.startswith("mlp"):
        model, bv = train_mlp(X_t, y_t, X_v, y_v, in_dim, label)
    else:
        model, bv = train_gnn(g_t, g_v, in_dim, label)
    print(f"  done in {time.time()-t0:.1f}s  best_val={bv:.6f}")
    trained[key] = (model, label, feat_key, in_dim)


# ── Inference on test set ──────────────────────────────────────────────────────
print("\n[Inference] Test set …")
embedder = Moirai2Embedder(
    size="small", context_length=CONTEXT_LEN,
    patch_size=cfg["model"].get("patch_size", 32), pooling=POOLING,
)

test_dates = rv_all.index[
    (rv_all.index >= test_start) & (~rv_all.isna().all(axis=1))
]

all_preds = {k: {tk: {} for tk in VN30_TICKERS} for k in trained}

t0 = time.time()
with torch.no_grad():
    for i, date in enumerate(test_dates):
        # Moirai2 embeddings (needed for full variants)
        embed = extract_embeddings(embedder, log_ret, date, CONTEXT_LEN)  # (31, 384)
        rv6   = get_extended_rv_features(log_ret, date, ALL_NODES, HORIZON)  # (31, 6)
        feats_full = torch.cat([embed, rv6], dim=-1).to(DEVICE)            # (31, 390)
        feats_rv   = rv6.to(DEVICE)                                        # (31, 6)

        g_data = build_graph(log_ret, end_date=date,
                             corr_window=cfg["model"]["corr_window"],
                             corr_threshold=cfg["model"]["corr_threshold"])

        for key, (model, label, feat_key, _) in trained.items():
            model.load_state_dict(
                torch.load(RESULTS / f"best_brv_{label}.pt", map_location=DEVICE)
            )
            model.eval()
            feats = feats_full if feat_key == "full" else feats_rv

            if key.startswith("mlp"):
                pred = model(feats[1:]).cpu().numpy().ravel()        # (30,)
                for j, tk in enumerate(VN30_TICKERS):
                    raw = float(pred[j])
                    all_preds[key][tk][date] = max(
                        float(np.exp(raw)) if USE_LOG_RV else raw, 0.0
                    )
            else:
                pred = model(g_data.g.to(DEVICE), feats).cpu().numpy().ravel()  # (31,)
                for j, tk in enumerate(VN30_TICKERS):
                    raw = float(pred[j + 1])
                    all_preds[key][tk][date] = max(
                        float(np.exp(raw)) if USE_LOG_RV else raw, 0.0
                    )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_dates)}  {time.time()-t0:.1f}s")

pred_dfs = {}
for key, disp in [("mlp_full","MLP_Moirai2_RV6"), ("gnn_full","GNN_Moirai2_RV6"),
                  ("mlp_rv","MLP_RV6only"), ("gnn_rv","GNN_RV6only")]:
    df = pd.DataFrame({t: pd.Series(all_preds[key][t]) for t in VN30_TICKERS})
    df.to_csv(RESULTS / f"pred_{disp}.csv")
    pred_dfs[key] = df
print(f"Saved 4 prediction CSVs.")


# ── Compare all models ─────────────────────────────────────────────────────────
print("\n[Comparison] All models …")

def pool(df):
    common = df.index.intersection(rv_all.index)
    yt = rv_all.loc[common].values.ravel()
    yp = df.loc[common].values.ravel()
    v  = ~(np.isnan(yt) | np.isnan(yp))
    return yt[v], yp[v]

yt, _ = pool(pred_dfs["mlp_full"])

har_df   = pd.read_csv(RESULTS / "pred_har.csv",   index_col=0, parse_dates=True)
lstm_df  = pd.read_csv(RESULTS / "pred_lstm.csv",  index_col=0, parse_dates=True)
bgnn_df  = pd.read_csv(RESULTS / "pred_batch_gnn.csv", index_col=0, parse_dates=True)
bmlp_df  = pd.read_csv(RESULTS / "pred_batch_mlp.csv", index_col=0, parse_dates=True)

n = len(yt)
def align(a): return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), np.nan)])

preds_dict = {
    "GNN  (Moirai2+RV6)": pool(pred_dfs["gnn_full"])[1],
    "MLP  (Moirai2+RV6)": pool(pred_dfs["mlp_full"])[1],
    "GNN  (RV6 only)":    pool(pred_dfs["gnn_rv"])[1],
    "MLP  (RV6 only)":    pool(pred_dfs["mlp_rv"])[1],
    "Batch GNN (Moirai2+RV3)": align(pool(bgnn_df)[1]),
    "Batch MLP (Moirai2+RV3)": align(pool(bmlp_df)[1]),
    "HAR-RV (OLS)":       align(pool(har_df)[1]),
    "LSTM":               align(pool(lstm_df)[1]),
}

results = compare_models(
    yt, preds_dict,
    dm_reference="GNN  (Moirai2+RV6)",
    dm_loss=cfg["evaluation"]["dm_loss"],
)

cols = ["MAE", "RMSE", "R2", "QLIKE", "Pearson_r", "DM_stat", "DM_pval"]
print("\n" + "=" * 76)
print("Extended RV Features Experiment — Batch Training")
print("=" * 76)
print(results[cols].round(4).to_string())
print("=" * 76)

best_model = results["MAE"].idxmin()
print(f"\nBest model by MAE: {best_model}  ({results.loc[best_model,'MAE']:.5f})")

# Q1: Extended RV vs original 3-feat
gnn_rv6   = results.loc["GNN  (RV6 only)",    "MAE"]
gnn_rv3   = results.loc["Batch GNN (Moirai2+RV3)", "MAE"]  # closest proxy
mlp_rv6   = results.loc["MLP  (RV6 only)",    "MAE"]
mlp_rv3   = results.loc["Batch MLP (Moirai2+RV3)", "MAE"]

# Q2: Moirai2 + RV6 vs RV6 only
gnn_full  = results.loc["GNN  (Moirai2+RV6)", "MAE"]
gnn_rv_only = results.loc["GNN  (RV6 only)",  "MAE"]
mlp_full  = results.loc["MLP  (Moirai2+RV6)", "MAE"]
mlp_rv_only = results.loc["MLP  (RV6 only)",  "MAE"]

print(f"\nQ1 — Does Moirai2 help on top of extended RV6?")
print(f"  GNN: Moirai2+RV6={gnn_full:.5f}  vs  RV6-only={gnn_rv_only:.5f}  "
      f"=> {'Moirai2 helps' if gnn_full < gnn_rv_only else 'RV6-only better'} "
      f"({(gnn_rv_only-gnn_full)/gnn_rv_only*100:+.1f}%)")
print(f"  MLP: Moirai2+RV6={mlp_full:.5f}  vs  RV6-only={mlp_rv_only:.5f}  "
      f"=> {'Moirai2 helps' if mlp_full < mlp_rv_only else 'RV6-only better'} "
      f"({(mlp_rv_only-mlp_full)/mlp_rv_only*100:+.1f}%)")

print(f"\nQ2 — Best neural model vs HAR-RV (OLS):")
har_mae = results.loc["HAR-RV (OLS)", "MAE"]
print(f"  HAR-RV MAE = {har_mae:.5f}")
print(f"  Best neural = {best_model}: {results.loc[best_model,'MAE']:.5f}  "
      f"(gap = {results.loc[best_model,'MAE']/har_mae:.1f}x)")

results.to_csv(RESULTS / "extended_rv_comparison.csv")
print(f"\nSaved: results/extended_rv_comparison.csv")
