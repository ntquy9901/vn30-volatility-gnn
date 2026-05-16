"""
D3 Diagnostic: Neural HAR-RV
MLP trained on ONLY 3 lagged-RV features [log(RV_d), log(RV_w), log(RV_m)].

Tests H3: "Lagged RV features contain all useful signal;
           Moirai2 embeddings add noise, not signal."

Expected outcomes:
  - If neural_HAR ≈ HAR-RV >> GNN  → H3 confirmed (RV signal overwhelmed by embedding noise)
  - If neural_HAR << HAR-RV        → nonlinear RV model is harder than OLS
  - If neural_HAR >> GNN           → 3 features beat 387 features → noise hypothesis
"""
import os, sys, warnings, yaml, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.volatility_labels import (
    load_close_prices, compute_log_returns, compute_rv, get_rv_node_features
)
from gnn.build_graph import VN30_TICKERS, ALL_NODES
from gnn.train import walk_forward_dates
from evaluation.metrics import compare_models

RESULTS   = Path("results")
DEVICE    = torch.device("cpu")
HORIZON   = cfg["model"]["horizon"]
USE_LOG   = cfg["model"].get("use_log_rv", True)


# ── Model ──────────────────────────────────────────────────────────────────────
class NeuralHAR(nn.Module):
    """MLP(3 → 32 → 16 → 1) — same depth as VolatilityMLP but tiny input."""
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading data...")
close   = load_close_prices(cfg["data"]["prices_dir"], tickers=VN30_TICKERS + ["VNINDEX"])
close   = close[close.index >= pd.Timestamp(cfg["data"].get("data_start", "2014-06-27"))]
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

train_end  = pd.Timestamp(cfg["data"]["train_end"])
test_start = pd.Timestamp(cfg["data"]["test_start"])
context_len = cfg["model"]["context_length"]
stride      = cfg["model"]["stride"]
epochs      = cfg["training"]["epochs"]
patience    = cfg["training"]["early_stopping"]
grad_clip   = cfg["training"].get("grad_clip", 1.0)

import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

model     = NeuralHAR(dropout=cfg["model"]["dropout"]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"],
                       weight_decay=cfg["training"].get("weight_decay", 1e-4))

train_dates  = pd.DatetimeIndex(log_ret.index)
window_ends  = walk_forward_dates(train_dates, train_end, context_len, stride)
print(f"Walk-forward windows: {len(window_ends)}")


# ── Training ───────────────────────────────────────────────────────────────────
print("Training Neural HAR-RV...")
t0 = time.time()
best_loss, patience_ctr = float("inf"), 0

for w_idx, window_end in enumerate(window_ends):
    rv_feats = get_rv_node_features(log_ret, window_end, ALL_NODES, HORIZON)
    # VN30 stocks only: indices 1-30
    x_vn30 = rv_feats[1:].to(DEVICE)           # (30, 3)

    rv_at = rv_all.loc[window_end] if window_end in rv_all.index else None
    if rv_at is None or rv_at.isna().all():
        continue

    labels = torch.tensor(
        [float(rv_at.get(tk, np.nan) or 0.0) for tk in VN30_TICKERS],
        dtype=torch.float, device=DEVICE,
    )
    if USE_LOG:
        labels = torch.log(labels.clamp(min=1e-8))
    valid = ~torch.isnan(labels)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_vn30).squeeze(-1)         # (30,)
        loss = torch.mean((pred[valid] - labels[valid]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if loss.item() < best_loss - 1e-6:
        best_loss = loss.item()
        patience_ctr = 0
        torch.save(model.state_dict(), RESULTS / "best_neural_har.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print(f"  Early stop at window {w_idx+1}")
            break

    if (w_idx + 1) % 20 == 0:
        print(f"  window {w_idx+1}/{len(window_ends)}  loss={loss.item():.5f}")

if not (RESULTS / "best_neural_har.pt").exists():
    torch.save(model.state_dict(), RESULTS / "best_neural_har.pt")

print(f"Training done in {time.time()-t0:.1f}s  best_loss={best_loss:.6f}")


# ── Inference ──────────────────────────────────────────────────────────────────
print("\nRunning inference on test set...")
model.load_state_dict(torch.load(RESULTS / "best_neural_har.pt", map_location=DEVICE))
model.eval()

test_dates = rv_all.index[
    (rv_all.index >= test_start) & (~rv_all.isna().all(axis=1))
]

neural_har_preds = {tk: {} for tk in VN30_TICKERS}
with torch.no_grad():
    for date in test_dates:
        rv_feats = get_rv_node_features(log_ret, date, ALL_NODES, HORIZON)
        x_vn30  = rv_feats[1:].to(DEVICE)
        pred    = model(x_vn30).cpu().numpy().ravel()   # (30,)
        for i, tk in enumerate(VN30_TICKERS):
            raw = float(pred[i])
            neural_har_preds[tk][date] = max(
                float(np.exp(raw)) if USE_LOG else raw, 0.0
            )

nhar_df = pd.DataFrame({t: pd.Series(neural_har_preds[t]) for t in VN30_TICKERS})
nhar_df.to_csv(RESULTS / "pred_neural_har.csv")
print(f"Saved: results/pred_neural_har.csv  shape={nhar_df.shape}")


# ── Compare ────────────────────────────────────────────────────────────────────
print("\nComparing all models...")

def pool(pred_df):
    common = pred_df.index.intersection(rv_all.index)
    yt = rv_all.loc[common].values.ravel()
    yp = pred_df.loc[common].values.ravel()
    v  = ~(np.isnan(yt) | np.isnan(yp))
    return yt[v], yp[v]

yt, yp_nhar  = pool(nhar_df)

gnn_df   = pd.read_csv(RESULTS / "pred_gnn.csv",   index_col=0, parse_dates=True)
mlp_df   = pd.read_csv(RESULTS / "pred_mlp.csv",   index_col=0, parse_dates=True)
har_df   = pd.read_csv(RESULTS / "pred_har.csv",   index_col=0, parse_dates=True)
lstm_df  = pd.read_csv(RESULTS / "pred_lstm.csv",  index_col=0, parse_dates=True)

_, yp_gnn  = pool(gnn_df)
_, yp_mlp  = pool(mlp_df)
_, yp_har  = pool(har_df)
_, yp_lstm = pool(lstm_df)

n = len(yt)
def align(a):
    return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), np.nan)])

results = compare_models(
    yt,
    {
        "Neural HAR (3-feat)": yp_nhar,
        "GNN (387-feat)":      align(yp_gnn),
        "MLP+Moirai2 (387)":   align(yp_mlp),
        "HAR-RV (OLS)":        align(yp_har),
        "LSTM":                align(yp_lstm),
    },
    dm_reference = "Neural HAR (3-feat)",
    dm_loss      = cfg["evaluation"]["dm_loss"],
)

cols = ["MAE", "RMSE", "R2", "QLIKE", "Pearson_r", "DM_stat", "DM_pval"]
print("\n" + "=" * 72)
print("D3 Result: Neural HAR (3 lagged-RV features) vs all models")
print("=" * 72)
print(results[cols].round(4).to_string())
print("=" * 72)

nhar_mae = results.loc["Neural HAR (3-feat)", "MAE"]
gnn_mae  = results.loc["GNN (387-feat)",      "MAE"]
mlp_mae  = results.loc["MLP+Moirai2 (387)",   "MAE"]
har_mae  = results.loc["HAR-RV (OLS)",        "MAE"]

print(f"\n[H3 test] Neural HAR (3 feats) MAE={nhar_mae:.5f}")
print(f"          GNN          (387 feats) MAE={gnn_mae:.5f}   ratio={gnn_mae/nhar_mae:.2f}x worse")
print(f"          MLP+Moirai2  (387 feats) MAE={mlp_mae:.5f}   ratio={mlp_mae/nhar_mae:.2f}x worse")
print(f"          HAR-RV (OLS, 3 feats)    MAE={har_mae:.5f}   ratio={nhar_mae/har_mae:.2f}x worse")

if nhar_mae < gnn_mae and nhar_mae < mlp_mae:
    print("\n  => H3 CONFIRMED: 3 lagged-RV features outperform 387-dim Moirai2 features.")
    print("     Moirai2 embeddings add noise, not signal, for volatility prediction.")
else:
    print("\n  => H3 NOT confirmed: Moirai2 embeddings carry some useful signal beyond RV.")

results.to_csv(RESULTS / "diag_d3_comparison.csv")
print(f"\nSaved: results/diag_d3_comparison.csv")
