"""
GNN inference on test set → compare all models → print results table.
Run after: run_gnn_train.py + run_baselines.py
"""
import os, sys, warnings, yaml, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

RESULTS    = "results"
TEST_START = pd.Timestamp(cfg["data"]["test_start"])
PRICES_DIR = cfg["data"]["prices_dir"]
HORIZON    = cfg["model"]["horizon"]

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv, get_rv_node_features
from src.embed_extractor import Moirai2Embedder
from gnn.model import VolatilityGNN
from gnn.build_graph import build_graph, VN30_TICKERS
from gnn.train import extract_embeddings
from evaluation.metrics import compare_models

device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LOG_RV    = cfg["model"].get("use_log_rv", False)
USE_LAGGED_RV = cfg["model"].get("use_lagged_rv", False)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading prices and RV labels...")
close   = load_close_prices(PRICES_DIR, tickers=VN30_TICKERS + ["VNINDEX"])
log_ret = compute_log_returns(close)
rv_all  = compute_rv(close[VN30_TICKERS], h=HORIZON)

test_dates = rv_all.index[
    (rv_all.index >= TEST_START) & (~rv_all.isna().all(axis=1))
]
print(f"Test dates: {len(test_dates)}  ({test_dates[0].date()} to {test_dates[-1].date()})")

# ── GNN inference ──────────────────────────────────────────────────────────
print("\nLoading GNN checkpoint...")
gnn_model = VolatilityGNN(
    in_dim     = Moirai2Embedder.D_MODEL + (3 if USE_LAGGED_RV else 0),
    hidden     = cfg["model"]["gnn_hidden"],
    mlp_hidden = cfg["model"]["mlp_hidden"],
    dropout    = cfg["model"]["dropout"],
).to(device)
gnn_model.load_state_dict(torch.load(f"{RESULTS}/best_gnn.pt", map_location=device))
gnn_model.eval()

embedder = Moirai2Embedder(
    size           = "small",
    context_length = cfg["model"]["context_length"],
    patch_size     = cfg["model"].get("patch_size", 32),
    pooling        = cfg["model"].get("pooling", "last_context"),
)

print(f"Running GNN inference on {len(test_dates)} test dates ({device})...")
t0 = time.time()
gnn_preds = {t: {} for t in VN30_TICKERS}
with torch.no_grad():
    for i, date in enumerate(test_dates):
        g     = build_graph(log_ret, end_date=date,
                            corr_window  = cfg["model"]["corr_window"],
                            corr_threshold = cfg["model"]["corr_threshold"])
        feats = extract_embeddings(embedder, log_ret, date,
                                   cfg["model"]["context_length"])
        if USE_LAGGED_RV:
            from gnn.build_graph import ALL_NODES
            rv_feats = get_rv_node_features(log_ret, date, ALL_NODES,
                                            cfg["model"]["horizon"])
            feats = torch.cat([feats, rv_feats], dim=-1)
        feats = feats.to(device)
        pred  = gnn_model(g.g.to(device), feats).cpu().numpy().ravel()
        for j, tk in enumerate(VN30_TICKERS):
            raw = float(pred[j + 1])
            gnn_preds[tk][date] = max(float(np.exp(raw)) if USE_LOG_RV else raw, 0.0)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_dates)}")

gnn_df = pd.DataFrame({t: pd.Series(gnn_preds[t]) for t in VN30_TICKERS})
gnn_df.to_csv(f"{RESULTS}/pred_gnn.csv")
print(f"GNN inference done in {time.time()-t0:.1f}s  shape={gnn_df.shape}")

# ── Load baseline predictions ──────────────────────────────────────────────
print("\nLoading baseline predictions...")
mlp_df   = pd.read_csv(f"{RESULTS}/pred_mlp.csv",   index_col=0, parse_dates=True)
garch_df = pd.read_csv(f"{RESULTS}/pred_garch.csv", index_col=0, parse_dates=True)
har_df   = pd.read_csv(f"{RESULTS}/pred_har.csv",   index_col=0, parse_dates=True)
lstm_df  = pd.read_csv(f"{RESULTS}/pred_lstm.csv",  index_col=0, parse_dates=True)

# ── Pool predictions across all stocks ────────────────────────────────────
def pool(pred_df, rv_df):
    common = pred_df.index.intersection(rv_df.index)
    yt = rv_df.loc[common].values.ravel()
    yp = pred_df.loc[common].values.ravel()
    v  = ~(np.isnan(yt) | np.isnan(yp))
    return yt[v], yp[v]

yt, yp_gnn   = pool(gnn_df,   rv_all)
_,  yp_mlp   = pool(mlp_df,   rv_all)
_,  yp_garch = pool(garch_df, rv_all)
_,  yp_har   = pool(har_df,   rv_all)
_,  yp_lstm  = pool(lstm_df,  rv_all)

n = len(yt)
def align(a):
    return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), np.nan)])

# ── Compare models ─────────────────────────────────────────────────────────
print("\nComputing metrics...")
results = compare_models(
    yt,
    {
        "GNN (ours)":  yp_gnn,
        "MLP+Moirai2": align(yp_mlp),
        "GARCH(1,1)":  align(yp_garch),
        "HAR-RV":      align(yp_har),
        "LSTM":        align(yp_lstm),
    },
    dm_reference = "GNN (ours)",
    dm_loss      = cfg["evaluation"]["dm_loss"],
)

# ── Print results ──────────────────────────────────────────────────────────
cols = ["MAE", "RMSE", "R2", "QLIKE", "Pearson_r", "DM_stat", "DM_pval"]
print("\n" + "=" * 70)
print("=== Model Comparison (test set, pooled 30 stocks) ===")
print("=" * 70)
print(results[cols].round(4).to_string())
print("=" * 70)

# Graph contribution
gnn_mae = results.loc["GNN (ours)", "MAE"]
mlp_mae = results.loc["MLP+Moirai2", "MAE"]
d = (mlp_mae - gnn_mae) / mlp_mae * 100
print(f"\nGraph contribution (MAE vs MLP): {d:+.1f}%  "
      f"=> {'Graph HELPS' if d > 0 else 'Graph does NOT help'}")

results.to_csv(f"{RESULTS}/model_comparison.csv")
print(f"\nSaved: {RESULTS}/model_comparison.csv")
