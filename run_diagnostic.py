"""Diagnose GNN prediction magnitudes vs true RV."""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import yaml, numpy as np, pandas as pd

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS

close   = load_close_prices(cfg["data"]["prices_dir"], tickers=VN30_TICKERS)
rv_all  = compute_rv(close[VN30_TICKERS], h=cfg["model"]["horizon"])
TEST_START = pd.Timestamp(cfg["data"]["test_start"])

gnn_df   = pd.read_csv("results/pred_gnn.csv",   index_col=0, parse_dates=True)
har_df   = pd.read_csv("results/pred_har.csv",   index_col=0, parse_dates=True)
garch_df = pd.read_csv("results/pred_garch.csv", index_col=0, parse_dates=True)

common = rv_all.index.intersection(gnn_df.index)
rv_test  = rv_all.loc[common]
gnn_test = gnn_df.loc[common]
har_test = har_df.reindex(common)

print("=" * 60)
print("TRUE RV (test set, all 30 stocks pooled):")
rv_vals = rv_test.values.ravel()
rv_vals = rv_vals[~np.isnan(rv_vals)]
print(f"  min={rv_vals.min():.6f}  mean={rv_vals.mean():.6f}  "
      f"max={rv_vals.max():.6f}  std={rv_vals.std():.6f}")

print("\nGNN PREDICTIONS (test set):")
gnn_vals = gnn_test.values.ravel()
gnn_vals = gnn_vals[~np.isnan(gnn_vals)]
print(f"  min={gnn_vals.min():.6f}  mean={gnn_vals.mean():.6f}  "
      f"max={gnn_vals.max():.6f}  std={gnn_vals.std():.6f}")

print("\nHAR-RV PREDICTIONS (test set):")
har_vals = har_test.values.ravel()
har_vals = har_vals[~np.isnan(har_vals)]
print(f"  min={har_vals.min():.6f}  mean={har_vals.mean():.6f}  "
      f"max={har_vals.max():.6f}  std={har_vals.std():.6f}")

print("\n--- Scale ratio ---")
print(f"  True RV mean   : {rv_vals.mean():.6f}")
print(f"  GNN pred mean  : {gnn_vals.mean():.6f}  ratio={gnn_vals.mean()/rv_vals.mean():.2f}x")
print(f"  HAR pred mean  : {har_vals.mean():.6f}  ratio={har_vals.mean()/rv_vals.mean():.2f}x")

print("\n--- Per-stock sample (5 stocks) ---")
for tk in ["VCB", "HPG", "VHM", "FPT", "MBB"]:
    if tk not in rv_test.columns or tk not in gnn_test.columns:
        continue
    rv_s   = rv_test[tk].dropna()
    gnn_s  = gnn_test[tk].reindex(rv_s.index).dropna()
    idx    = rv_s.index.intersection(gnn_s.index)
    print(f"  {tk}: true_mean={rv_s[idx].mean():.5f}  "
          f"gnn_mean={gnn_s[idx].mean():.5f}  "
          f"ratio={gnn_s[idx].mean()/rv_s[idx].mean():.2f}x")

print("\n--- log(RV) stats (what we should train on) ---")
log_rv = np.log(rv_vals + 1e-8)
print(f"  log(RV): mean={log_rv.mean():.4f}  std={log_rv.std():.4f}  "
      f"min={log_rv.min():.4f}  max={log_rv.max():.4f}")
print(f"  RV range: [{np.exp(log_rv.min()):.5f}, {np.exp(log_rv.max()):.5f}]")
