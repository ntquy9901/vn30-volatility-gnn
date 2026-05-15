"""Run all 4 baselines and save results to results/."""
import os, sys, time, yaml
sys.path.insert(0, ".")
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs("results", exist_ok=True)

kwargs = dict(
    prices_dir  = cfg["data"]["prices_dir"],
    train_end   = cfg["data"]["train_end"],
    test_start  = cfg["data"]["test_start"],
    horizon     = cfg["model"]["horizon"],
)

# ── 1. MLP + Moirai2 ────────────────────────────────────────────────────────
print("=" * 50)
print("[1/4] MLP + Moirai2 (no graph)...")
t0 = time.time()
from baselines.mlp_baseline import train_mlp_walkforward, run_mlp_inference
train_mlp_walkforward(cfg, results_dir="results")
mlp_df = pd.DataFrame(run_mlp_inference(cfg, checkpoint_path="results/best_mlp.pt",
                                         results_dir="results"))
mlp_df.to_csv("results/pred_mlp.csv")
print(f"  Done in {time.time()-t0:.1f}s  shape={mlp_df.shape}")

# ── 2. GARCH(1,1) ───────────────────────────────────────────────────────────
print("[2/4] GARCH(1,1)...")
t0 = time.time()
from baselines.garch_baseline import run_garch_baseline
garch_df = pd.DataFrame(run_garch_baseline(**kwargs))
garch_df.to_csv("results/pred_garch.csv")
print(f"  Done in {time.time()-t0:.1f}s  shape={garch_df.shape}")

# ── 3. HAR-RV ───────────────────────────────────────────────────────────────
print("[3/4] HAR-RV...")
t0 = time.time()
from baselines.har_rv_baseline import run_har_baseline
har_df = pd.DataFrame(run_har_baseline(**kwargs))
har_df.to_csv("results/pred_har.csv")
print(f"  Done in {time.time()-t0:.1f}s  shape={har_df.shape}")

# ── 4. LSTM ─────────────────────────────────────────────────────────────────
print("[4/4] LSTM...")
t0 = time.time()
from baselines.lstm_baseline import run_lstm_baseline
lstm_df = pd.DataFrame(run_lstm_baseline(**kwargs))
lstm_df.to_csv("results/pred_lstm.csv")
print(f"  Done in {time.time()-t0:.1f}s  shape={lstm_df.shape}")

print("=" * 50)
print("All baselines done. CSVs saved to results/")
