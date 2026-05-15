"""Run full GNN walk-forward training — no smoke-test overrides."""
import os, sys, yaml
sys.path.insert(0, ".")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

print(f"Windows estimate: stride={cfg['model']['stride']} days, "
      f"epochs={cfg['training']['epochs']}, "
      f"early_stop={cfg['training']['early_stopping']}")

from gnn.train import train_walkforward
import time

os.makedirs("results", exist_ok=True)
t0 = time.time()
metrics = train_walkforward(cfg, results_dir="results")
elapsed = time.time() - t0

print(f"\nDone: {len(metrics)} windows in {elapsed/60:.1f} min")
if len(metrics):
    print(metrics.tail(5).to_string(index=False))
