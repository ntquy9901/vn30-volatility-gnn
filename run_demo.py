"""
Demo: Moirai zero-shot inference - exact code from HuggingFace page
https://huggingface.co/Pranavv/moirai-base
"""
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.evaluation import Evaluator
import os

# --- Config ---
SIZE = "base"   # small | base | large
PDT  = 20
CTX  = 200
PSZ  = "auto"
BSZ  = 32
TEST = 100

os.makedirs("results", exist_ok=True)

# --- Load demo data ---
print("Loading demo dataset...")
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)
print(f"  Shape: {df.shape}")

ds = PandasDataset(dict(df))
train, test_template = split(ds, offset=-TEST)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# --- Load model (new API: MoiraiModule.from_pretrained) ---
print(f"Loading Moirai-1.0-R-{SIZE} (downloading safetensors if needed)...")

model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)
print("  Model loaded OK.")

# --- Predict ---
print("Running zero-shot inference...")
predictor = model.create_predictor(batch_size=BSZ)
forecasts = list(predictor.predict(test_data.input))
print(f"  Forecasts: {len(forecasts)} | Shape: {forecasts[0].samples.shape}")

# --- Plot 3 windows ---
print("Saving forecast plots...")
test_data2 = test_template.generate_instances(PDT, windows=TEST // PDT, distance=PDT)
input_it    = iter(test_data2.input)
label_it    = iter(test_data2.label)
forecast_it = iter(forecasts)

for i in range(3):
    inp      = next(input_it)
    label    = next(label_it)
    forecast = next(forecast_it)
    fig, ax  = plt.subplots(figsize=(12, 4))
    plot_single(inp, label, forecast, context_length=CTX,
                name=f"Window {i+1}", show_label=True, ax=ax)
    ax.set_title(f"Moirai BASE Zero-Shot (window {i+1})")
    plt.tight_layout()
    path = f"results/demo_forecast_{i+1}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# --- Metrics ---
print("Computing metrics...")
test_data3 = test_template.generate_instances(PDT, windows=TEST // PDT, distance=PDT)
forecasts3 = list(predictor.predict(test_data3.input))

from gluonts.dataset.util import to_pandas

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg, _ = evaluator(
    ts_iterator=(to_pandas(e) for e in test_data3.label),
    fcst_iterator=iter(forecasts3),
)

print("\n=== Moirai BASE Zero-Shot Results ===")
for label, key in [("MAE","MAE"),("MSE","MSE"),("MASE","MASE"),("CRPS","mean_wQuantileLoss")]:
    v = agg.get(key)
    if v is not None:
        print(f"  {label:6s}: {v:.4f}")

results = {k: round(agg.get(v), 4) for k, v in
           [("MAE","MAE"),("MSE","MSE"),("MASE","MASE"),("CRPS","mean_wQuantileLoss")]
           if agg.get(v) is not None}
pd.DataFrame([results]).to_csv("results/metrics_demo.csv", index=False)
print("\nSaved: results/metrics_demo.csv")
print("Done.")
