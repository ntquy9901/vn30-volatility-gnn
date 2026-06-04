"""
Moirai 2.0 demo - adapted from:
https://github.com/SalesforceAIResearch/uni2ts/blob/main/example/moirai_forecast.ipynb
"""
import os
import warnings
warnings.filterwarnings("ignore", message="The mean prediction is not stored")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.util import to_pandas

from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

os.makedirs("results", exist_ok=True)

# ── Config (từ notebook) ──────────────────────────────────────────
SIZE = "small"   # moirai2 hiện chỉ có 'small'
PDT  = 20        # prediction length
CTX  = 1000      # context length (moirai2 hỗ trợ dài hơn v1)
BSZ  = 32
# ─────────────────────────────────────────────────────────────────

# ── 1. Load demo dataset ─────────────────────────────────────────
print("Loading demo dataset...")
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)
print(f"  Shape: {df.shape}")

ds = PandasDataset(dict(df))
TEST = 100
_, test_template = split(ds, offset=-TEST)
test_data_inf = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)
print(f"  Test windows: {TEST // PDT} x {PDT} steps")

# ── 2. Load Moirai 2.0 model ─────────────────────────────────────
print(f"\nLoading Moirai 2.0-R-{SIZE} (downloading if needed)...")
model = Moirai2Forecast(
    module=Moirai2Module.from_pretrained(
        f"Salesforce/moirai-2.0-R-{SIZE}",
    ),
    prediction_length=PDT,
    context_length=CTX,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)
print("  Model loaded OK.")

# ── 3. Run inference ─────────────────────────────────────────────
print("\nRunning zero-shot inference...")
predictor   = model.create_predictor(batch_size=BSZ)
forecasts   = list(predictor.predict(test_data_inf.input))
print(f"  Forecasts: {len(forecasts)}")

# Check forecast type (Moirai2 may output point or distributional)
f0 = forecasts[0]
if hasattr(f0, "samples") and f0.samples is not None:
    print(f"  Forecast shape (samples x horizon): {f0.samples.shape}")
    is_probabilistic = True
else:
    print(f"  Forecast type: point (mean only) | horizon={PDT}")
    is_probabilistic = False

# ── 4. Visualize ─────────────────────────────────────────────────
print("\nSaving forecast plots...")
test_data_plot = test_template.generate_instances(PDT, windows=TEST // PDT, distance=PDT)

input_it    = iter(test_data_plot.input)
label_it    = iter(test_data_plot.label)
forecast_it = iter(forecasts)

for i in range(3):
    inp      = next(input_it)
    label    = next(label_it)
    forecast = next(forecast_it)

    fig, ax = plt.subplots(figsize=(12, 4))
    # Use plot_next_multi if available, fallback to manual plot
    try:
        plot_next_multi(
            [inp], [label], [forecast],
            context_length=min(CTX, 200),
            intervals=(0.5, 0.9),
            ax=ax,
        )
    except Exception:
        # Manual fallback plot
        ctx_vals = inp["target"][-200:]
        ax.plot(range(len(ctx_vals)), ctx_vals, color="steelblue", label="Context")
        if hasattr(forecast, "samples") and forecast.samples is not None:
            median = np.median(forecast.samples, axis=0)
            p10    = np.percentile(forecast.samples, 10, axis=0)
            p90    = np.percentile(forecast.samples, 90, axis=0)
            x_fc   = range(len(ctx_vals), len(ctx_vals) + PDT)
            ax.plot(x_fc, median, color="tomato", label="Median")
            ax.fill_between(x_fc, p10, p90, alpha=0.3, color="tomato", label="80% CI")
        else:
            x_fc = range(len(ctx_vals), len(ctx_vals) + PDT)
            ax.plot(x_fc, forecast.mean, color="tomato", label="Forecast")
        ax.plot(
            range(len(ctx_vals), len(ctx_vals) + PDT),
            label["target"],
            color="green", linestyle="--", label="Actual",
        )
        ax.legend(fontsize=9)

    ax.set_title(f"Moirai 2.0-R-{SIZE} Zero-Shot (window {i+1})")
    plt.tight_layout()
    path = f"results/moirai2_forecast_{i+1}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── 5. Metrics ───────────────────────────────────────────────────
print("\nComputing metrics...")
test_data_eval = test_template.generate_instances(PDT, windows=TEST // PDT, distance=PDT)
forecasts_eval = list(predictor.predict(test_data_eval.input))

mae_list, mse_list = [], []
for fcst, label_entry in zip(forecasts_eval, test_data_eval.label):
    actual = label_entry["target"]
    pred   = fcst.mean if hasattr(fcst, "mean") else np.median(fcst.samples, axis=0)
    mae_list.append(np.mean(np.abs(actual - pred)))
    mse_list.append(np.mean((actual - pred) ** 2))

mae  = np.mean(mae_list)
mse  = np.mean(mse_list)
rmse = np.sqrt(mse)

# CRPS (probabilistic, if available)
crps = None
if is_probabilistic:
    crps_list = []
    for fcst, label_entry in zip(forecasts_eval, test_data_eval.label):
        actual = label_entry["target"]
        qs = np.arange(0.05, 1.0, 0.05)
        pw = []
        for q in qs:
            q_pred = fcst.quantile(q)
            pw.append(np.mean(np.where(
                actual >= q_pred, q * (actual - q_pred), (1 - q) * (q_pred - actual)
            )))
        crps_list.append(np.mean(pw) * 2)
    crps = np.mean(crps_list)

print(f"\n=== Moirai 2.0-R-{SIZE} Zero-Shot Results ===")
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
if crps is not None:
    print(f"  CRPS : {crps:.4f}")

results = {"Model": f"Moirai2-{SIZE}", "MAE": round(mae,4),
           "MSE": round(mse,4), "RMSE": round(rmse,4)}
if crps is not None:
    results["CRPS"] = round(crps, 4)
pd.DataFrame([results]).to_csv("results/metrics_moirai2.csv", index=False)
print("\nSaved: results/metrics_moirai2.csv")
print("Done.")
