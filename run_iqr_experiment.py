"""
run_iqr_experiment.py — Moirai2 IQR Quantile Spread as Volatility Proxy

Instead of hooking intermediate embeddings, this experiment uses Moirai2's
own forecast output directly:

    IQR_t = q90(r_{t+1}) - q10(r_{t+1})   [1-step-ahead return distribution]

Hypothesis: If Moirai2 is calibrated, its return uncertainty (IQR) should
            track realized volatility via volatility clustering.

Comparison:
  - IQR Pearson_r vs RV          ← this experiment
  - Embedding median|corr| = 0.128 ← existing approach (diag_pooling_compare)
  - Best neural Pearson_r = 0.551  ← MLP(RV6-only), batch-trained

If IQR_r > 0.3  → better than raw embeddings; potentially useful as feature
If IQR_r < 0.128 → using quantile output is WORSE than embedding approach
Either way, confirms or contradicts the domain-gap hypothesis.
"""
import os, sys, warnings, yaml
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, ".")

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv

# ── Config ───────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

PRICES_DIR  = Path(cfg["data"]["prices_dir"])
CONTEXT_LEN = cfg["model"]["context_length"]      # 200
HORIZON     = cfg["model"]["horizon"]              # 20
# Moirai2 returns QuantileForecast (9 fixed quantile levels: 0.1 to 0.9)
# No num_samples needed — quantiles are exact, not sampled
TEST_START  = pd.Timestamp("2026-01-05")
TEST_END    = pd.Timestamp("2026-04-07")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SHB", "SSB",
    "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM",
]


# ── Forecast IQR extraction ──────────────────────────────────────────────────

def get_forecast_iqr(module, log_returns, window_end, tickers, context_length=200):
    """
    Run Moirai2 1-step-ahead forecast for all tickers at window_end.
    Returns {ticker: IQR_scalar} where IQR = q0.9 - q0.1 of predicted return dist.

    Moirai2 returns QuantileForecast with 9 exact quantile levels (0.1..0.9).
    IQR uses the outermost quantiles (q0.1, q0.9) = 80% prediction interval width.

    Data: last `context_length` log-returns ending at window_end (no lookahead).
    Forecast target: return at window_end+1 (genuine 1-step-ahead).
    """
    data_list     = []
    valid_tickers = []

    for ticker in tickers:
        if ticker not in log_returns.columns:
            continue
        s = log_returns[ticker][log_returns[ticker].index <= window_end].dropna()
        if len(s) < context_length:
            continue
        s_ctx = s.iloc[-context_length:]
        data_list.append({
            "start":  s_ctx.index[0],
            "target": s_ctx.values.astype(np.float32),
        })
        valid_tickers.append(ticker)

    if not data_list:
        return {}

    # ListDataset: feed entire series as context → forecast 1 step beyond end
    ds = ListDataset(data_list, freq="B")

    forecast_model = Moirai2Forecast(
        module=module,
        prediction_length=1,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = forecast_model.create_predictor(batch_size=len(valid_tickers))

    with torch.no_grad():
        forecasts = list(predictor.predict(iter(ds)))

    iqr_dict = {}
    for ticker, fc in zip(valid_tickers, forecasts):
        # QuantileForecast: fc.quantile('0.1') returns array shape (prediction_length,)
        q10 = float(fc.quantile("0.1")[0])
        q90 = float(fc.quantile("0.9")[0])
        iqr_dict[ticker] = max(q90 - q10, 0.0)

    return iqr_dict


# ── Diagnostic: inspect one forecast object ──────────────────────────────────

def diagnose_forecast(module, log_returns, tickers, context_length):
    """Print forecast object type and quantile values for one stock (debug)."""
    for ticker in tickers:
        s = log_returns[ticker].dropna()
        if len(s) < context_length:
            continue
        s_ctx = s.iloc[-context_length:]
        ds = ListDataset([{"start": s_ctx.index[0],
                           "target": s_ctx.values.astype(np.float32)}], freq="B")
        fm = Moirai2Forecast(
            module=module, prediction_length=1, context_length=context_length,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        predictor = fm.create_predictor(batch_size=1)
        with torch.no_grad():
            fc = next(predictor.predict(iter(ds)))
        print(f"  Forecast type : {type(fc).__name__}")
        print(f"  Quantile keys : {fc.forecast_keys}")
        print(f"  Array shape   : {fc.forecast_array.shape}  (n_quantiles × pred_len)")
        for q in ["0.1", "0.5", "0.9"]:
            print(f"  q{q} = {float(fc.quantile(q)[0]):+.6f}")
        q10 = float(fc.quantile("0.1")[0])
        q90 = float(fc.quantile("0.9")[0])
        print(f"  IQR (q0.9-q0.1) = {q90-q10:.6f}  (in log-return units)")
        break


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Moirai2 IQR Quantile Spread — Volatility Proxy Experiment")
    print(f"  context={CONTEXT_LEN}d | pred_length=1 | QuantileForecast (9 exact quantiles)")
    print(f"  RV horizon={HORIZON}d | test={TEST_START.date()} -> {TEST_END.date()}")
    print("=" * 70)

    # Load data
    print("\nLoading prices & computing log-returns / RV labels...")
    close       = load_close_prices(PRICES_DIR, tickers=VN30_TICKERS)
    log_returns = compute_log_returns(close)
    rv_labels   = compute_rv(close, h=HORIZON)

    # Test dates = valid RV label dates in test window
    test_dates = rv_labels.loc[TEST_START:TEST_END].dropna(how="all").index
    print(f"  Test dates: {test_dates[0].date()} -> {test_dates[-1].date()} "
          f"({len(test_dates)} dates × {len(VN30_TICKERS)} stocks)")

    # Load Moirai2
    print("\nLoading Moirai2-small (frozen)...")
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    module.eval()

    # Diagnose forecast format once
    print("\n[Diagnostic] Inspecting forecast object format...")
    diagnose_forecast(module, log_returns, VN30_TICKERS, CONTEXT_LEN)

    # Main loop: collect IQR + RV for all (date, stock) pairs
    print(f"\nRunning {len(test_dates)} forecast batches (30 stocks each)...")
    records = []

    for i, date in enumerate(test_dates):
        if i % 10 == 0:
            print(f"  [{i+1:3d}/{len(test_dates)}] {date.date()}")

        iqr_dict = get_forecast_iqr(
            module, log_returns, date, VN30_TICKERS,
            context_length=CONTEXT_LEN,
        )

        for ticker, iqr_val in iqr_dict.items():
            if ticker not in rv_labels.columns:
                continue
            if date not in rv_labels.index:
                continue
            rv_val = float(rv_labels.loc[date, ticker])
            if np.isnan(rv_val) or np.isnan(iqr_val) or iqr_val <= 0:
                continue
            records.append({"date": date, "ticker": ticker,
                            "IQR": iqr_val, "RV": rv_val})

    df = pd.DataFrame(records)
    n  = len(df)
    print(f"\nCollected {n} valid (date, stock) pairs")

    if n == 0:
        print("ERROR: No valid pairs — check data alignment.")
        return

    iqr = df["IQR"].values
    rv  = df["RV"].values

    # ── Metrics ──────────────────────────────────────────────────────────────
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, r2_score

    r, pval   = pearsonr(iqr, rv)
    mae_raw   = mean_absolute_error(rv, iqr)
    r2_raw    = r2_score(rv, iqr)

    # Scale correction: IQR ≈ 1.35×std for Gaussian → scale to RV units
    scale     = np.mean(rv) / np.mean(iqr)
    iqr_sc    = iqr * scale
    mae_sc    = mean_absolute_error(rv, iqr_sc)
    r2_sc     = r2_score(rv, iqr_sc)

    # Per-stock correlation (same way diag_d2 computes embedding|corr|)
    per_stock = []
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        if len(sub) < 5:
            continue
        r_s, _ = pearsonr(sub["IQR"].values, sub["RV"].values)
        per_stock.append(abs(r_s))
    median_per_stock = float(np.median(per_stock)) if per_stock else np.nan

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS — Moirai2 IQR vs Realized Volatility")
    print(f"{'='*65}")
    print(f"  Pearson_r (pooled)      : {r:+.4f}   (p={pval:.2e})")
    print(f"  Median |r| per stock    : {median_per_stock:.4f}")
    print(f"  MAE (raw IQR)           : {mae_raw:.6f}")
    print(f"  MAE (IQR×{scale:.3f})      : {mae_sc:.6f}")
    print(f"  R²  (raw IQR)           : {r2_raw:+.4f}")
    print(f"  R²  (IQR scaled)        : {r2_sc:+.4f}")
    print(f"\n  --- Comparison with existing approaches ---")
    print(f"  Embedding median|corr|  : 0.1276  (last_context pooling)")
    print(f"  IQR median|r| per stock : {median_per_stock:.4f}")
    print(f"  MLP (RV6) Pearson_r     : 0.551")
    print(f"  HAR-RV    Pearson_r     : 0.985")
    print(f"{'='*65}")

    if r > 0.3:
        verdict = f"PROMISING — IQR (r={r:.3f}) > 0.3; better than embeddings (0.128)"
    elif r > 0.128:
        verdict = f"MARGINAL — IQR (r={r:.3f}) slightly better than embeddings (0.128)"
    else:
        verdict = f"WEAK — IQR (r={r:.3f}) ≤ embedding median|corr| (0.128)"

    print(f"\n  Verdict: {verdict}")
    print(f"  Domain-gap hypothesis: ", end="")
    if r < 0.2:
        print("CONFIRMED — Moirai2 forecast output also weak for RV, consistent")
        print("  with pretraining on macro/competition data (no equity data)")
    else:
        print("PARTIALLY REJECTED — Moirai2 quantile uncertainty tracks RV")
        print("  Suggests model has learned some vol-like signal despite domain gap")

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(RESULTS_DIR / "iqr_experiment.csv", index=False)

    summary = {
        "model": "Moirai2 IQR (q90-q10, 1-step forecast)",
        "Pearson_r": round(r, 4),
        "median_abs_r_per_stock": round(median_per_stock, 4),
        "MAE_raw": round(mae_raw, 6),
        "MAE_scaled": round(mae_sc, 6),
        "R2_raw": round(r2_raw, 4),
        "R2_scaled": round(r2_sc, 4),
        "scale_factor": round(scale, 4),
        "n_pairs": n,
        "quantile_levels": "0.1,0.2,...,0.9 (exact, QuantileForecast)",
        "context_length": CONTEXT_LEN,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "iqr_summary.csv", index=False)
    print(f"\n  Saved: results/iqr_experiment.csv, results/iqr_summary.csv")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(iqr, rv, alpha=0.25, s=8, c="steelblue")
        axes[0].set_xlabel("Moirai2 IQR (q90−q10, raw log-return units)")
        axes[0].set_ylabel("Realized Volatility (h=20)")
        axes[0].set_title(f"IQR vs RV  [r={r:.3f}, R²={r2_raw:.3f}]")
        axes[0].grid(True, alpha=0.3)

        lim = max(rv.max(), iqr_sc.max()) * 1.05
        axes[1].scatter(iqr_sc, rv, alpha=0.25, s=8, c="tomato")
        axes[1].plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x")
        axes[1].set_xlabel(f"Moirai2 IQR × {scale:.3f} (scaled to RV units)")
        axes[1].set_ylabel("Realized Volatility (h=20)")
        axes[1].set_title(f"IQR (scaled) vs RV  [MAE={mae_sc:.4f}, R²={r2_sc:.3f}]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        plt.suptitle(
            f"Moirai2 IQR Quantile Spread as Volatility Proxy\n"
            f"{len(test_dates)} test dates × 30 VN30 stocks = {n} pairs  "
            f"|  QuantileForecast (q0.1 to q0.9)",
            y=1.02, fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "iqr_vs_rv.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: results/iqr_vs_rv.png")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
