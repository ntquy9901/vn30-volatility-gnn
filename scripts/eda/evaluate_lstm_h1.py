"""
Evaluate trained LSTM h=1 models on test set (2026-01-01 onwards).

For each model saved in models/lstm_h1/{ticker}_model.pt:
  1. Load model + normalization constants
  2. Rebuild test sequences (with pre-test tail as context)
  3. Predict + denormalize
  4. Compute MAE, RMSE, R², Directional Accuracy vs naïve baseline
  5. Aggregate and produce summary table + metrics chart

Output:
  results/lstm_h1_results.csv          — per-stock metrics (overwritten)
  results/lstm_h1_summary_metrics.png  — 4-panel metrics chart

Usage:
  python scripts/eda/evaluate_lstm_h1.py
  python scripts/eda/evaluate_lstm_h1.py --tickers VCB HPG FPT
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns

# ─────────────────────────── CONFIG ───────────────────────────────────────────
GLOBAL_TEST_START     = "2026-01-01"   # R6
TRAIN_VAL_SPLIT_RATIO = 0.8            # R6
LOOKBACK              = 20             # must match training config
HORIZONS              = [1]
MAX_H                 = max(HORIZONS)
HIDDEN                = 32

_root       = Path(__file__).parent.parent.parent
MODELS_DIR  = _root / "models" / "lstm_h1"
DATA_DIR    = _root / "data" / "raw" / "prices"
RESULTS_DIR = _root / "results"
CSV_OUT     = RESULTS_DIR / "lstm_h1_results.csv"
PLOT_OUT    = RESULTS_DIR / "lstm_h1_summary_metrics.png"


# ─────────────────────────── MODEL (must match train_lstm_h1.py) ──────────────
class LSTMModelMIMO(nn.Module):
    """Mirrors baselines/train_lstm_h1.py — same architecture for checkpoint loading."""

    def __init__(self, horizons=HORIZONS, hidden=HIDDEN, dropout=0.1):
        super().__init__()
        self.horizons = horizons
        self.lstm  = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        feat = self.drop(out[:, -1, :])
        return torch.cat([h(feat) for h in self.heads], dim=1)


# ─────────────────────────── HELPERS ──────────────────────────────────────────
def compute_rv_h1(close: pd.DataFrame) -> pd.DataFrame:
    """h=1 RV proxy: |log(P_{t+1}/P_t)| — must match train_lstm_h1.py."""
    log_ret = compute_log_returns(close)
    return log_ret.abs().shift(-1)


def make_sequences(rv_arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(rv_arr) - lookback):
        X.append(rv_arr[i : i + lookback])
        y.append([rv_arr[i + lookback]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_r = float(np.sum((y_true - y_pred) ** 2))
    ss_t = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = float(1.0 - ss_r / (ss_t + 1e-12))
    da   = (float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
            if len(y_true) > 1 else float("nan"))
    return dict(mae=mae, rmse=rmse, r2=r2, da=da)


# ─────────────────────────── EVALUATE ONE STOCK ───────────────────────────────
def evaluate_ticker(ticker: str, rv_series: pd.Series) -> dict | None:
    model_path = MODELS_DIR / f"{ticker}_model.pt"
    if not model_path.exists():
        print(f"  [{ticker}] Model not found: {model_path}")
        return None

    # load checkpoint
    ckpt     = torch.load(model_path, map_location="cpu", weights_only=False)
    rv_mu    = float(ckpt["rv_mu"])
    rv_sig   = float(ckpt["rv_sig"])
    hidden   = int(ckpt.get("hidden",   HIDDEN))
    lookback = int(ckpt.get("lookback", LOOKBACK))
    horizons = ckpt.get("horizons", HORIZONS)

    model = LSTMModelMIMO(horizons=horizons, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── reconstruct splits (must match train_lstm_h1.py logic) ───────────────
    test_ts = pd.Timestamp(GLOBAL_TEST_START)
    pre     = rv_series[rv_series.index < test_ts].dropna()
    test_rv = rv_series[rv_series.index >= test_ts].dropna()

    if len(test_rv) == 0:
        print(f"  [{ticker}] No test data (2026-01-01 onwards)")
        return None

    n_tr     = int(len(pre) * TRAIN_VAL_SPLIT_RATIO)
    train_rv = pre.iloc[:n_tr]
    val_rv   = pre.iloc[n_tr:]

    # ── build test sequences ──────────────────────────────────────────────────
    pre_n    = (pre.values  - rv_mu) / rv_sig
    test_n   = (test_rv.values - rv_mu) / rv_sig
    test_ctx = np.concatenate([pre_n[-lookback:], test_n])
    X_te, y_te = make_sequences(test_ctx, lookback)

    if len(X_te) == 0:
        print(f"  [{ticker}] Insufficient test sequences")
        return None

    Xte_t = torch.tensor(X_te).unsqueeze(-1)
    with torch.no_grad():
        preds_n = model(Xte_t).numpy()

    y_true = y_te[:, 0] * rv_sig + rv_mu
    y_pred = np.clip(preds_n[:, 0] * rv_sig + rv_mu, 0.0, None)

    # naïve baseline: predict constant = mean of train RV
    y_base = np.full_like(y_true, float(train_rv.mean()))

    m_lstm = compute_metrics(y_true, y_pred)
    m_base = compute_metrics(y_true, y_base)

    print(f"  [{ticker}] LSTM:     "
          f"MAE={m_lstm['mae']:.6f}  RMSE={m_lstm['rmse']:.6f}  "
          f"R2={m_lstm['r2']:.4f}  DA={m_lstm['da']:.3f}")
    print(f"  [{ticker}] Baseline: "
          f"MAE={m_base['mae']:.6f}  RMSE={m_base['rmse']:.6f}  "
          f"R2={m_base['r2']:.4f}")

    return {
        "ticker":       ticker,
        "n_train":      len(train_rv),
        "n_val":        len(val_rv),
        "n_test":       len(X_te),
        "ess_train":    len(train_rv) // MAX_H,
        "lstm_mae":     m_lstm["mae"],
        "lstm_rmse":    m_lstm["rmse"],
        "lstm_r2":      m_lstm["r2"],
        "lstm_da":      m_lstm["da"],
        "base_mae":     m_base["mae"],
        "base_rmse":    m_base["rmse"],
        "base_r2":      m_base["r2"],
        "mae_vs_base":  m_lstm["mae"]  / (m_base["mae"]  + 1e-12),
        "rmse_vs_base": m_lstm["rmse"] / (m_base["rmse"] + 1e-12),
    }


# ─────────────────────────── SUMMARY PLOT ─────────────────────────────────────
def plot_summary(df: pd.DataFrame, out_path: Path):
    """4-panel metrics chart: MAE, RMSE, R², Directional Accuracy."""
    tickers = df["ticker"].tolist()
    x = np.arange(len(tickers))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE: LSTM vs baseline
    ax = axes[0, 0]
    ax.bar(x - w/2, df["lstm_mae"], w, label="LSTM",          color="steelblue", alpha=0.85)
    ax.bar(x + w/2, df["base_mae"], w, label="Baseline (mean)", color="silver",  alpha=0.80)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE")
    ax.set_title("MAE: LSTM vs Naïve Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE: LSTM vs baseline
    ax = axes[0, 1]
    ax.bar(x - w/2, df["lstm_rmse"], w, label="LSTM",          color="steelblue", alpha=0.85)
    ax.bar(x + w/2, df["base_rmse"], w, label="Baseline (mean)", color="silver",  alpha=0.80)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE: LSTM vs Naïve Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[1, 0]
    colors_r2 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["lstm_r2"]]
    ax.bar(x, df["lstm_r2"], color=colors_r2, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("R² (LSTM): green=positive, red=negative")
    ax.grid(True, alpha=0.3)

    # Directional Accuracy
    ax = axes[1, 1]
    colors_da = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in df["lstm_da"]]
    ax.bar(x, df["lstm_da"], color=colors_da, alpha=0.85)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="Random = 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Directional Accuracy: green≥0.5, red<0.5")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"LSTM h=1 — Test Set Metrics ({GLOBAL_TEST_START} onwards)\n"
        f"{len(df)} stocks  |  naïve baseline = mean(train RV)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Summary plot saved: {out_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate LSTM h=1 models on test set")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Subset of tickers to evaluate (default: all trained models)")
    args = ap.parse_args()

    model_files = sorted(MODELS_DIR.glob("*_model.pt"))
    if not model_files:
        print(f"\n  No trained models found in {MODELS_DIR}")
        print("  Run first:  python baselines/train_lstm_h1.py --all\n")
        return

    trained_tickers = [f.stem.replace("_model", "") for f in model_files]
    if args.tickers:
        tickers = [t for t in args.tickers if t in trained_tickers]
        missing = [t for t in args.tickers if t not in trained_tickers]
        if missing:
            print(f"  [WARN] No model for: {missing}")
    else:
        tickers = trained_tickers

    print(f"\n{'='*62}")
    print(f"  LSTM h=1 Evaluation -- {len(tickers)} models")
    print(f"  Test set: {GLOBAL_TEST_START} onwards  (R6)")
    print(f"{'='*62}\n")

    print(f"[Loading] {DATA_DIR} ...")
    close = load_close_prices(DATA_DIR, tickers=tickers)
    rv_h1 = compute_rv_h1(close)
    print(f"  Loaded: {close.shape[0]} dates x {close.shape[1]} tickers\n")

    rows: list[dict] = []
    for ticker in tickers:
        if ticker not in rv_h1.columns:
            print(f"  [{ticker}] Not in price data -- skip")
            continue
        print(f"\n  Evaluating {ticker}...")
        row = evaluate_ticker(ticker, rv_h1[ticker])
        if row:
            rows.append(row)

    if not rows:
        print("\n  No evaluation results.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    print(f"\n{'='*62}")
    print(f"  Results saved: {CSV_OUT}")
    print(f"{'='*62}")

    display_cols = [
        "ticker", "n_test", "ess_train",
        "lstm_mae", "lstm_rmse", "lstm_r2", "lstm_da",
        "base_mae", "base_rmse"
    ]
    avail_cols = [c for c in display_cols if c in df.columns]
    print(df[avail_cols].to_string(index=False))

    plot_summary(df, PLOT_OUT)

    print(f"\n  SUMMARY:")
    print(f"    Stocks evaluated : {len(df)}")
    print(f"    Avg R2           : {df['lstm_r2'].mean():.4f}")
    print(f"    Avg DA           : {df['lstm_da'].mean():.3f}")
    print(f"    Avg MAE          : {df['lstm_mae'].mean():.6f}")
    better = (df["lstm_mae"] < df["base_mae"]).sum()
    print(f"    LSTM beats baseline (MAE): {better}/{len(df)} stocks")
    print(f"\n{'='*62}")
    print(f"  DONE -- Evaluation complete")
    print(f"  Plot: {PLOT_OUT}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
