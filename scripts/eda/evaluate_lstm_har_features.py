"""
Evaluate trained LSTM HAR-Features MIMO h=1,5,10,20 models vs HAR-RV and naive baseline.

Mirrors evaluate_lstm_h20.py but:
  - Loads from models/lstm_har/
  - Uses feat_mu/feat_sig from checkpoint for HAR feature normalization
  - Input reconstruction: build_har_features(rv_h1) -> normalize per feat

Output:
  results/lstm_har_results.csv            -- per-stock, per-horizon metrics
  results/lstm_har_summary_metrics.png    -- 4-panel R2 per horizon (LSTM-HAR/HAR-RV/naive)
  results/lstm_har_vs_har.png             -- LSTM-HAR vs HAR-RV R2 scatter by horizon

Usage:
  python scripts/eda/evaluate_lstm_har_features.py
  python scripts/eda/evaluate_lstm_har_features.py --tickers VCB HPG FPT
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
import yaml

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_rv, compute_log_returns
from baselines.har_rv_baseline import fit_har, predict_har

# ─────────────────────────── CONFIG (must match train_lstm_har_features.py) ────
GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8
LOOKBACK    = 20
HORIZONS    = [1, 5, 10, 20]
MAX_H       = max(HORIZONS)
HIDDEN      = 64
DROPOUT     = 0.2
N_FEATURES  = 3
FEAT_NAMES  = ["rv_d", "rv_w", "rv_m"]

_root        = Path(__file__).parent.parent.parent
CONFIG_PATH  = _root / "config.yaml"
MODELS_DIR   = _root / "models" / "lstm_har"
RESULTS_DIR  = _root / "results"
CSV_OUT      = RESULTS_DIR / "lstm_har_results.csv"
PLOT_METRICS = RESULTS_DIR / "lstm_har_summary_metrics.png"
PLOT_VS_HAR  = RESULTS_DIR / "lstm_har_vs_har.png"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR = _root / _cfg["data"]["prices_dir"]


# ─────────────────────────── MODEL (mirrors train_lstm_har_features.py) ───────
class LSTMModelMIMO(nn.Module):
    """Single LSTM backbone (input_size=3) + one Linear head per horizon."""

    def __init__(self, n_features=N_FEATURES, horizons=HORIZONS,
                 hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.horizons = horizons
        self.lstm  = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        feat   = self.drop(out[:, -1, :])
        return torch.cat([h(feat) for h in self.heads], dim=1)


# ─────────────────────────── RV + HAR FEATURES ────────────────────────────────
def compute_all_rv(close: pd.DataFrame) -> pd.DataFrame:
    log_ret = compute_log_returns(close)
    rv_h1   = log_ret.abs().shift(-1)
    frames  = {"rv_h1": rv_h1}
    for h in [5, 10, 20]:
        frames[f"rv_h{h}"] = compute_rv(close, h=h)
    return pd.concat(frames, axis=1)


def build_har_features(rv_h1: pd.Series) -> pd.DataFrame:
    rv_d = rv_h1
    rv_w = rv_h1.rolling(5,  min_periods=5).mean()
    rv_m = rv_h1.rolling(20, min_periods=20).mean()
    return pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}, index=rv_h1.index)


# ─────────────────────────── SEQUENCES ────────────────────────────────────────
def make_sequences(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """x_arr: (N, 3); y_arr: (N, 4). Returns X (M, lookback, 3), y (M, 4)."""
    X, y = [], []
    for i in range(len(x_arr) - lookback):
        X.append(x_arr[i : i + lookback])
        y.append(y_arr[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────── METRICS ──────────────────────────────────────────
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
def evaluate_ticker(ticker: str, rv_all: pd.DataFrame) -> dict | None:
    model_path = MODELS_DIR / f"{ticker}_model.pt"
    if not model_path.exists():
        print(f"  [{ticker}] Model not found: {model_path}")
        return None

    # load checkpoint
    ckpt       = torch.load(model_path, map_location="cpu", weights_only=False)
    rv_mu      = ckpt["rv_mu"]      # {1: float, 5: float, 10: float, 20: float}
    rv_sig     = ckpt["rv_sig"]
    feat_mu    = ckpt["feat_mu"]    # {"rv_d": float, "rv_w": float, "rv_m": float}
    feat_sig   = ckpt["feat_sig"]
    horizons   = ckpt.get("horizons",   HORIZONS)
    hidden     = int(ckpt.get("hidden",    HIDDEN))
    lookback   = int(ckpt.get("lookback",  LOOKBACK))
    n_features = int(ckpt.get("n_features", N_FEATURES))

    model = LSTMModelMIMO(n_features=n_features, horizons=horizons,
                          hidden=hidden, dropout=DROPOUT)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── reconstruct combined (features + targets) same as training ────────────
    if ("rv_h1", ticker) not in rv_all.columns:
        print(f"  [{ticker}] Missing rv_h1")
        return None

    rv_h1_series = rv_all[("rv_h1", ticker)]
    har_df       = build_har_features(rv_h1_series)

    cols = {h: f"rv_h{h}" for h in horizons}
    ticker_rv = pd.DataFrame({
        h: rv_all[(cols[h], ticker)]
        for h in horizons
        if (cols[h], ticker) in rv_all.columns
    })
    combined = ticker_rv.join(har_df).dropna()

    if len(combined) < lookback + 2:
        print(f"  [{ticker}] Insufficient valid rows ({len(combined)})")
        return None

    # R6 split
    test_ts   = pd.Timestamp(GLOBAL_TEST_START)
    comb_pre  = combined[combined.index < test_ts]
    comb_test = combined[combined.index >= test_ts]

    if len(comb_test) == 0:
        print(f"  [{ticker}] No test data (>= {GLOBAL_TEST_START})")
        return None

    n_tr       = int(len(comb_pre) * TRAIN_VAL_SPLIT_RATIO)
    comb_train = comb_pre.iloc[:n_tr]
    comb_val   = comb_pre.iloc[n_tr:]

    # ── normalization functions (using checkpoint stats) ──────────────────────
    def _norm_x(df_slice: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df_slice), n_features), dtype=np.float32)
        for j, f in enumerate(FEAT_NAMES):
            out[:, j] = (df_slice[f].values - feat_mu[f]) / feat_sig[f]
        return out

    def _norm_y(df_slice: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df_slice), len(horizons)), dtype=np.float32)
        for j, h in enumerate(horizons):
            out[:, j] = (df_slice[h].values - rv_mu[h]) / rv_sig[h]
        return out

    # ── build test sequences (same as training) ────────────────────────────────
    test_ctx = pd.concat([comb_pre.iloc[-lookback:], comb_test])
    X_te, y_te = make_sequences(_norm_x(test_ctx), _norm_y(test_ctx), lookback)

    if len(X_te) == 0:
        print(f"  [{ticker}] No test sequences")
        return None

    # ── LSTM predictions ──────────────────────────────────────────────────────
    Xte_t = torch.tensor(X_te)   # (M, lookback, 3)
    with torch.no_grad():
        preds_n = model(Xte_t).numpy()

    y_true_all = np.zeros((len(y_te),    len(horizons)), dtype=np.float64)
    y_pred_all = np.zeros((len(preds_n), len(horizons)), dtype=np.float64)
    for j, h in enumerate(horizons):
        y_true_all[:, j] = y_te[:, j]    * rv_sig[h] + rv_mu[h]
        y_pred_all[:, j] = np.clip(preds_n[:, j] * rv_sig[h] + rv_mu[h], 0.0, None)

    n_seq      = len(X_te)
    test_index = comb_test.index[:n_seq]

    # ── per-horizon LSTM metrics ──────────────────────────────────────────────
    lstm_m = {}
    for j, h in enumerate(horizons):
        lstm_m[h] = compute_metrics(y_true_all[:, j], y_pred_all[:, j])

    # ── naive baseline ────────────────────────────────────────────────────────
    naive_m = {}
    for j, h in enumerate(horizons):
        y_naive    = np.full(n_seq, float(comb_train[h].mean()))
        naive_m[h] = compute_metrics(y_true_all[:, j], y_naive)

    # ── HAR-RV per horizon: fit on train, predict test ────────────────────────
    har_m = {}
    train_end_ts = comb_train.index[-1]
    for j, h in enumerate(horizons):
        col = f"rv_h{h}"
        if (col, ticker) not in rv_all.columns:
            continue
        rv_series = rv_all[(col, ticker)]
        try:
            coeffs   = fit_har(rv_series, train_end_ts)
            har_pred = predict_har(rv_series, coeffs, test_ts)
            common   = test_index.intersection(har_pred.index)
            if len(common) == 0:
                continue
            true_ser = pd.Series(y_true_all[:, j], index=test_index)
            har_m[h] = compute_metrics(
                true_ser.loc[common].values,
                har_pred.loc[common].values,
            )
        except Exception as e:
            print(f"  [{ticker}] HAR h={h} failed: {e}")

    # ── console summary ───────────────────────────────────────────────────────
    m20 = lstm_m.get(20, {})
    print(f"  [{ticker}] LSTM-HAR h=20: R2={m20.get('r2', float('nan')):7.4f}  "
          f"MAE={m20.get('mae', float('nan')):.5f}  "
          f"DA={m20.get('da', float('nan')):.3f}")
    if 20 in har_m:
        h20 = har_m[20]
        print(f"  [{ticker}] HAR-RV  h=20: R2={h20.get('r2', float('nan')):7.4f}  "
              f"MAE={h20.get('mae', float('nan')):.5f}")

    # ── build CSV row ─────────────────────────────────────────────────────────
    row: dict = {
        "ticker":  ticker,
        "n_train": len(comb_train),
        "n_val":   len(comb_val),
        "n_test":  n_seq,
        "ess_h20": len(comb_train) // MAX_H,
    }
    for j, h in enumerate(horizons):
        for k, v in lstm_m[h].items():
            row[f"lstm_h{h}_{k}"] = v
        for k, v in naive_m[h].items():
            row[f"naive_h{h}_{k}"] = v
        if h in har_m:
            for k, v in har_m[h].items():
                row[f"har_h{h}_{k}"] = v

    return row


# ─────────────────────────── PLOTS ────────────────────────────────────────────
def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return df[col].values.astype(float)
    return np.full(len(df), float("nan"))


def plot_summary_metrics(df: pd.DataFrame, out_path: Path):
    """4-panel: one per horizon showing R2 for LSTM-HAR, HAR-RV, naive mean."""
    tickers = df["ticker"].tolist()
    x = np.arange(len(tickers))
    w = 0.28

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, h in enumerate(HORIZONS):
        ax = axes[idx // 2, idx % 2]

        lstm_r2  = _safe_col(df, f"lstm_h{h}_r2")
        har_r2   = _safe_col(df, f"har_h{h}_r2")
        naive_r2 = _safe_col(df, f"naive_h{h}_r2")

        has_har   = not np.all(np.isnan(har_r2))
        has_naive = not np.all(np.isnan(naive_r2))

        ax.bar(x - w,   lstm_r2,  w, label="LSTM-HAR",  color="steelblue",  alpha=0.85)
        if has_har:
            ax.bar(x,       har_r2,   w, label="HAR-RV",   color="darkorange", alpha=0.85)
        if has_naive:
            ax.bar(x + w,   naive_r2, w, label="Naive mean", color="silver",   alpha=0.80)

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("R2")
        ax.set_title(f"h={h}: R2 -- LSTM-HAR vs HAR-RV vs Naive")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"LSTM HAR-Features MIMO h=1,5,10,20 -- Test Set R2 ({GLOBAL_TEST_START} onwards)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Summary metrics plot saved: {out_path}")


def plot_vs_har(df: pd.DataFrame, out_path: Path):
    """4-panel scatter: LSTM-HAR R2 vs HAR-RV R2 per horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    tickers   = df["ticker"].tolist()

    for idx, h in enumerate(HORIZONS):
        ax       = axes[idx // 2, idx % 2]
        lstm_col = f"lstm_h{h}_r2"
        har_col  = f"har_h{h}_r2"

        if lstm_col not in df.columns or har_col not in df.columns:
            ax.set_title(f"h={h}: no HAR data")
            continue

        xv = df[har_col].values.astype(float)
        yv = df[lstm_col].values.astype(float)
        mask = ~(np.isnan(xv) | np.isnan(yv))
        xv, yv = xv[mask], yv[mask]
        lbls = [t for t, m in zip(tickers, mask) if m]

        ax.scatter(xv, yv, color="steelblue", s=60, zorder=5)
        for xi, yi, lbl in zip(xv, yv, lbls):
            ax.annotate(lbl, (xi, yi), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", color="gray")

        lim_lo = min(np.nanmin(xv), np.nanmin(yv)) - 0.05
        lim_hi = max(np.nanmax(xv), np.nanmax(yv)) + 0.05
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="red", lw=1.0, ls="--", label="y=x (equal)")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.axvline(0, color="black", lw=0.5, ls=":")

        lstm_wins = int(np.sum(yv > xv))
        ax.set_xlabel(f"HAR-RV R2 (h={h})")
        ax.set_ylabel(f"LSTM-HAR R2 (h={h})")
        ax.set_title(f"h={h}: LSTM-HAR vs HAR-RV  ({lstm_wins}/{len(xv)} LSTM-HAR wins)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"LSTM HAR-Features vs HAR-RV -- R2 by Stock and Horizon\n"
        f"Test set: {GLOBAL_TEST_START} onwards  |  above diagonal = LSTM-HAR wins",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  LSTM-HAR vs HAR plot saved: {out_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate LSTM HAR-Features MIMO h=1,5,10,20 vs HAR-RV"
    )
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Subset of tickers (default: all trained models)")
    args = ap.parse_args()

    model_files = sorted(MODELS_DIR.glob("*_model.pt"))
    if not model_files:
        print(f"\n  No trained models found in {MODELS_DIR}")
        print("  Run first:  python baselines/train_lstm_har_features.py --all\n")
        return

    trained = [f.stem.replace("_model", "") for f in model_files]
    if args.tickers:
        tickers = [t for t in args.tickers if t in trained]
        missing = [t for t in args.tickers if t not in trained]
        if missing:
            print(f"  [WARN] No model for: {missing}")
    else:
        tickers = trained

    print(f"\n{'='*65}")
    print(f"  LSTM HAR-Features Evaluation -- {len(tickers)} models")
    print(f"  Horizons: {HORIZONS}  |  Primary: h=20")
    print(f"  Test set: {GLOBAL_TEST_START} onwards  (R6)")
    print(f"  Comparing: LSTM-HAR vs HAR-RV vs Naive mean")
    print(f"{'='*65}\n")

    print(f"[Loading] {DATA_DIR} ...")
    close  = load_close_prices(DATA_DIR, tickers=tickers)
    rv_all = compute_all_rv(close)
    print(f"  Loaded: {close.shape[0]} dates x {close.shape[1]} tickers\n")

    rows: list[dict] = []
    for ticker in tickers:
        print(f"\n  Evaluating {ticker}...")
        row = evaluate_ticker(ticker, rv_all)
        if row:
            rows.append(row)

    if not rows:
        print("\n  No evaluation results.")
        return

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    print(f"\n{'='*65}")
    print(f"  Results saved: {CSV_OUT}")
    print(f"{'='*65}")

    display = ["ticker", "n_test", "ess_h20",
               "lstm_h20_r2", "har_h20_r2", "naive_h20_r2",
               "lstm_h20_mae", "har_h20_mae",
               "lstm_h1_r2", "lstm_h5_r2", "lstm_h10_r2"]
    avail = [c for c in display if c in df.columns]
    print(df[avail].to_string(index=False))

    print(f"\n  h=20 PRIMARY METRIC:")
    if "lstm_h20_r2" in df.columns:
        print(f"    Avg LSTM-HAR R2 : {df['lstm_h20_r2'].mean():.4f}")
    if "har_h20_r2" in df.columns:
        print(f"    Avg HAR-RV  R2  : {df['har_h20_r2'].mean():.4f}")
    if "naive_h20_r2" in df.columns:
        print(f"    Avg Naive   R2  : {df['naive_h20_r2'].mean():.4f}")
    if "lstm_h20_r2" in df.columns and "har_h20_r2" in df.columns:
        wins = int((df["lstm_h20_r2"] > df["har_h20_r2"]).sum())
        print(f"    LSTM-HAR > HAR  : {wins}/{len(df)} stocks")
    if "lstm_h20_mae" in df.columns:
        print(f"    Avg LSTM-HAR MAE: {df['lstm_h20_mae'].mean():.5f}")
    if "lstm_h20_da" in df.columns:
        print(f"    Avg LSTM-HAR DA : {df['lstm_h20_da'].mean():.3f}")

    plot_summary_metrics(df, PLOT_METRICS)
    plot_vs_har(df, PLOT_VS_HAR)

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"  CSV:  {CSV_OUT}")
    print(f"  Plot: {PLOT_METRICS}")
    print(f"  Plot: {PLOT_VS_HAR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
