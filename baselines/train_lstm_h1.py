"""
CONSTRAINTS.md-compliant LSTM training — h=1 single-horizon MIMO.

R1: MIMO architecture (HORIZONS=[1], single head, configurable)
R2: Per-horizon loss printed every epoch + learning curve PNG saved
R3: Data split with ESS printed before each stock's training
R4: stride=1, verbose data loading
R6: Test from 2026-01-01; Train 80% / Val 20% from pre-2026 data

Usage:
  python baselines/train_lstm_h1.py --all
  python baselines/train_lstm_h1.py --tickers VCB HPG FPT
  python baselines/train_lstm_h1.py --stocks-from results/stocks_ess_h1_over_3500.csv
"""
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)   # real-time output in terminal
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns
from gnn.build_graph import VN30_TICKERS

# ─────────────────────────── CONFIG ───────────────────────────────────────────
LOOKBACK              = 20
HORIZONS              = [1]       # R1: configurable, MIMO single head for h=1
MAX_H                 = max(HORIZONS)
STRIDE                = 1         # R4: stride=1

GLOBAL_TEST_START     = "2026-01-01"   # R6
TRAIN_VAL_SPLIT_RATIO = 0.8            # R6: 80/20

EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 32
PATIENCE   = 20
SEED       = 42
HIDDEN     = 32
DROPOUT    = 0.1

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
CURVES_DIR  = Path(__file__).parent.parent / "results" / "lstm_h1_curves"
MODELS_DIR  = Path(__file__).parent.parent / "models" / "lstm_h1"
CSV_OUT     = Path(__file__).parent.parent / "results" / "lstm_h1_results.csv"

CURVES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR = Path(__file__).parent.parent / _cfg["data"]["prices_dir"]


# ─────────────────────────── RV for h=1 ───────────────────────────────────────
def compute_rv_h1(close: pd.DataFrame) -> pd.DataFrame:
    """
    1-day RV proxy: absolute log-return shifted by -1.
    RV_t = |log(P_{t+1}/P_t)|

    Note: rolling(1).std(ddof=1) is NaN everywhere (N-1=0).
    Absolute return is the standard h=1 RV proxy in the literature.
    """
    log_ret = compute_log_returns(close)
    return log_ret.abs().shift(-1)


# ─────────────────────────── MODEL (R1: MIMO) ─────────────────────────────────
class LSTMModelMIMO(nn.Module):
    """Single LSTM backbone + one Linear head per horizon (R1: MIMO)."""

    def __init__(self, horizons=HORIZONS, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.horizons = horizons
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        # R1: one output head per horizon — single model, no recursive strategy
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, LOOKBACK, 1)  →  (B, len(horizons))"""
        out, _ = self.lstm(x)
        feat = self.drop(out[:, -1, :])          # last timestep hidden state
        return torch.cat([h(feat) for h in self.heads], dim=1)


# ─────────────────────────── SEQUENCES (R4: stride=1) ────────────────────────
def make_sequences(rv_arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding window with stride=1 (R4).
    X[i] = rv_arr[i : i+lookback]   shape: (lookback,)
    y[i] = [rv_arr[i+lookback]]     shape: (1,)   — MIMO-compatible target
    """
    X, y = [], []
    for i in range(len(rv_arr) - lookback):
        X.append(rv_arr[i : i + lookback])
        y.append([rv_arr[i + lookback]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────── DATA SPLIT PRINT (R3) ───────────────────────────
def print_split(ticker: str, train_rv: pd.Series, val_rv: pd.Series, test_rv: pd.Series):
    """R3: print data split with dates, sample counts, and ESS before training."""
    n_tr, n_va, n_te = len(train_rv), len(val_rv), len(test_rv)
    ess_tr = n_tr // MAX_H
    ess_va = n_va // MAX_H
    ess_te = n_te // MAX_H if n_te > 0 else 0

    print(f"\n{'='*62}")
    print(f"  DATA SPLIT -- LSTM h=1  |  Stock: {ticker}")
    print(f"{'='*62}")
    print(f"  Train: {train_rv.index[0].date()} -> {train_rv.index[-1].date()}"
          f"  ({n_tr:,} days, ESS={ess_tr})")
    print(f"  Val:   {val_rv.index[0].date()} -> {val_rv.index[-1].date()}"
          f"  ({n_va:,} days, ESS={ess_va})")
    if n_te > 0:
        print(f"  Test:  {test_rv.index[0].date()} -> {test_rv.index[-1].date()}"
              f"  ({n_te:,} days, ESS={ess_te})")
    else:
        print(f"  Test:  (no data from {GLOBAL_TEST_START})")
    print(f"{'='*62}")
    print(f"  Stock {ticker}: stride={STRIDE} | LOOKBACK={LOOKBACK} | HORIZONS={HORIZONS}")
    print(f"  ESS train = {ess_tr} independent obs  (N_train / MAX_H={MAX_H})")
    print(f"{'='*62}\n")


# ─────────────────────────── TRAINING CURVE (R2 + R6) ────────────────────────
def save_learning_curve(
    train_losses: list, val_losses: list, val_per_h: list,
    horizons: list, path: Path, ticker: str
):
    """R2+R6: Embedded plotting — Total loss + per-horizon val breakdown."""
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, train_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(ep, val_losses,   label="Val",   color="darkorange", lw=1.5)
    axes[0].set_title(f"{ticker} — Total MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, h in enumerate(horizons):
        axes[1].plot(ep, val_per_h[i], label=f"H={h}", lw=1.5)
    axes[1].set_title(f"{ticker} — Val Loss per Horizon")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"LSTM h=1 Learning Curves — {ticker}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved: {path}")


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


# ─────────────────────────── TRAIN ONE STOCK ──────────────────────────────────
def train_stock(ticker: str, rv_series: pd.Series) -> dict | None:
    """Full R1-R6-compliant training pipeline for one stock."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── R6: global split ──────────────────────────────────────────────────────
    test_ts = pd.Timestamp(GLOBAL_TEST_START)
    pre     = rv_series[rv_series.index < test_ts].dropna()
    test_rv = rv_series[rv_series.index >= test_ts].dropna()

    if len(pre) < LOOKBACK + 20:
        print(f"  [{ticker}] Skipped: only {len(pre)} pre-test points (need >={LOOKBACK+20})")
        return None

    # R6: 80/20 train/val from pre-2026 data
    n_tr     = int(len(pre) * TRAIN_VAL_SPLIT_RATIO)
    train_rv = pre.iloc[:n_tr]
    val_rv   = pre.iloc[n_tr:]

    print_split(ticker, train_rv, val_rv, test_rv)  # R3

    # ── Normalize on train stats ──────────────────────────────────────────────
    rv_mu  = float(train_rv.mean())
    rv_sig = float(train_rv.std()) + 1e-8

    train_n = (train_rv.values - rv_mu) / rv_sig
    val_n   = (val_rv.values   - rv_mu) / rv_sig

    # ── R4: stride=1 sequences ────────────────────────────────────────────────
    X_tr, y_tr = make_sequences(train_n, LOOKBACK)
    # val context: prepend last LOOKBACK train values so first val seq has full context
    val_ctx    = np.concatenate([train_n[-LOOKBACK:], val_n])
    X_va, y_va = make_sequences(val_ctx, LOOKBACK)

    if len(X_tr) < BATCH_SIZE:
        print(f"  [{ticker}] Skipped: only {len(X_tr)} train sequences")
        return None

    Xt = torch.tensor(X_tr).unsqueeze(-1)   # (N_tr, LOOKBACK, 1)
    yt = torch.tensor(y_tr)                  # (N_tr, 1)
    Xv = torch.tensor(X_va).unsqueeze(-1)
    yv = torch.tensor(y_va)

    # ── Model, optimizer ─────────────────────────────────────────────────────
    model = LSTMModelMIMO(horizons=HORIZONS, hidden=HIDDEN, dropout=DROPOUT)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    # ── Training loop (R2 + R6) ───────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_per_h:    list[list]  = [[] for _ in HORIZONS]

    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0
    n_seqs = len(Xt)

    print(f"  Training {ticker}: {n_seqs} train seqs | {len(Xv)} val seqs | "
          f"EPOCHS={EPOCHS} PATIENCE={PATIENCE} BATCH={BATCH_SIZE}")

    for ep in range(1, EPOCHS + 1):
        # train
        model.train()
        perm     = np.random.permutation(n_seqs)
        ep_loss, n_b = 0.0, 0
        for s in range(0, n_seqs, BATCH_SIZE):
            bi  = perm[s : s + BATCH_SIZE]
            xb, yb = Xt[bi], yt[bi]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b     += 1
        tr_l = ep_loss / n_b

        # validate
        model.eval()
        with torch.no_grad():
            vp    = model(Xv)
            va_l  = crit(vp, yv).item()
            per_h = [crit(vp[:, i:i+1], yv[:, i:i+1]).item() for i in range(len(HORIZONS))]

        train_losses.append(tr_l)
        val_losses.append(va_l)
        for i, hl in enumerate(per_h):
            val_per_h[i].append(hl)

        # R6: console print — train per-horizon = tr_l when HORIZONS=[1]
        tr_h_str = " ".join(f"H{h}={tr_l:.4f}" for h in HORIZONS)
        va_h_str = " ".join(f"H{h}={per_h[j]:.4f}" for j, h in enumerate(HORIZONS))
        print(f"  Epoch {ep:3d}/{EPOCHS} | "
              f"Train: {tr_l:.4f} [{tr_h_str}] | "
              f"Val: {va_l:.4f} [{va_h_str}] | "
              f"LR: {LR:.2e}")

        # early stopping
        if va_l < best_val:
            best_val, best_ep = va_l, ep
            best_sd   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at ep {ep}  (best val={best_val:.4f} @ ep {best_ep})")
                break

    model.load_state_dict(best_sd)

    # R2+R6: save learning curve PNG
    save_learning_curve(
        train_losses, val_losses, val_per_h, HORIZONS,
        CURVES_DIR / f"{ticker}_curve.png", ticker
    )

    # save model checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "rv_mu":  rv_mu,
        "rv_sig": rv_sig,
        "horizons": HORIZONS,
        "lookback": LOOKBACK,
        "hidden":   HIDDEN,
    }, MODELS_DIR / f"{ticker}_model.pt")
    print(f"  Model saved: {MODELS_DIR / f'{ticker}_model.pt'}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    row: dict = {
        "ticker":       ticker,
        "n_train_seqs": len(X_tr),
        "n_val_seqs":   len(X_va),
        "best_val_mse": best_val,
        "best_epoch":   best_ep,
        "ess_train":    len(train_rv) // MAX_H,
    }

    if len(test_rv) > 0:
        pre_n    = (pre.values - rv_mu) / rv_sig       # full pre-test normalized
        test_n   = (test_rv.values - rv_mu) / rv_sig
        test_ctx = np.concatenate([pre_n[-LOOKBACK:], test_n])
        X_te, y_te = make_sequences(test_ctx, LOOKBACK)

        if len(X_te) > 0:
            Xte_t = torch.tensor(X_te).unsqueeze(-1)
            model.eval()
            with torch.no_grad():
                preds_n = model(Xte_t).numpy()

            y_true = y_te[:, 0] * rv_sig + rv_mu
            y_pred = np.clip(preds_n[:, 0] * rv_sig + rv_mu, 0.0, None)
            m = compute_metrics(y_true, y_pred)
            row.update({
                "n_test_seqs": len(X_te),
                "h1_mae":  m["mae"],
                "h1_rmse": m["rmse"],
                "h1_r2":   m["r2"],
                "h1_da":   m["da"],
            })
            print(f"  Test [{ticker} h=1]: "
                  f"MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  "
                  f"R2={m['r2']:.4f}  DA={m['da']:.3f}")
        else:
            print(f"  [{ticker}] No valid test sequences")
    else:
        print(f"  [{ticker}] No test data (2026-01-01 onwards)")

    return row


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train LSTM h=1 per stock (CONSTRAINTS.md R1-R6)")
    ap.add_argument("--stocks-from", default=None,
                    help="CSV from filter_stocks_by_ess.py (ticker column)")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Explicit ticker list, e.g. --tickers VCB HPG FPT")
    ap.add_argument("--all", action="store_true",
                    help="Train all 30 VN30 stocks (ignores ESS filter)")
    args = ap.parse_args()

    # determine ticker list
    if args.tickers:
        tickers = args.tickers
    elif args.all:
        tickers = VN30_TICKERS
    elif args.stocks_from:
        tickers = pd.read_csv(args.stocks_from)["ticker"].tolist()
    else:
        ess_csv = Path(__file__).parent.parent / "results" / "stocks_ess_h1_over_3500.csv"
        if ess_csv.exists():
            tickers = pd.read_csv(ess_csv)["ticker"].tolist()
        else:
            tickers = []

    if not tickers:
        print("\n  No qualifying stocks  (ESS > 3500 yields empty set — expected diagnostic).")
        print("  Use --all for all VN30 stocks, or --tickers VCB HPG FPT for a subset.\n")
        return

    print(f"\n{'='*62}")
    print(f"  LSTM h=1 Training — {len(tickers)} stocks")
    print(f"  R1: HORIZONS={HORIZONS} | LOOKBACK={LOOKBACK} | STRIDE={STRIDE}")
    print(f"  R6: test>={GLOBAL_TEST_START} | train/val="
          f"{int(TRAIN_VAL_SPLIT_RATIO*100)}/{100-int(TRAIN_VAL_SPLIT_RATIO*100)}")
    print(f"  EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | LR={LR} | BATCH={BATCH_SIZE}")
    print(f"{'='*62}\n")

    print(f"[Loading] {DATA_DIR} ...")
    close = load_close_prices(DATA_DIR, tickers=tickers)
    rv_h1 = compute_rv_h1(close)
    print(f"  Loaded: {close.shape[0]} dates x {close.shape[1]} tickers")
    print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}\n")

    all_rows: list[dict] = []

    for ticker in tickers:
        if ticker not in rv_h1.columns:
            print(f"  [{ticker}] Not in data — skip")
            continue

        print(f"\n{'-'*62}")
        print(f"  Stock: {ticker}")
        print(f"{'-'*62}")

        row = train_stock(ticker, rv_h1[ticker])
        if row:
            all_rows.append(row)

    # save aggregated results
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n{'='*62}")
        print(f"  Results -> {CSV_OUT}")
        print(f"{'='*62}")
        print(df.to_string(index=False))
    else:
        print("\n  No training results to save.")

    print(f"\n{'='*62}")
    print(f"  DONE  |  Models: {MODELS_DIR}  |  Curves: {CURVES_DIR}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
