"""
CONSTRAINTS.md-compliant LSTM training -- MIMO h=1,5,10,20.

R1: MIMO -- HORIZONS=[1,5,10,20], 1 backbone, 4 heads, single forward pass
R2: Per-horizon loss printed every epoch + learning curve PNG per stock
R3: Data split + ESS_h20 printed before each stock's training
R4: stride=1, verbose
R6: Test from 2026-01-01; Train 80% / Val 20% from pre-2026

User requirements:
  - Run directly: python baselines/train_lstm_h20.py --all
  - Real-time terminal output: sys.stdout.reconfigure(line_buffering=True)
  - Learning curve PNG saved automatically after each stock

Usage:
  python baselines/train_lstm_h20.py --all
  python baselines/train_lstm_h20.py --stocks-from results/stocks_ess_h20_over100.csv
  python baselines/train_lstm_h20.py --tickers VCB HPG FPT VNM
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
sys.stdout.reconfigure(line_buffering=True)   # real-time terminal output
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_rv, compute_log_returns
from gnn.build_graph import VN30_TICKERS

# ─────────────────────────── CONFIG ───────────────────────────────────────────
LOOKBACK              = 20
HORIZONS              = [1, 5, 10, 20]   # R1: full MIMO
MAX_H                 = max(HORIZONS)     # 20
STRIDE                = 1                 # R4

GLOBAL_TEST_START     = "2026-01-01"     # R6
TRAIN_VAL_SPLIT_RATIO = 0.8              # R6: 80/20

EPOCHS     = 150
LR         = 1e-3
BATCH_SIZE = 32
PATIENCE   = 25
SEED       = 42
HIDDEN     = 64    # larger than h=1 (32) to support 4 output heads
DROPOUT    = 0.2

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
CURVES_DIR  = Path(__file__).parent.parent / "results" / "lstm_h20_curves"
MODELS_DIR  = Path(__file__).parent.parent / "models" / "lstm_h20"
CSV_OUT     = Path(__file__).parent.parent / "results" / "lstm_h20_results.csv"

CURVES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR = Path(__file__).parent.parent / _cfg["data"]["prices_dir"]

H_LABELS = [f"H{h}" for h in HORIZONS]   # ["H1","H5","H10","H20"]


# ─────────────────────────── RV LABELS ────────────────────────────────────────
def compute_all_rv(close: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RV for all horizons and return aligned DataFrame.

    h=1:        abs(log_ret).shift(-1)  -- rolling(1).std(ddof=1) is NaN
    h=5,10,20:  compute_rv(close, h)    -- standard formula
    """
    log_ret = compute_log_returns(close)
    rv_h1   = log_ret.abs().shift(-1)

    frames = {"rv_h1": rv_h1}
    for h in [5, 10, 20]:
        frames[f"rv_h{h}"] = compute_rv(close, h=h)

    return pd.concat(frames, axis=1)   # MultiIndex columns: (rv_hX, ticker)


# ─────────────────────────── MODEL (R1: MIMO) ─────────────────────────────────
class LSTMModelMIMO(nn.Module):
    """Single LSTM backbone + one Linear head per horizon (R1)."""

    def __init__(self, horizons=HORIZONS, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.horizons = horizons
        self.lstm  = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, LOOKBACK, 1)  ->  (B, n_horizons)"""
        out, _ = self.lstm(x)
        feat   = self.drop(out[:, -1, :])
        return torch.cat([h(feat) for h in self.heads], dim=1)


# ─────────────────────────── SEQUENCES (R4: stride=1) ────────────────────────
def make_sequences(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    stride=1 sequences (R4).
    x_arr: (N,)   -- input feature (rv_h1, normalized)
    y_arr: (N, 4) -- multi-horizon targets (normalized per horizon)
    Returns X (M, lookback), y (M, 4)
    """
    X, y = [], []
    for i in range(len(x_arr) - lookback):
        X.append(x_arr[i : i + lookback])
        y.append(y_arr[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────── DATA SPLIT PRINT (R3) ───────────────────────────
def print_split(
    ticker: str,
    rv_train: pd.DataFrame,
    rv_val:   pd.DataFrame,
    rv_test:  pd.DataFrame,
):
    """R3: print data split with dates, counts, ESS_h20."""
    n_tr = len(rv_train)
    n_va = len(rv_val)
    n_te = len(rv_test)
    ess  = n_tr // MAX_H

    print(f"\n{'='*65}")
    print(f"  DATA SPLIT -- LSTM MIMO | Stock: {ticker}")
    print(f"{'='*65}")
    print(f"  Train: {rv_train.index[0].date()} -> {rv_train.index[-1].date()}"
          f"  ({n_tr:,} samples, ESS_h20={ess})")
    print(f"  Val:   {rv_val.index[0].date()} -> {rv_val.index[-1].date()}"
          f"  ({n_va:,} samples, ESS_h20={n_va//MAX_H})")
    if n_te > 0:
        print(f"  Test:  {rv_test.index[0].date()} -> {rv_test.index[-1].date()}"
              f"  ({n_te:,} samples)")
    else:
        print(f"  Test:  (no data from {GLOBAL_TEST_START})")
    print(f"{'='*65}")
    print(f"  stride={STRIDE} | LOOKBACK={LOOKBACK} | HORIZONS={HORIZONS} | MAX_H={MAX_H}")
    print(f"  ESS_h20 train = {ess} independent obs  (N_train / MAX_H={MAX_H})")
    print(f"{'='*65}\n")


# ─────────────────────────── LEARNING CURVE (R2 + User req.) ─────────────────
def save_learning_curve(
    train_losses: list,
    val_losses:   list,
    val_per_h:    list,
    path: Path,
    ticker: str,
):
    """R2 + User requirement: save PNG after each stock, embedded in training loop."""
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, train_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(ep, val_losses,   label="Val",   color="darkorange", lw=1.5)
    axes[0].set_title(f"{ticker} -- Total MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for i, h in enumerate(HORIZONS):
        axes[1].plot(ep, val_per_h[i], label=f"H={h}", color=colors[i], lw=1.5)
    axes[1].set_title(f"{ticker} -- Val Loss per Horizon")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"LSTM MIMO Learning Curves -- {ticker}  (HORIZONS={HORIZONS})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Curve saved: {path}")


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
def train_stock(ticker: str, rv_all: pd.DataFrame) -> dict | None:
    """Full R1-R6 + User requirements training pipeline for one stock."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── extract per-horizon series for this ticker ────────────────────────────
    cols = {h: f"rv_h{h}" for h in HORIZONS}
    ticker_rv = pd.DataFrame({
        h: rv_all[(cols[h], ticker)] for h in HORIZONS
        if (cols[h], ticker) in rv_all.columns
    }).dropna()

    if len(ticker_rv) < LOOKBACK + MAX_H + 10:
        print(f"  [{ticker}] Skipped: only {len(ticker_rv)} valid rows")
        return None

    # ── R6: global test split ─────────────────────────────────────────────────
    test_ts = pd.Timestamp(GLOBAL_TEST_START)
    rv_pre  = ticker_rv[ticker_rv.index < test_ts]
    rv_test = ticker_rv[ticker_rv.index >= test_ts]

    if len(rv_pre) < LOOKBACK + 10:
        print(f"  [{ticker}] Skipped: insufficient pre-test rows ({len(rv_pre)})")
        return None

    # R6: 80/20 train/val
    n_tr     = int(len(rv_pre) * TRAIN_VAL_SPLIT_RATIO)
    rv_train = rv_pre.iloc[:n_tr]
    rv_val   = rv_pre.iloc[n_tr:]

    print_split(ticker, rv_train, rv_val, rv_test)  # R3

    # ── per-horizon normalization on train stats ──────────────────────────────
    rv_mu:  dict[int, float] = {}
    rv_sig: dict[int, float] = {}
    for h in HORIZONS:
        rv_mu[h]  = float(rv_train[h].mean())
        rv_sig[h] = float(rv_train[h].std()) + 1e-8

    def _norm_x(arr):
        return (arr - rv_mu[1]) / rv_sig[1]

    def _norm_y(df):
        out = np.zeros((len(df), len(HORIZONS)), dtype=np.float32)
        for j, h in enumerate(HORIZONS):
            out[:, j] = (df[h].values - rv_mu[h]) / rv_sig[h]
        return out

    # ── stride=1 sequences (R4) ───────────────────────────────────────────────
    # val context: prepend last LOOKBACK train rows
    val_ctx  = pd.concat([rv_train.iloc[-LOOKBACK:], rv_val])
    test_ctx = pd.concat([rv_pre.iloc[-LOOKBACK:],   rv_test])

    X_tr, y_tr = make_sequences(_norm_x(rv_train[1].values), _norm_y(rv_train), LOOKBACK)
    X_va, y_va = make_sequences(_norm_x(val_ctx[1].values),  _norm_y(val_ctx),  LOOKBACK)

    if len(X_tr) < BATCH_SIZE:
        print(f"  [{ticker}] Skipped: only {len(X_tr)} train sequences")
        return None

    Xt = torch.tensor(X_tr).unsqueeze(-1)
    yt = torch.tensor(y_tr)
    Xv = torch.tensor(X_va).unsqueeze(-1)
    yv = torch.tensor(y_va)

    # ── model + optimizer ─────────────────────────────────────────────────────
    model = LSTMModelMIMO(horizons=HORIZONS, hidden=HIDDEN, dropout=DROPOUT)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    # ── training loop (R2 + R6 + User req.) ──────────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_per_h:    list[list]  = [[] for _ in HORIZONS]
    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0
    n_seqs = len(Xt)

    print(f"  Training {ticker}: {n_seqs} train seqs | {len(Xv)} val seqs | "
          f"EPOCHS={EPOCHS} PATIENCE={PATIENCE} BATCH={BATCH_SIZE} HIDDEN={HIDDEN}")

    for ep in range(1, EPOCHS + 1):
        # train
        model.train()
        perm = np.random.permutation(n_seqs)
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

        # R2 + R6: per-epoch console print
        h_str = " ".join(f"H{h}={per_h[j]:.4f}" for j, h in enumerate(HORIZONS))
        print(f"  Epoch {ep:3d}/{EPOCHS} | "
              f"Train: {tr_l:.4f} | "
              f"Val: {va_l:.4f} [{h_str}] | "
              f"LR: {LR:.2e}")

        # early stopping
        if va_l < best_val:
            best_val, best_ep = va_l, ep
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at ep {ep}  (best val={best_val:.4f} @ ep {best_ep})")
                break

    model.load_state_dict(best_sd)

    # R2 + User req: save learning curve PNG automatically
    save_learning_curve(
        train_losses, val_losses, val_per_h,
        CURVES_DIR / f"{ticker}_curve.png", ticker
    )

    # save model checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "horizons":   HORIZONS,
        "lookback":   LOOKBACK,
        "hidden":     HIDDEN,
        "rv_mu":      rv_mu,
        "rv_sig":     rv_sig,
    }, MODELS_DIR / f"{ticker}_model.pt")
    print(f"  Model saved: {MODELS_DIR / f'{ticker}_model.pt'}")

    # ── test evaluation ───────────────────────────────────────────────────────
    row: dict = {
        "ticker":       ticker,
        "n_train":      len(rv_train),
        "n_val":        len(rv_val),
        "n_train_seqs": len(X_tr),
        "n_val_seqs":   len(X_va),
        "ess_h20":      len(rv_train) // MAX_H,
        "best_val_mse": best_val,
        "best_epoch":   best_ep,
    }

    if len(rv_test) > 0:
        X_te, y_te = make_sequences(
            _norm_x(test_ctx[1].values),
            _norm_y(test_ctx),
            LOOKBACK,
        )
        if len(X_te) > 0:
            Xte_t = torch.tensor(X_te).unsqueeze(-1)
            model.eval()
            with torch.no_grad():
                preds_n = model(Xte_t).numpy()

            # denormalize per horizon
            y_true_all = np.zeros_like(y_te)
            y_pred_all = np.zeros_like(preds_n)
            for j, h in enumerate(HORIZONS):
                y_true_all[:, j] = y_te[:, j]     * rv_sig[h] + rv_mu[h]
                y_pred_all[:, j] = preds_n[:, j]  * rv_sig[h] + rv_mu[h]
                y_pred_all[:, j] = np.clip(y_pred_all[:, j], 0.0, None)

            row["n_test_seqs"] = len(X_te)
            for j, h in enumerate(HORIZONS):
                m = compute_metrics(y_true_all[:, j], y_pred_all[:, j])
                row[f"h{h}_mae"]  = m["mae"]
                row[f"h{h}_rmse"] = m["rmse"]
                row[f"h{h}_r2"]   = m["r2"]
                row[f"h{h}_da"]   = m["da"]

            # print test results (focus on h=20 as primary metric)
            parts = " | ".join(
                f"H{h}: R2={row[f'h{h}_r2']:.4f} MAE={row[f'h{h}_mae']:.5f}"
                for h in HORIZONS
            )
            print(f"  Test [{ticker}]: {parts}")
        else:
            print(f"  [{ticker}] No valid test sequences")
    else:
        print(f"  [{ticker}] No test data from {GLOBAL_TEST_START}")

    return row


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train LSTM MIMO h=1,5,10,20 per stock")
    ap.add_argument("--stocks-from", default=None,
                    help="CSV from filter_stocks_by_ess_h20.py (ticker column)")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Explicit tickers, e.g. --tickers VCB HPG FPT")
    ap.add_argument("--all", action="store_true",
                    help="Train all 30 VN30 stocks")
    args = ap.parse_args()

    if args.tickers:
        tickers = args.tickers
    elif args.all:
        tickers = VN30_TICKERS
    elif args.stocks_from:
        tickers = pd.read_csv(args.stocks_from)["ticker"].tolist()
    else:
        ess_csv = Path(__file__).parent.parent / "results" / "stocks_ess_h20_over100.csv"
        if ess_csv.exists():
            tickers = pd.read_csv(ess_csv)["ticker"].tolist()
        else:
            tickers = []

    if not tickers:
        print("\n  No tickers. Run filter_stocks_by_ess_h20.py first, or use --all.\n")
        return

    print(f"\n{'='*65}")
    print(f"  LSTM MIMO Training -- {len(tickers)} stocks")
    print(f"  R1: HORIZONS={HORIZONS} | LOOKBACK={LOOKBACK} | STRIDE={STRIDE}")
    print(f"  R6: test>={GLOBAL_TEST_START} | train/val="
          f"{int(TRAIN_VAL_SPLIT_RATIO*100)}/{100-int(TRAIN_VAL_SPLIT_RATIO*100)}")
    print(f"  EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | LR={LR} | BATCH={BATCH_SIZE} | HIDDEN={HIDDEN}")
    print(f"{'='*65}\n")

    print(f"[Loading] {DATA_DIR} ...")
    close  = load_close_prices(DATA_DIR, tickers=tickers)
    rv_all = compute_all_rv(close)
    print(f"  Loaded: {close.shape[0]} dates x {close.shape[1]} tickers")
    print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}\n")

    all_rows: list[dict] = []
    for ticker in tickers:
        # check all horizons available
        missing = [h for h in HORIZONS if (f"rv_h{h}", ticker) not in rv_all.columns]
        if missing:
            print(f"  [{ticker}] Missing RV for horizons {missing} -- skip")
            continue

        print(f"\n{'-'*65}")
        print(f"  Stock: {ticker}")
        print(f"{'-'*65}")

        row = train_stock(ticker, rv_all)
        if row:
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n{'='*65}")
        print(f"  Results -> {CSV_OUT}")
        print(f"{'='*65}")
        # summary: show h=20 metrics
        cols = ["ticker", "ess_h20", "best_epoch",
                "h1_r2", "h5_r2", "h10_r2", "h20_r2",
                "h20_mae", "h20_rmse"]
        avail = [c for c in cols if c in df.columns]
        print(df[avail].to_string(index=False))
    else:
        print("\n  No results to save.")

    print(f"\n{'='*65}")
    print(f"  DONE  |  Models: {MODELS_DIR}  |  Curves: {CURVES_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
