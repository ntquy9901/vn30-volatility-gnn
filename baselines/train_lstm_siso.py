"""
CONSTRAINTS.md R1 EXCEPTION: LSTM SISO ablation study.

Tests MIMO gradient interference hypothesis:
  If SISO h=H >> MIMO h=H  -> interference is the cause
  If SISO h=H ~= MIMO h=H  -> ESS bottleneck confirmed

Controlled ablation vs train_lstm_har_features.py (MIMO):
  Same:      HAR features [rv_d, rv_w, rv_m], hyperparameters, data split
  Different: 1 output head (horizon H only), loss = MSE on H only

Usage:
  python baselines/train_lstm_siso.py --horizon 5 --all
  python baselines/train_lstm_siso.py --horizon 1 --all
  python baselines/train_lstm_siso.py --horizon 10 --all
  python baselines/train_lstm_siso.py --horizon 5 --tickers VCB HPG
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
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_rv, compute_log_returns
from gnn.build_graph import VN30_TICKERS
from baselines.har_rv_baseline import fit_har, predict_har

# ─────────────────────────── STATIC CONFIG ────────────────────────────────────
N_FEATURES            = 3
FEAT_NAMES            = ["rv_d", "rv_w", "rv_m"]
LOOKBACK              = 20
STRIDE                = 1
GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8
EPOCHS     = 150
LR         = 1e-3
BATCH_SIZE = 32
PATIENCE   = 80
SEED       = 42
HIDDEN     = 64
DROPOUT    = 0.2

# Horizon-dependent globals — set in main() after argparse
HORIZONS:   list = []
MAX_H:      int  = 0
CURVES_DIR: Path = Path()
MODELS_DIR: Path = Path()
CSV_OUT:    Path = Path()

_root       = Path(__file__).parent.parent
CONFIG_PATH = _root / "config.yaml"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR = _root / _cfg["data"]["prices_dir"]


# ─────────────────────────── RV LABELS ────────────────────────────────────────
def compute_all_rv(close: pd.DataFrame) -> pd.DataFrame:
    log_ret = compute_log_returns(close)
    rv_h1   = log_ret.abs().shift(-1)
    frames  = {"rv_h1": rv_h1}
    for h in [5, 10, 20]:
        frames[f"rv_h{h}"] = compute_rv(close, h=h)
    return pd.concat(frames, axis=1)


# ─────────────────────────── HAR FEATURES ─────────────────────────────────────
def build_har_features(rv_h1: pd.Series) -> pd.DataFrame:
    rv_d = rv_h1
    rv_w = rv_h1.rolling(5,  min_periods=5).mean()
    rv_m = rv_h1.rolling(20, min_periods=20).mean()
    return pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}, index=rv_h1.index)


# ─────────────────────────── MODEL (R1 exception: 1 head) ─────────────────────
class LSTMModelSISO(nn.Module):
    """LSTM backbone + 1 output head. horizons=[H] with H in {1, 5, 10}."""

    def __init__(self, n_features=N_FEATURES, horizons=None,
                 hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        if horizons is None:
            horizons = HORIZONS
        self.horizons = horizons
        self.lstm  = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        feat   = self.drop(out[:, -1, :])
        return torch.cat([h(feat) for h in self.heads], dim=1)


# ─────────────────────────── SEQUENCES ────────────────────────────────────────
def make_sequences(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(x_arr) - lookback):
        X.append(x_arr[i : i + lookback])
        y.append(y_arr[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────── DATA SPLIT PRINT (R3) ───────────────────────────
def print_split(ticker: str, rv_train, rv_val, rv_test, h: int):
    n_tr = len(rv_train)
    n_va = len(rv_val)
    n_te = len(rv_test)
    ess  = n_tr // h
    print(f"\n{'='*65}")
    print(f"  DATA SPLIT -- LSTM SISO h={h} | Stock: {ticker}")
    print(f"{'='*65}")
    print(f"  Train: {rv_train.index[0].date()} -> {rv_train.index[-1].date()}"
          f"  ({n_tr:,} samples, ESS_h{h}={ess})")
    print(f"  Val:   {rv_val.index[0].date()} -> {rv_val.index[-1].date()}"
          f"  ({n_va:,} samples)")
    if n_te > 0:
        print(f"  Test:  {rv_test.index[0].date()} -> {rv_test.index[-1].date()}"
              f"  ({n_te:,} samples)")
    else:
        print(f"  Test:  (no data from {GLOBAL_TEST_START})")
    print(f"{'='*65}")
    print(f"  stride={STRIDE} | LOOKBACK={LOOKBACK} | N_FEATURES={N_FEATURES}"
          f" | HORIZONS={HORIZONS}")
    print(f"  ESS_h{h} train = {ess} independent obs  (N_train / H={h})")
    print(f"{'='*65}\n")


# ─────────────────────────── LEARNING CURVE ───────────────────────────────────
def save_learning_curve(
    train_losses: list,
    val_losses:   list,
    val_h_losses: list,
    path: Path,
    ticker: str,
    h: int,
):
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ep, train_losses, label="Train", color="steelblue")
    ax.plot(ep, val_losses,   label="Val",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{ticker} -- Total Loss (SISO h={h})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ep, val_h_losses, label=f"Val H={h}", color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE")
    ax.set_title(f"{ticker} -- Val Loss H={h}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"LSTM SISO h={h} Curves -- {ticker}",
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    H = HORIZONS[0]

    if ("rv_h1", ticker) not in rv_all.columns:
        print(f"  [{ticker}] Missing rv_h1 -- skip")
        return None

    rv_h1_series = rv_all[("rv_h1", ticker)]
    har_df       = build_har_features(rv_h1_series)

    rv_col = f"rv_h{H}"
    if (rv_col, ticker) not in rv_all.columns:
        print(f"  [{ticker}] Missing {rv_col} -- skip")
        return None

    ticker_rv = pd.DataFrame({H: rv_all[(rv_col, ticker)]})
    combined  = ticker_rv.join(har_df).dropna()

    min_rows = LOOKBACK + H + 10
    if len(combined) < min_rows:
        print(f"  [{ticker}] Skipped: only {len(combined)} valid rows (need {min_rows})")
        return None

    test_ts   = pd.Timestamp(GLOBAL_TEST_START)
    comb_pre  = combined[combined.index < test_ts]
    comb_test = combined[combined.index >= test_ts]

    if len(comb_pre) < LOOKBACK + 10:
        print(f"  [{ticker}] Skipped: insufficient pre-test rows ({len(comb_pre)})")
        return None

    n_tr       = int(len(comb_pre) * TRAIN_VAL_SPLIT_RATIO)
    comb_train = comb_pre.iloc[:n_tr]
    comb_val   = comb_pre.iloc[n_tr:]

    print_split(ticker, comb_train, comb_val, comb_test, H)

    # per-feature normalization on train stats
    feat_mu:  dict[str, float] = {}
    feat_sig: dict[str, float] = {}
    for feat in FEAT_NAMES:
        feat_mu[feat]  = float(comb_train[feat].mean())
        feat_sig[feat] = float(comb_train[feat].std()) + 1e-8

    def _norm_x(df_slice: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df_slice), N_FEATURES), dtype=np.float32)
        for j, f in enumerate(FEAT_NAMES):
            out[:, j] = (df_slice[f].values - feat_mu[f]) / feat_sig[f]
        return out

    rv_mu_h  = float(comb_train[H].mean())
    rv_sig_h = float(comb_train[H].std()) + 1e-8

    def _norm_y(df_slice: pd.DataFrame) -> np.ndarray:
        return ((df_slice[H].values - rv_mu_h) / rv_sig_h
                ).reshape(-1, 1).astype(np.float32)

    val_ctx  = pd.concat([comb_train.iloc[-LOOKBACK:], comb_val])
    test_ctx = pd.concat([comb_pre.iloc[-LOOKBACK:],   comb_test])

    X_tr, y_tr = make_sequences(_norm_x(comb_train), _norm_y(comb_train), LOOKBACK)
    X_va, y_va = make_sequences(_norm_x(val_ctx),    _norm_y(val_ctx),    LOOKBACK)

    if len(X_tr) < BATCH_SIZE:
        print(f"  [{ticker}] Skipped: only {len(X_tr)} train sequences")
        return None

    Xt = torch.tensor(X_tr)
    yt = torch.tensor(y_tr)
    Xv = torch.tensor(X_va)
    yv = torch.tensor(y_va)

    model = LSTMModelSISO(n_features=N_FEATURES, horizons=HORIZONS,
                          hidden=HIDDEN, dropout=DROPOUT)
    opt  = optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_h_losses: list[float] = []
    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0
    n_seqs = len(Xt)

    print(f"  Training {ticker}: {n_seqs} train seqs | {len(Xv)} val seqs | "
          f"SISO h={H} | EPOCHS={EPOCHS} PATIENCE={PATIENCE}"
          f" HIDDEN={HIDDEN}")

    for ep in range(1, EPOCHS + 1):
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

        model.eval()
        with torch.no_grad():
            vp   = model(Xv)
            va_l = crit(vp, yv).item()

        train_losses.append(tr_l)
        val_losses.append(va_l)
        val_h_losses.append(va_l)

        print(f"  Epoch {ep:3d}/{EPOCHS} | "
              f"Train: {tr_l:.4f} | "
              f"Val: {va_l:.4f} [H{H}={va_l:.4f}] | "
              f"LR: {LR:.2e}")

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

    save_learning_curve(
        train_losses, val_losses, val_h_losses,
        CURVES_DIR / f"{ticker}_curve.png", ticker, H
    )

    torch.save({
        "state_dict": model.state_dict(),
        "horizons":   HORIZONS,
        "lookback":   LOOKBACK,
        "hidden":     HIDDEN,
        "n_features": N_FEATURES,
        "rv_mu":      {H: rv_mu_h},
        "rv_sig":     {H: rv_sig_h},
        "feat_mu":    feat_mu,
        "feat_sig":   feat_sig,
    }, MODELS_DIR / f"{ticker}_model.pt")
    print(f"  Model saved: {MODELS_DIR / f'{ticker}_model.pt'}")

    row: dict = {
        "ticker":   ticker,
        "n_train":  len(comb_train),
        "n_val":    len(comb_val),
        f"ess_h{H}": len(comb_train) // H,
        "best_epoch": best_ep,
        "best_val_mse": best_val,
    }

    if len(comb_test) > 0:
        X_te, y_te = make_sequences(_norm_x(test_ctx), _norm_y(test_ctx), LOOKBACK)
        if len(X_te) > 0:
            Xte_t = torch.tensor(X_te)
            model.eval()
            with torch.no_grad():
                preds_n = model(Xte_t).numpy()

            y_true = y_te[:, 0] * rv_sig_h + rv_mu_h
            y_pred = np.clip(preds_n[:, 0] * rv_sig_h + rv_mu_h, 0.0, None)

            m = compute_metrics(y_true, y_pred)
            for k, v in m.items():
                row[f"lstm_h{H}_{k}"] = v

            n_seq      = len(X_te)
            test_index = comb_test.index[:n_seq]

            # HAR-RV comparison inline
            rv_series = rv_all[(f"rv_h{H}", ticker)]
            try:
                train_end_ts = comb_train.index[-1]
                coeffs   = fit_har(rv_series, train_end_ts)
                har_pred = predict_har(rv_series, coeffs, test_ts)
                common   = test_index.intersection(har_pred.index)
                if len(common) > 0:
                    true_ser = pd.Series(y_true, index=test_index)
                    har_m = compute_metrics(
                        true_ser.loc[common].values,
                        har_pred.loc[common].values,
                    )
                    for k, v in har_m.items():
                        row[f"har_h{H}_{k}"] = v
            except Exception as e:
                print(f"  [{ticker}] HAR h={H} failed: {e}")

            print(f"  Test [{ticker}]: SISO h={H}: "
                  f"R2={m['r2']:.4f} MAE={m['mae']:.5f}  |  "
                  f"HAR-RV: R2={row.get(f'har_h{H}_r2', float('nan')):.4f}")
        else:
            print(f"  [{ticker}] No valid test sequences")
    else:
        print(f"  [{ticker}] No test data from {GLOBAL_TEST_START}")

    return row


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train LSTM SISO for ablation study (R1 exception)"
    )
    ap.add_argument("--horizon", type=int, required=True, choices=[1, 5, 10],
                    help="Single forecast horizon to train (1, 5, or 10)")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Explicit tickers, e.g. --tickers VCB HPG FPT")
    ap.add_argument("--all", action="store_true",
                    help="Train all 30 VN30 stocks")
    args = ap.parse_args()

    H = args.horizon

    # set horizon-dependent globals
    global HORIZONS, MAX_H, CURVES_DIR, MODELS_DIR, CSV_OUT
    HORIZONS   = [H]
    MAX_H      = H
    CURVES_DIR = _root / "results" / f"lstm_siso_h{H}_curves"
    MODELS_DIR = _root / "models"  / f"lstm_siso_h{H}"
    CSV_OUT    = _root / "results" / f"lstm_siso_h{H}_results.csv"

    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        tickers = args.tickers
    elif args.all:
        tickers = VN30_TICKERS
    else:
        print("\n  Specify --all or --tickers.\n")
        return

    print(f"\n{'='*65}")
    print(f"  LSTM SISO Training (Ablation) -- h={H} -- {len(tickers)} stocks")
    print(f"  R1 EXCEPTION: single horizon, testing MIMO interference hypothesis")
    print(f"  HORIZONS={HORIZONS} | LOOKBACK={LOOKBACK} | N_FEATURES={N_FEATURES}")
    print(f"  Input: [rv_d, rv_w, rv_m]  (same as MIMO for controlled ablation)")
    print(f"  R6: test>={GLOBAL_TEST_START} | train/val="
          f"{int(TRAIN_VAL_SPLIT_RATIO*100)}/{100-int(TRAIN_VAL_SPLIT_RATIO*100)}")
    print(f"  EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | LR={LR}"
          f" | BATCH={BATCH_SIZE} | HIDDEN={HIDDEN}")
    print(f"{'='*65}\n")

    print(f"[Loading] {DATA_DIR} ...")
    close  = load_close_prices(DATA_DIR, tickers=tickers)
    rv_all = compute_all_rv(close)
    print(f"  Loaded: {close.shape[0]} dates x {close.shape[1]} tickers")
    print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}\n")

    all_rows: list[dict] = []
    for ticker in tickers:
        if ("rv_h1", ticker) not in rv_all.columns:
            print(f"  [{ticker}] Missing rv_h1 -- skip")
            continue
        if (f"rv_h{H}", ticker) not in rv_all.columns:
            print(f"  [{ticker}] Missing rv_h{H} -- skip")
            continue

        print(f"\n{'-'*65}")
        print(f"  Stock: {ticker}  (SISO h={H})")
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
        cols_show = ["ticker", f"ess_h{H}", "best_epoch",
                     f"lstm_h{H}_r2", f"lstm_h{H}_mae",
                     f"har_h{H}_r2",  f"har_h{H}_mae"]
        avail = [c for c in cols_show if c in df.columns]
        print(df[avail].to_string(index=False))

        print(f"\n  SISO h={H} AVG TEST METRICS:")
        if f"lstm_h{H}_r2" in df.columns:
            print(f"    Avg SISO   R2 : {df[f'lstm_h{H}_r2'].mean():.4f}")
        if f"har_h{H}_r2" in df.columns:
            print(f"    Avg HAR-RV R2 : {df[f'har_h{H}_r2'].mean():.4f}")
        if f"lstm_h{H}_mae" in df.columns:
            print(f"    Avg SISO   MAE: {df[f'lstm_h{H}_mae'].mean():.5f}")
    else:
        print("\n  No results to save.")

    print(f"\n{'='*65}")
    print(f"  DONE  |  Models: {MODELS_DIR}  |  Curves: {CURVES_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
