"""
CONSTRAINTS.md R1 EXCEPTION: Pooled cross-stock LSTM SISO ablation.

Pools sequences from all 30 VN30 stocks into a single training set.
Each stock is normalized independently (per-stock mu/sig) before pooling
so the model learns the structural HAR -> RV relationship at a common scale.

Hypothesis: Per-stock ESS bottleneck (obs/param ~= 0.09) is the primary
cause of LSTM failure. Pooled ESS ~= 2.26 (30x improvement) should help.

Comparison:
  Pooled  >> Per-stock SISO  -> ESS bottleneck confirmed, transfer works
  Pooled ~= Per-stock SISO   -> Distribution heterogeneity blocks transfer
  Pooled  <  Per-stock SISO  -> Cross-stock noise hurts

Usage:
  python baselines/train_lstm_siso_pooled.py --horizon 5
  python baselines/train_lstm_siso_pooled.py --horizon 1
  python baselines/train_lstm_siso_pooled.py --horizon 10
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
PATIENCE   = 25
SEED       = 42
HIDDEN     = 64
DROPOUT    = 0.2

# Horizon-dependent globals — set in main() after argparse
HORIZONS:   list = []
MAX_H:      int  = 0
MODELS_DIR: Path = Path()
CSV_OUT:    Path = Path()
CURVE_OUT:  Path = Path()

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


# ─────────────────────────── MODEL ────────────────────────────────────────────
class LSTMModelSISO(nn.Module):
    """LSTM backbone + 1 output head. Same as train_lstm_siso.py."""

    def __init__(self, n_features=N_FEATURES, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        feat   = self.drop(out[:, -1, :])
        return self.head(feat)


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


# ─────────────────────────── LEARNING CURVE ───────────────────────────────────
def save_learning_curve(
    train_losses: list,
    val_losses:   list,
    path: Path,
    h: int,
    n_stocks: int,
):
    ep = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, train_losses, label="Train (pooled)", color="steelblue")
    ax.plot(ep, val_losses,   label="Val (pooled)",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized scale)")
    ax.set_title(
        f"LSTM SISO Pooled h={h} -- Learning Curve\n"
        f"({n_stocks} stocks pooled, per-stock normalization)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Curve saved: {path}")


# ─────────────────────────── BUILD POOLED DATASET ─────────────────────────────
def build_pooled_dataset(
    rv_all: pd.DataFrame,
    tickers: list[str],
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    """
    Returns:
      all_X_train, all_y_train, all_X_val, all_y_val  -- pooled normalized sequences
      norm_params  -- {ticker: {feat_mu, feat_sig, rv_mu_h, rv_sig_h}}
      valid_tickers -- tickers that contributed to the pool
    """
    test_ts = pd.Timestamp(GLOBAL_TEST_START)

    all_X_train: list[np.ndarray] = []
    all_y_train: list[np.ndarray] = []
    all_X_val:   list[np.ndarray] = []
    all_y_val:   list[np.ndarray] = []
    norm_params: dict = {}
    valid_tickers: list[str] = []

    print(f"\n  Building pooled dataset for h={H} ...")
    print(f"  {'Ticker':<8} {'N_train':>8} {'N_val':>7} "
          f"{'ESS_h{H}':>10} {'Seqs_tr':>8} {'Seqs_va':>8}")
    print(f"  {'-'*58}")

    for ticker in tickers:
        if ("rv_h1", ticker) not in rv_all.columns:
            continue
        rv_col = f"rv_h{H}"
        if (rv_col, ticker) not in rv_all.columns:
            continue

        rv_h1_series = rv_all[("rv_h1", ticker)]
        har_df       = build_har_features(rv_h1_series)
        combined     = pd.DataFrame({H: rv_all[(rv_col, ticker)]}).join(har_df).dropna()

        min_rows = LOOKBACK + H + 10
        if len(combined) < min_rows:
            continue

        comb_pre  = combined[combined.index < test_ts]
        if len(comb_pre) < LOOKBACK + 10:
            continue

        n_tr       = int(len(comb_pre) * TRAIN_VAL_SPLIT_RATIO)
        comb_train = comb_pre.iloc[:n_tr]
        comb_val   = comb_pre.iloc[n_tr:]

        # per-stock normalization from train split only
        feat_mu:  dict = {}
        feat_sig: dict = {}
        for feat in FEAT_NAMES:
            feat_mu[feat]  = float(comb_train[feat].mean())
            feat_sig[feat] = float(comb_train[feat].std()) + 1e-8

        rv_mu_h  = float(comb_train[H].mean())
        rv_sig_h = float(comb_train[H].std()) + 1e-8

        def _norm_x(df_slice: pd.DataFrame, _mu=feat_mu, _sig=feat_sig) -> np.ndarray:
            out = np.zeros((len(df_slice), N_FEATURES), dtype=np.float32)
            for j, f in enumerate(FEAT_NAMES):
                out[:, j] = (df_slice[f].values - _mu[f]) / _sig[f]
            return out

        def _norm_y(df_slice: pd.DataFrame, _mu=rv_mu_h, _sig=rv_sig_h) -> np.ndarray:
            return ((df_slice[H].values - _mu) / _sig).reshape(-1, 1).astype(np.float32)

        val_ctx = pd.concat([comb_train.iloc[-LOOKBACK:], comb_val])

        X_tr, y_tr = make_sequences(_norm_x(comb_train), _norm_y(comb_train), LOOKBACK)
        X_va, y_va = make_sequences(_norm_x(val_ctx),    _norm_y(val_ctx),    LOOKBACK)

        if len(X_tr) < 5:
            continue

        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_X_val.append(X_va)
        all_y_val.append(y_va)

        norm_params[ticker] = {
            "feat_mu":  feat_mu,
            "feat_sig": feat_sig,
            "rv_mu_h":  rv_mu_h,
            "rv_sig_h": rv_sig_h,
            "comb_train": comb_train,
            "comb_val":   comb_val,
        }
        valid_tickers.append(ticker)

        ess = len(comb_train) // H
        print(f"  {ticker:<8} {len(comb_train):>8,} {len(comb_val):>7,} "
              f"{ess:>10,} {len(X_tr):>8,} {len(X_va):>8,}")

    pool_X_tr = np.concatenate(all_X_train, axis=0)
    pool_y_tr = np.concatenate(all_y_train, axis=0)
    pool_X_va = np.concatenate(all_X_val,   axis=0)
    pool_y_va = np.concatenate(all_y_val,   axis=0)

    return pool_X_tr, pool_y_tr, pool_X_va, pool_y_va, norm_params, valid_tickers


# ─────────────────────────── PRINT SPLIT SUMMARY (R3) ────────────────────────
def print_pooled_split(
    pool_X_tr: np.ndarray,
    pool_X_va: np.ndarray,
    valid_tickers: list[str],
    H: int,
):
    n_tr   = len(pool_X_tr)
    n_va   = len(pool_X_va)
    ess    = n_tr // H
    n_stk  = len(valid_tickers)
    print(f"\n{'='*65}")
    print(f"  POOLED DATA SPLIT -- LSTM SISO Pooled h={H}")
    print(f"{'='*65}")
    print(f"  Stocks pooled:     {n_stk}")
    print(f"  Train sequences:   {n_tr:,}  (sum across {n_stk} stocks)")
    print(f"  Val   sequences:   {n_va:,}  (sum across {n_stk} stocks)")
    print(f"  ESS_pooled_h{H}:    {ess:,}  (pooled_train_seqs / H={H})")
    print(f"  Approx obs/param:  {ess / 4000:.2f}  (ESS / 4000 LSTM params)")
    print(f"  Normalization:     per-stock (train split only)")
    print(f"  Test split:        per-stock, date >= {GLOBAL_TEST_START}")
    print(f"  LOOKBACK={LOOKBACK} | N_FEATURES={N_FEATURES} | BATCH={BATCH_SIZE}")
    print(f"  EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | HIDDEN={HIDDEN}")
    print(f"{'='*65}\n")


# ─────────────────────────── TRAIN POOLED MODEL ───────────────────────────────
def train_pooled(
    pool_X_tr: np.ndarray,
    pool_y_tr: np.ndarray,
    pool_X_va: np.ndarray,
    pool_y_va: np.ndarray,
    H: int,
) -> tuple[LSTMModelSISO, int]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xt = torch.tensor(pool_X_tr)
    yt = torch.tensor(pool_y_tr)
    Xv = torch.tensor(pool_X_va)
    yv = torch.tensor(pool_y_va)

    model = LSTMModelSISO(n_features=N_FEATURES, hidden=HIDDEN, dropout=DROPOUT)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0
    n_seqs = len(Xt)

    print(f"  Training pooled model: {n_seqs:,} train seqs | "
          f"{len(Xv):,} val seqs | SISO h={H}")
    print(f"  EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | LR={LR} | BATCH={BATCH_SIZE}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_seqs)
        ep_loss, n_b = 0.0, 0
        for s in range(0, n_seqs, BATCH_SIZE):
            bi  = perm[s : s + BATCH_SIZE]
            xb  = Xt[bi]
            yb  = yt[bi]
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

        print(f"  Epoch {ep:3d}/{EPOCHS} | "
              f"Train: {tr_l:.4f} | "
              f"Val: {va_l:.4f} | "
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
    save_learning_curve(train_losses, val_losses, CURVE_OUT, H, n_stk=0)

    return model, best_ep


# ─────────────────────────── PER-STOCK TEST EVALUATION ────────────────────────
def evaluate_per_stock(
    model: LSTMModelSISO,
    rv_all: pd.DataFrame,
    valid_tickers: list[str],
    norm_params: dict,
    H: int,
) -> list[dict]:
    test_ts   = pd.Timestamp(GLOBAL_TEST_START)
    all_rows: list[dict] = []

    print(f"\n  Per-stock test evaluation (h={H}) ...")
    print(f"  {'Ticker':<8} {'N_test':>7} {'Pooled R2':>10} {'HAR-RV R2':>10}")
    print(f"  {'-'*40}")

    for ticker in valid_tickers:
        params     = norm_params[ticker]
        feat_mu    = params["feat_mu"]
        feat_sig   = params["feat_sig"]
        rv_mu_h    = params["rv_mu_h"]
        rv_sig_h   = params["rv_sig_h"]
        comb_pre   = pd.concat([params["comb_train"], params["comb_val"]])

        rv_col   = f"rv_h{H}"
        rv_h1_s  = rv_all[("rv_h1", ticker)]
        har_df   = build_har_features(rv_h1_s)
        combined = pd.DataFrame({H: rv_all[(rv_col, ticker)]}).join(har_df).dropna()
        comb_test = combined[combined.index >= test_ts]

        row: dict = {
            "ticker":  ticker,
            "n_train": len(params["comb_train"]),
            "n_val":   len(params["comb_val"]),
            "n_test":  len(comb_test),
        }

        if len(comb_test) == 0:
            print(f"  {ticker:<8} {'no test':>7}")
            all_rows.append(row)
            continue

        def _norm_x(df_slice):
            out = np.zeros((len(df_slice), N_FEATURES), dtype=np.float32)
            for j, f in enumerate(FEAT_NAMES):
                out[:, j] = (df_slice[f].values - feat_mu[f]) / feat_sig[f]
            return out

        def _norm_y(df_slice):
            return ((df_slice[H].values - rv_mu_h) / rv_sig_h
                    ).reshape(-1, 1).astype(np.float32)

        test_ctx = pd.concat([comb_pre.iloc[-LOOKBACK:], comb_test])
        X_te, y_te = make_sequences(_norm_x(test_ctx), _norm_y(test_ctx), LOOKBACK)

        if len(X_te) == 0:
            print(f"  {ticker:<8} {'0 seqs':>7}")
            all_rows.append(row)
            continue

        Xte_t = torch.tensor(X_te)
        model.eval()
        with torch.no_grad():
            preds_n = model(Xte_t).numpy()

        y_true = y_te[:, 0] * rv_sig_h + rv_mu_h
        y_pred = np.clip(preds_n[:, 0] * rv_sig_h + rv_mu_h, 0.0, None)

        m = compute_metrics(y_true, y_pred)
        for k, v in m.items():
            row[f"lstm_pooled_h{H}_{k}"] = v

        # inline HAR-RV
        rv_series = rv_all[(rv_col, ticker)]
        try:
            train_end_ts = params["comb_train"].index[-1]
            coeffs       = fit_har(rv_series, train_end_ts)
            har_pred     = predict_har(rv_series, coeffs, test_ts)
            n_seq        = len(X_te)
            test_index   = comb_test.index[:n_seq]
            common       = test_index.intersection(har_pred.index)
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

        lstm_r2 = m["r2"]
        har_r2  = row.get(f"har_h{H}_r2", float("nan"))
        print(f"  {ticker:<8} {len(comb_test):>7} {lstm_r2:>10.4f} {har_r2:>10.4f}")
        all_rows.append(row)

    return all_rows


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train pooled cross-stock LSTM SISO (R1 exception)"
    )
    ap.add_argument("--horizon", type=int, required=True, choices=[1, 5, 10],
                    help="Single forecast horizon (1, 5, or 10)")
    args = ap.parse_args()

    H = args.horizon

    global HORIZONS, MAX_H, MODELS_DIR, CSV_OUT, CURVE_OUT
    HORIZONS   = [H]
    MAX_H      = H
    MODELS_DIR = _root / "models"  / f"lstm_siso_pooled_h{H}"
    CSV_OUT    = _root / "results" / f"lstm_siso_pooled_h{H}_results.csv"
    CURVE_OUT  = _root / "results" / f"lstm_siso_pooled_h{H}_curve.png"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  LSTM SISO Pooled Training -- h={H} -- All 30 VN30 stocks")
    print(f"  R1 EXCEPTION: pooled cross-stock, testing ESS bottleneck hypothesis")
    print(f"  Input: [rv_d, rv_w, rv_m]  (per-stock normalized)")
    print(f"  Normalization: per-stock (train split only) before pooling")
    print(f"  R6: test>={GLOBAL_TEST_START} | train/val="
          f"{int(TRAIN_VAL_SPLIT_RATIO*100)}/{100-int(TRAIN_VAL_SPLIT_RATIO*100)}")
    print(f"{'='*65}\n")

    print(f"[Loading] {DATA_DIR} ...")
    close  = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    rv_all = compute_all_rv(close)
    print(f"  Loaded: {close.shape[0]} dates x {close.shape[1]} tickers")
    print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}\n")

    # Phase 1: build pooled dataset
    pool_X_tr, pool_y_tr, pool_X_va, pool_y_va, norm_params, valid_tickers = \
        build_pooled_dataset(rv_all, VN30_TICKERS, H)

    n_stk = len(valid_tickers)
    if n_stk == 0:
        print("  [ERROR] No valid stocks found.")
        return

    # Phase 2: print split summary (R3)
    print_pooled_split(pool_X_tr, pool_X_va, valid_tickers, H)

    # monkey-patch n_stk into curve function
    global _n_stk_for_curve
    _n_stk_for_curve = n_stk

    # Phase 3: train
    print(f"{'='*65}")
    print(f"  Training pooled model on {len(pool_X_tr):,} sequences "
          f"({n_stk} stocks)")
    print(f"{'='*65}")

    # rebuild the learning curve saver with n_stk available
    def _save_curve_with_nstk(train_l, val_l, path, h):
        save_learning_curve(train_l, val_l, path, h, n_stk)

    # train
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xt = torch.tensor(pool_X_tr)
    yt = torch.tensor(pool_y_tr)
    Xv = torch.tensor(pool_X_va)
    yv = torch.tensor(pool_y_va)

    model = LSTMModelSISO(n_features=N_FEATURES, hidden=HIDDEN, dropout=DROPOUT)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0
    n_seqs = len(Xt)

    print(f"  {n_seqs:,} train seqs | {len(Xv):,} val seqs | SISO h={H}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_seqs)
        ep_loss, n_b = 0.0, 0
        for s in range(0, n_seqs, BATCH_SIZE):
            bi  = perm[s : s + BATCH_SIZE]
            xb  = Xt[bi]
            yb  = yt[bi]
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

        print(f"  Epoch {ep:3d}/{EPOCHS} | "
              f"Train: {tr_l:.4f} | "
              f"Val: {va_l:.4f} | "
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
    _save_curve_with_nstk(train_losses, val_losses, CURVE_OUT, H)

    torch.save({
        "state_dict":    model.state_dict(),
        "horizons":      HORIZONS,
        "lookback":      LOOKBACK,
        "hidden":        HIDDEN,
        "n_features":    N_FEATURES,
        "valid_tickers": valid_tickers,
        "norm_params":   {
            t: {
                k: v for k, v in norm_params[t].items()
                if k not in ("comb_train", "comb_val")
            }
            for t in valid_tickers
        },
        "best_epoch":    best_ep,
        "pooled_n_train": len(pool_X_tr),
        "pooled_n_val":   len(pool_X_va),
    }, MODELS_DIR / "pooled_model.pt")
    print(f"  Model saved: {MODELS_DIR / 'pooled_model.pt'}")

    # Phase 4: per-stock test evaluation
    all_rows = evaluate_per_stock(model, rv_all, valid_tickers, norm_params, H)

    # add pooled ESS to each row
    ess_pooled = len(pool_X_tr) // H
    for row in all_rows:
        row[f"ess_pooled_h{H}"] = ess_pooled

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n{'='*65}")
        print(f"  Results -> {CSV_OUT}")
        print(f"{'='*65}")
        cols_show = ["ticker", "n_test", f"ess_pooled_h{H}",
                     f"lstm_pooled_h{H}_r2", f"lstm_pooled_h{H}_mae",
                     f"har_h{H}_r2",         f"har_h{H}_mae"]
        avail = [c for c in cols_show if c in df.columns]
        print(df[avail].to_string(index=False))

        print(f"\n  POOLED SISO h={H} AVG TEST METRICS:")
        lstm_col = f"lstm_pooled_h{H}_r2"
        har_col  = f"har_h{H}_r2"
        if lstm_col in df.columns:
            print(f"    Avg Pooled R2  : {df[lstm_col].mean():.4f}")
            print(f"    Med Pooled R2  : {df[lstm_col].median():.4f}")
        if har_col in df.columns:
            print(f"    Avg HAR-RV R2  : {df[har_col].mean():.4f}")
    else:
        print("\n  No results to save.")

    print(f"\n{'='*65}")
    print(f"  DONE  |  Model: {MODELS_DIR / 'pooled_model.pt'}")
    print(f"  Curve: {CURVE_OUT}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
