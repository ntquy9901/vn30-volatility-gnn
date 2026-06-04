"""
Comparison experiment: paper's GNNHAR1L architecture vs our HAR-residual approach.

Paper (Zhang et al. IJF 2024): GNNHAR1L
  H1 = Linear(in, n_h, bias=True)   -- trainable HAR-like skip
  H2 = GCN(in, hidden) -> Linear(hidden, n_h)   -- GNN correction
  output = H1 + H2   (ELU in GCN path; no final activation -- z-scored space)

Ours: GNNHARModel + pre-computed OLS HAR residuals
  output = OLS_HAR_pred + GNN(z_scored_residuals)

Key question: does trainable HAR skip generalize better than pre-computed OLS on VN30?

Usage:
  python gnn/compare_gnnhar_paper.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgl.nn import SAGEConv
from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS, ALL_NODES, NODE_IDX
from gnn.har_graph import build_static_graph, build_snapshots
from gnn.har_model import GNNHARModel
from baselines.har_rv_baseline import fit_har, predict_har

# ─────────────────────────── CONFIG ───────────────────────────────────────────
HORIZONS              = [1, 5, 10, 20]
MAX_H                 = max(HORIZONS)
STRIDE                = MAX_H
GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

IN_CHANNELS = 3
HIDDEN      = 16
DROPOUT     = 0.1
EPOCHS      = 500
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 50
SEED        = 42

N_NODES    = 31
N_STOCKS   = len(VN30_TICKERS)

_root       = Path(__file__).parent.parent
CONFIG_PATH = _root / "config.yaml"
RESULTS_DIR = _root / "results"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR       = _root / _cfg["data"]["prices_dir"]
CORR_THRESHOLD = float(_cfg["model"]["corr_threshold"])


# ─────────────────────────── PAPER MODEL ──────────────────────────────────────
class GNNHAR1L_MIMO(nn.Module):
    """
    Paper's GNNHAR1L adapted for MIMO (4 horizons, DGL SAGEConv).

    H1 = Linear(3, n_h)   -- trainable HAR-like skip (paper's innovation)
    H2 = SAGEConv -> Linear(hidden, n_h)  -- GNN graph-aware correction
    output = H1 + H2   -- skip connection, trained on z-scored targets
    """
    def __init__(self, in_channels=3, hidden=16, horizons=None, dropout=0.1):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 10, 20]
        n_h = len(horizons)
        self.har_skip = nn.Linear(in_channels, n_h, bias=True)  # paper's H1 (trainable)
        self.conv1    = SAGEConv(in_channels, hidden, aggregator_type="mean")
        self.drop     = nn.Dropout(dropout)
        self.heads    = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, g, x: torch.Tensor) -> torch.Tensor:
        h1 = self.har_skip(x)                           # (N, n_h) — trainable HAR skip
        h2 = F.elu(self.conv1(g, x))
        h2 = self.drop(h2)
        h2 = torch.cat([head(h2) for head in self.heads], dim=1)  # (N, n_h)
        return h1 + h2                                  # (N, n_h) — no final activation

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────── HELPERS ──────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    eps  = 1e-8
    ss_r = float(np.sum((y_true - y_pred) ** 2))
    ss_t = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = float(1.0 - ss_r / (ss_t + eps))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # QLIKE (Patton 2011): u = pred/true, loss = u - log(u) - 1
    u     = np.clip(y_pred, eps, None) / np.clip(y_true, eps, None)
    qlike = float(np.mean(u - np.log(u) - 1))
    return dict(r2=r2, mae=mae, rmse=rmse, qlike=qlike)


def _har_pred_for_dates(close_vn30, snapshot_dates, train_end_ts):
    n = len(snapshot_dates)
    out = np.zeros((n, N_STOCKS, len(HORIZONS)), dtype=np.float32)
    if n == 0:
        return out
    start_dt = snapshot_dates[0]
    for hi, h in enumerate(HORIZONS):
        rv_h_df = compute_rv(close_vn30, h=h)
        for si, ticker in enumerate(VN30_TICKERS):
            if ticker not in rv_h_df.columns:
                continue
            try:
                rv_full    = rv_h_df[ticker].dropna()
                coeffs     = fit_har(rv_full, train_end_ts)
                har_series = predict_har(rv_full, coeffs, start_dt)
                aligned    = har_series.reindex(snapshot_dates).fillna(0.0)
                out[:, si, hi] = aligned.values.astype(np.float32)
            except Exception:
                pass
    return out


def train_model(model, graph, loss_mask, X_train_n, y_train_n, X_val_n, y_val_n,
                n_train, n_val, label="model"):
    """Train any model, return (best_val, best_epoch, best_state_dict)."""
    crit = nn.MSELoss()
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xv = [torch.tensor(X_val_n[t]) for t in range(n_val)]
    yv = [torch.tensor(y_val_n[t]) for t in range(n_val)]

    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0

    print(f"\n  [{label}] params={model.count_params():,}")
    print(f"  {'Epoch':>5}  {'Train':>8}  {'H1':>7}  {'H5':>7}  {'H10':>7}  {'H20':>7}  {'Val':>8}")
    print(f"  {'-'*62}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm    = np.random.permutation(n_train)
        ep_loss = 0.0
        ep_h    = [0.0] * len(HORIZONS)
        for t_idx in perm:
            x_t = torch.tensor(X_train_n[t_idx])
            y_t = torch.tensor(y_train_n[t_idx])
            pred = model(graph, x_t)[loss_mask]
            h_losses = [crit(pred[:, i], y_t[:, i]) for i in range(len(HORIZONS))]
            loss = sum(h_losses)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item()
            for i, hl in enumerate(h_losses):
                ep_h[i] += hl.item()
        ep_loss /= n_train
        ep_h     = [v / n_train for v in ep_h]

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for t_idx in range(n_val):
                pred_v = model(graph, Xv[t_idx])[loss_mask]
                h_losses_v = [crit(pred_v[:, i], yv[t_idx][:, i]) for i in range(len(HORIZONS))]
                va_loss += sum(l.item() for l in h_losses_v)
        va_loss /= n_val

        if ep % 50 == 0 or ep <= 5:
            print(f"  Epoch {ep:3d}/{EPOCHS} | "
                  f"Train: {ep_loss:.4f} "
                  f"[{ep_h[0]:.3f} {ep_h[1]:.3f} {ep_h[2]:.3f} {ep_h[3]:.3f}] | "
                  f"Val: {va_loss:.4f}")

        if va_loss < best_val:
            best_val, best_ep = va_loss, ep
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop ep={ep}, best val={best_val:.4f} @ ep {best_ep}")
                break

    model.load_state_dict(best_sd)
    return best_val, best_ep


def predict_model(model, graph, loss_mask, X_test_n, n_test):
    model.eval()
    out = np.zeros((n_test, N_STOCKS, len(HORIZONS)), np.float32)
    with torch.no_grad():
        for t_idx in range(n_test):
            x_t = torch.tensor(X_test_n[t_idx])
            pred = model(graph, x_t)
            out[t_idx] = pred[loss_mask].numpy()
    return out  # (n_test, 30, 4) in z-scored space


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  GNNHAR Paper vs Our Approach -- VN30 Comparison")
    print(f"  Paper: GNNHAR1L (trainable HAR skip + GCN, Zhang et al. IJF 2024)")
    print(f"  Ours:  GNNHARModel + pre-computed OLS HAR residuals")
    print(f"{'='*70}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    tickers_all = VN30_TICKERS + ["VNINDEX"]
    close_all   = load_close_prices(DATA_DIR, tickers=tickers_all)
    log_ret_all = compute_log_returns(close_all)
    close_vn30  = close_all[[t for t in VN30_TICKERS if t in close_all.columns]]
    print(f"[Data] {close_all.shape[0]} dates x {close_all.shape[1]} tickers")

    # ── Snapshots ─────────────────────────────────────────────────────────────
    test_ts = pd.Timestamp(GLOBAL_TEST_START)
    X_pre, y_pre, dates_pre = build_snapshots(
        close_vn30, log_ret_all, HORIZONS, STRIDE,
        date_end=test_ts - pd.Timedelta(days=1),
    )
    n_pre   = len(dates_pre)
    n_train = int(n_pre * TRAIN_VAL_SPLIT_RATIO)
    n_val   = n_pre - n_train
    train_dates = dates_pre[:n_train]
    val_dates   = dates_pre[n_train:]
    X_train, y_train = X_pre[:n_train], y_pre[:n_train]
    X_val,   y_val   = X_pre[n_train:], y_pre[n_train:]

    X_test, y_test, test_dates = build_snapshots(
        close_vn30, log_ret_all, HORIZONS, stride=1, date_start=test_ts,
    )
    n_test = len(test_dates)
    train_end_ts = train_dates[-1]
    print(f"[Split] Train={n_train} Val={n_val} Test={n_test} snapshots")
    print(f"  Train: {train_dates[0].date()} -> {train_end_ts.date()}")

    # ── Graph ─────────────────────────────────────────────────────────────────
    gd        = build_static_graph(log_ret_all, train_end_ts, CORR_THRESHOLD)
    g         = gd.g
    loss_mask = gd.loss_mask

    # ── Feature normalization (shared by both models) ─────────────────────────
    feat_mu  = X_train.mean(axis=0)
    feat_sig = X_train.std(axis=0) + 1e-8
    def norm_x(X):
        return ((X - feat_mu[np.newaxis]) / feat_sig[np.newaxis]).astype(np.float32)
    X_train_n = norm_x(X_train)
    X_val_n   = norm_x(X_val)
    X_test_n  = norm_x(X_test) if n_test > 0 else np.empty((0, N_NODES, 3), np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL A: Paper's GNNHAR1L (trainable HAR skip, z-scored raw RV targets)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL A: Paper's GNNHAR1L -- trainable HAR skip, raw RV z-scored")
    print(f"{'='*70}")

    rv_mu_a  = y_train.mean(axis=0)          # (30, 4)
    rv_sig_a = y_train.std(axis=0)  + 1e-8
    def norm_y_a(y): return ((y - rv_mu_a[np.newaxis]) / rv_sig_a[np.newaxis]).astype(np.float32)
    y_train_a = norm_y_a(y_train)
    y_val_a   = norm_y_a(y_val)

    model_a = GNNHAR1L_MIMO(IN_CHANNELS, HIDDEN, HORIZONS, DROPOUT)
    train_model(model_a, g, loss_mask, X_train_n, y_train_a, X_val_n, y_val_a,
                n_train, n_val, label="GNNHAR1L_paper")

    pred_norm_a = predict_model(model_a, g, loss_mask, X_test_n, n_test)
    pred_orig_a = pred_norm_a * rv_sig_a[np.newaxis] + rv_mu_a[np.newaxis]
    pred_orig_a = np.clip(pred_orig_a, 0.0, None)

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL B: Our approach (HAR residuals, pre-computed OLS)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL B: Our GNNHARModel -- OLS HAR residuals")
    print(f"{'='*70}")

    print(f"\n  [HAR] Computing HAR predictions ...")
    har_train = _har_pred_for_dates(close_vn30, train_dates, train_end_ts)
    har_val   = _har_pred_for_dates(close_vn30, val_dates,   train_end_ts)
    har_test  = _har_pred_for_dates(close_vn30, test_dates,  train_end_ts) if n_test > 0 else np.zeros((0, N_STOCKS, len(HORIZONS)), np.float32)

    y_res_train = (y_train - har_train).astype(np.float32)
    y_res_val   = (y_val   - har_val).astype(np.float32)
    rv_mu_b     = y_res_train.mean(axis=0)
    rv_sig_b    = y_res_train.std(axis=0) + 1e-8
    def norm_y_b(yr): return ((yr - rv_mu_b[np.newaxis]) / rv_sig_b[np.newaxis]).astype(np.float32)
    y_train_b = norm_y_b(y_res_train)
    y_val_b   = norm_y_b(y_res_val)

    model_b = GNNHARModel(IN_CHANNELS, HIDDEN, HORIZONS, DROPOUT)
    train_model(model_b, g, loss_mask, X_train_n, y_train_b, X_val_n, y_val_b,
                n_train, n_val, label="GNNHARModel_ours")

    pred_norm_b = predict_model(model_b, g, loss_mask, X_test_n, n_test)
    gnn_residuals = pred_norm_b * rv_sig_b[np.newaxis] + rv_mu_b[np.newaxis]
    pred_orig_b   = har_test + gnn_residuals
    pred_orig_b   = np.clip(pred_orig_b, 0.0, None)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TEST RESULTS COMPARISON  ({n_test} test snapshots)")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} "
          f"{'HAR_h5':>8} {'PaperA_h5':>10} {'OursB_h5':>9} "
          f"{'HAR_h20':>8} {'PaperA_h20':>11} {'OursB_h20':>10}")
    print(f"  {'-'*70}")

    results = []
    for si, ticker in enumerate(VN30_TICKERS):
        row = {"ticker": ticker}
        for hi, h in enumerate(HORIZONS):
            rv_h_df = compute_rv(close_vn30, h=h)
            if ticker not in rv_h_df.columns:
                continue
            y_true = rv_h_df[ticker].reindex(test_dates).values
            valid  = ~np.isnan(y_true)
            if valid.sum() < 2:
                continue

            # Model A (paper)
            m_a = compute_metrics(y_true[valid], pred_orig_a[valid, si, hi])
            for met in ("r2", "mae", "rmse", "qlike"):
                row[f"paper_h{h}_{met}"] = m_a[met]

            # Model B (ours)
            m_b = compute_metrics(y_true[valid], pred_orig_b[valid, si, hi])
            for met in ("r2", "mae", "rmse", "qlike"):
                row[f"ours_h{h}_{met}"] = m_b[met]

            # Inline HAR
            try:
                rv_full    = rv_h_df[ticker].dropna()
                har_coeffs = fit_har(rv_full, train_end_ts)
                har_preds  = predict_har(rv_full, har_coeffs, test_ts)
                common     = test_dates.intersection(har_preds.index)
                if len(common) >= 2:
                    yt_har = rv_full.reindex(common).values
                    yp_har = har_preds.reindex(common).values
                    vm     = ~(np.isnan(yt_har) | np.isnan(yp_har))
                    if vm.sum() >= 2:
                        m_har = compute_metrics(yt_har[vm], yp_har[vm])
                        for met in ("r2", "mae", "rmse", "qlike"):
                            row[f"har_h{h}_{met}"] = m_har[met]
            except Exception:
                pass

        results.append(row)
        r_har5  = row.get("har_h5_r2",   float("nan"))
        r_a5    = row.get("paper_h5_r2", float("nan"))
        r_b5    = row.get("ours_h5_r2",  float("nan"))
        r_har20 = row.get("har_h20_r2",  float("nan"))
        r_a20   = row.get("paper_h20_r2", float("nan"))
        r_b20   = row.get("ours_h20_r2",  float("nan"))
        print(f"  {ticker:<8} "
              f"{r_har5:>8.4f} {r_a5:>10.4f} {r_b5:>9.4f} "
              f"{r_har20:>8.4f} {r_a20:>11.4f} {r_b20:>10.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "gnnhar_comparison.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  SUMMARY -- avg across {len(df)} stocks")
    print(f"{'='*70}")

    metrics_cfg = [
        ("r2",    "R2     ",  True),    # higher = better
        ("mae",   "MAE    ", False),    # lower = better
        ("rmse",  "RMSE   ", False),    # lower = better
        ("qlike", "QLIKE  ", False),    # lower = better
    ]

    for met, label, higher_better in metrics_cfg:
        print(f"\n  [{label}] (avg across stocks, {'higher' if higher_better else 'lower'} is better)")
        print(f"  {'h':>4}  {'HAR':>9}  {'Paper(A)':>10}  {'Ours(B)':>10}  {'B-HAR':>8}  {'Winner':>10}")
        print(f"  {'-'*62}")
        for h in HORIZONS:
            har_c  = f"har_h{h}_{met}"
            a_c    = f"paper_h{h}_{met}"
            b_c    = f"ours_h{h}_{met}"
            har_v  = df[har_c].mean() if har_c in df.columns else float("nan")
            a_v    = df[a_c].mean()   if a_c   in df.columns else float("nan")
            b_v    = df[b_c].mean()   if b_c   in df.columns else float("nan")
            d_bhar = b_v - har_v
            if higher_better:
                vals = {"HAR": har_v, "Paper": a_v, "Ours": b_v}
                winner = max(vals, key=vals.get)
            else:
                vals = {"HAR": har_v, "Paper": a_v, "Ours": b_v}
                winner = min(vals, key=vals.get)
            print(f"  h={h:>2}: {har_v:+10.4f}  {a_v:+10.4f}  {b_v:+10.4f}  {d_bhar:+8.4f}  {winner:>10}")

    print(f"\n  Full results -> {RESULTS_DIR / 'gnnhar_comparison.csv'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
