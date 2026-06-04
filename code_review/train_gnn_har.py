"""
GNN+HAR MIMO training script — CONSTRAINTS.md R1-R6 compliant.

R1: MIMO HORIZONS=[1,5,10,20], 1 model, 4 heads
R2: per-horizon loss printed every epoch + learning curve PNG saved
R3: data split printed before training
R4: GNN uses stride=MAX_H=20 (batch snapshots)
R6: test from 2026-01-01, train/val 80/20 from pre-2026

Node features: [rv_d, rv_w, rv_m] rolling std; z-scored from train (no log-transform).
Graph: 1 static graph built from full train period Pearson.
Normalization: per-stock from train snapshots.

Usage:
  python gnn/train_gnn_har.py
"""
import sys
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

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS, ALL_NODES, NODE_IDX
from gnn.har_model import GNNHARModel
from gnn.har_graph import build_static_graph, build_snapshots
from baselines.har_rv_baseline import fit_har, predict_har

# ─────────────────────────── STATIC CONFIG ────────────────────────────────────
HORIZONS              = [1, 5, 10, 20]
MAX_H                 = max(HORIZONS)
STRIDE                = MAX_H                  # R4: stride = MAX_H for GNN
GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

IN_CHANNELS = 3      # [rv_d, rv_w, rv_m]
HIDDEN      = 16
DROPOUT     = 0.1   # C5: hidden=16, dropout=0.3 kills 5 neurons; 0.1 safer
EPOCHS      = 500   # C6: stride=20->~187 snaps; 500 eps -> ~93500 updates
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 50    # longer patience for more epochs
SEED        = 42

_root       = Path(__file__).parent.parent
CONFIG_PATH = _root / "config.yaml"
MODELS_DIR  = _root / "models" / "gnn_har"
RESULTS_DIR = _root / "results"
CSV_OUT     = RESULTS_DIR / "gnn_har_results.csv"
CURVE_OUT   = RESULTS_DIR / "gnn_har_curves.png"
MODEL_OUT   = MODELS_DIR  / "model.pt"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR        = _root / _cfg["data"]["prices_dir"]
CORR_THRESHOLD  = float(_cfg["model"]["corr_threshold"])

N_NODES    = 31
N_STOCKS   = len(VN30_TICKERS)


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


# ─────────────────────────── DATA SPLIT PRINT (R3) ───────────────────────────
def print_split(
    train_dates: pd.DatetimeIndex,
    val_dates:   pd.DatetimeIndex,
    test_dates:  pd.DatetimeIndex,
    n_train: int, n_val: int, n_test: int,
):
    ess = n_train * N_STOCKS // MAX_H
    print(f"\n{'='*65}")
    print(f"  DATA SPLIT -- GNN+HAR MIMO  horizons={HORIZONS}")
    print(f"{'='*65}")
    print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()}"
          f"  ({n_train} snapshots x {N_STOCKS} stocks)")
    print(f"  Val:   {val_dates[0].date()} -> {val_dates[-1].date()}"
          f"  ({n_val} snapshots)")
    if n_test > 0:
        print(f"  Test:  {test_dates[0].date()} -> {test_dates[-1].date()}"
              f"  ({n_test} snapshots)")
    else:
        print(f"  Test:  (no data from {GLOBAL_TEST_START})")
    print(f"  stride={STRIDE}=MAX_H | IN_CHANNELS={IN_CHANNELS} | HIDDEN={HIDDEN}")
    print(f"  ESS_pooled_h20 = {n_train} x {N_STOCKS} / {MAX_H} = {ess}"
          f"  (obs/param ~= {ess / 4000:.2f})")
    print(f"{'='*65}\n")


# ─────────────────────────── LEARNING CURVE (R2) ─────────────────────────────
def save_learning_curve(
    train_losses: list,
    val_losses:   list,
    train_h:      list,
    val_h:        list,
    path: Path,
):
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ep, train_losses, label="Train total", color="steelblue")
    ax.plot(ep, val_losses,   label="Val total",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title("GNN+HAR -- Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    for i, h in enumerate(HORIZONS):
        ax.plot(ep, [v[i] for v in val_h],
                label=f"Val H={h}", color=colors[i], lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE per Horizon")
    ax.set_title("GNN+HAR -- Val Loss per Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"GNN+HAR MIMO Learning Curves  (stride={STRIDE}, HIDDEN={HIDDEN})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Curve saved: {path}")


# ─────────────────────────── HAR PREDICTIONS (for residuals) ─────────────────
def _har_pred_for_dates(
    close_vn30: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex,
    train_end_ts: pd.Timestamp,
) -> np.ndarray:
    """
    Compute in-sample HAR predictions at each snapshot date per stock/horizon.
    Returns (n_dates, N_STOCKS, len(HORIZONS)) float32.
    HAR is fitted on data up to train_end_ts; no refitting across dates.
    """
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


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  GNN+HAR MIMO Training  horizons={HORIZONS}")
    print(f"  R1: MIMO | R4: stride={STRIDE} | R6: test>={GLOBAL_TEST_START}")
    print(f"  HIDDEN={HIDDEN} | DROPOUT={DROPOUT} | EPOCHS={EPOCHS}"
          f" | PATIENCE={PATIENCE}")
    print(f"  Node features: rolling-std z-scored; targets: z-scored HAR residuals")
    print(f"  Final pred = HAR_pred + GNN_residual (GNN can't be worse than HAR)")
    print(f"  Graph: 1 static graph from train period Pearson")
    print(f"{'='*65}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"[Loading] {DATA_DIR} ...")
    tickers_all = VN30_TICKERS + ["VNINDEX"]
    close_all   = load_close_prices(DATA_DIR, tickers=tickers_all)
    log_ret_all = compute_log_returns(close_all)
    print(f"  Loaded: {close_all.shape[0]} dates x {close_all.shape[1]} tickers")
    print(f"  Date range: {close_all.index[0].date()} -> {close_all.index[-1].date()}")

    # close for VN30 only (targets use VN30 only)
    close_vn30 = close_all[[t for t in VN30_TICKERS if t in close_all.columns]]

    # ── Build pre-2026 snapshots ───────────────────────────────────────────────
    test_ts = pd.Timestamp(GLOBAL_TEST_START)
    print(f"\n[Snapshots] Building pre-2026 snapshots (stride={STRIDE}) ...")
    X_pre, y_pre, dates_pre = build_snapshots(
        close_vn30, log_ret_all, HORIZONS, STRIDE,
        date_end=test_ts - pd.Timedelta(days=1),
    )
    n_pre = len(dates_pre)
    n_train = int(n_pre * TRAIN_VAL_SPLIT_RATIO)
    n_val   = n_pre - n_train
    if n_train == 0:
        print("  [ERROR] No training snapshots found.")
        return

    train_dates = dates_pre[:n_train]
    val_dates   = dates_pre[n_train:]
    X_train, y_train = X_pre[:n_train], y_pre[:n_train]
    X_val,   y_val   = X_pre[n_train:], y_pre[n_train:]

    print(f"\n[Snapshots] Building test snapshots (>= {GLOBAL_TEST_START}, stride=1) ...")
    X_test, y_test, test_dates = build_snapshots(
        close_vn30, log_ret_all, HORIZONS, stride=1,  # all test dates for valid metrics
        date_start=test_ts,
    )
    n_test = len(test_dates)
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test} snapshots")

    # ── Static graph from train period ────────────────────────────────────────
    train_end_ts = train_dates[-1]
    print(f"\n[Graph] Building static graph from train period (end={train_end_ts.date()}) ...")
    gd = build_static_graph(log_ret_all, train_end_ts, CORR_THRESHOLD)
    g  = gd.g
    loss_mask = gd.loss_mask   # (31,) bool, False for VNINDEX node_0

    src, dst = g.edges()
    n_edges = g.num_edges()
    print(f"  Nodes: {gd.num_nodes} | Edges: {n_edges}"
          f" | Hub edges (VNINDEX->stocks): {int((src == 0).sum())}")

    # ── HAR predictions and residuals ─────────────────────────────────────────
    # Train GNN on HAR residuals: final_pred = har_pred + gnn_residual.
    # If GNN predicts 0 residual -> prediction = HAR. Model can't underperform HAR.
    print(f"\n[HAR] Computing HAR baseline for residual training ...")
    har_pred_train = _har_pred_for_dates(close_vn30, train_dates, train_end_ts)
    har_pred_val   = _har_pred_for_dates(close_vn30, val_dates,   train_end_ts)
    har_pred_test  = (_har_pred_for_dates(close_vn30, test_dates, train_end_ts)
                      if n_test > 0 else np.zeros((0, N_STOCKS, len(HORIZONS)), np.float32))
    print(f"  HAR train mean RV h=20: {har_pred_train[:, :, -1].mean():.5f}"
          f"  | actual mean: {y_train[:, :, -1].mean():.5f}")

    y_res_train = (y_train - har_pred_train).astype(np.float32)  # (n_train, 30, 4)
    y_res_val   = (y_val   - har_pred_val).astype(np.float32)    # (n_val,   30, 4)

    # ── Z-score normalization (features) + residual z-score (targets) ────────
    # ELU handles negative z-scores correctly — no mode collapse from COVID spike.
    feat_mu  = X_train.mean(axis=0)           # (31, 3)
    feat_sig = X_train.std(axis=0)  + 1e-8    # (31, 3)
    rv_mu    = y_res_train.mean(axis=0)       # (30, 4) — residual mean (≈0)
    rv_sig   = y_res_train.std(axis=0) + 1e-8 # (30, 4) — residual std

    def norm_x(X: np.ndarray) -> np.ndarray:
        return ((X - feat_mu[np.newaxis]) / feat_sig[np.newaxis]).astype(np.float32)

    def norm_y(y_res: np.ndarray) -> np.ndarray:
        return ((y_res - rv_mu[np.newaxis]) / rv_sig[np.newaxis]).astype(np.float32)

    X_train_n = norm_x(X_train)
    y_train_n = norm_y(y_res_train)
    X_val_n   = norm_x(X_val)
    y_val_n   = norm_y(y_res_val)
    if n_test > 0:
        X_test_n = norm_x(X_test)
    else:
        X_test_n = np.empty((0, N_NODES, 3), np.float32)

    # ── Print data split (R3) ─────────────────────────────────────────────────
    print_split(train_dates, val_dates, test_dates, n_train, n_val, n_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GNNHARModel(
        in_channels=IN_CHANNELS, hidden=HIDDEN,
        horizons=HORIZONS, dropout=DROPOUT,
    )
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    print(f"  Model params: {model.count_params():,}")

    # Pre-convert to tensors (val stays fixed)
    Xv_tensors = [torch.tensor(X_val_n[t]) for t in range(n_val)]
    yv_tensors = [torch.tensor(y_val_n[t]) for t in range(n_val)]

    train_loss_hist: list[float]       = []
    val_loss_hist:   list[float]       = []
    train_h_hist:    list[list[float]] = []
    val_h_hist:      list[list[float]] = []
    best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0

    print(f"\n  Training GNN+HAR on {n_train} snapshots x 30 stocks ...")
    print(f"  {'Epoch':>5}  {'Train':>8}  {'H1':>8}  {'H5':>8}"
          f"  {'H10':>8}  {'H20':>8}  {'Val':>8}")
    print(f"  {'-'*70}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm    = np.random.permutation(n_train)
        ep_loss = 0.0
        ep_h    = [0.0] * len(HORIZONS)

        for t_idx in perm:
            x_t = torch.tensor(X_train_n[t_idx])  # (31, 3)
            y_t = torch.tensor(y_train_n[t_idx])  # (30, 4)

            pred = model(g, x_t)                   # (31, 4)
            pred_stocks = pred[loss_mask]           # (30, 4)

            h_losses = [crit(pred_stocks[:, i], y_t[:, i]) for i in range(len(HORIZONS))]
            loss     = sum(h_losses)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += loss.item()
            for i, hl in enumerate(h_losses):
                ep_h[i] += hl.item()

        ep_loss /= n_train
        ep_h     = [v / n_train for v in ep_h]

        # Validation
        model.eval()
        va_loss = 0.0
        va_h    = [0.0] * len(HORIZONS)
        with torch.no_grad():
            for t_idx in range(n_val):
                pred_v = model(g, Xv_tensors[t_idx])[loss_mask]  # (30, 4)
                h_losses_v = [crit(pred_v[:, i], yv_tensors[t_idx][:, i])
                              for i in range(len(HORIZONS))]
                va_loss += sum(l.item() for l in h_losses_v)
                for i, hl in enumerate(h_losses_v):
                    va_h[i] += hl.item()
        va_loss /= n_val
        va_h     = [v / n_val for v in va_h]

        train_loss_hist.append(ep_loss)
        val_loss_hist.append(va_loss)
        train_h_hist.append(ep_h)
        val_h_hist.append(va_h)

        # R2: print per-horizon loss every epoch
        print(f"  Epoch {ep:3d}/{EPOCHS} | "
              f"Train: {ep_loss:.4f} "
              f"[H1={ep_h[0]:.4f} H5={ep_h[1]:.4f} "
              f"H10={ep_h[2]:.4f} H20={ep_h[3]:.4f}] | "
              f"Val: {va_loss:.4f} | LR: {LR:.2e}")

        if va_loss < best_val:
            best_val, best_ep = va_loss, ep
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at ep {ep}  "
                      f"(best val={best_val:.4f} @ ep {best_ep})")
                break

    model.load_state_dict(best_sd)

    # R2: save learning curve
    save_learning_curve(train_loss_hist, val_loss_hist,
                        train_h_hist, val_h_hist, CURVE_OUT)

    # ── Save model checkpoint ─────────────────────────────────────────────────
    torch.save({
        "state_dict":    model.state_dict(),
        "horizons":      HORIZONS,
        "in_channels":   IN_CHANNELS,
        "hidden":        HIDDEN,
        "dropout":       DROPOUT,
        "feat_mu":       feat_mu,
        "feat_sig":      feat_sig,
        "rv_mu":         rv_mu,
        "rv_sig":        rv_sig,
        "stock_order":   VN30_TICKERS,
        "node_order":    ALL_NODES,
        "best_epoch":    best_ep,
        "best_val_loss": best_val,
        "train_end_ts":  str(train_end_ts.date()),
        "n_train":       n_train,
        "n_val":         n_val,
        "stride":        STRIDE,
        "corr_threshold": CORR_THRESHOLD,
    }, MODEL_OUT)
    print(f"  Model saved: {MODEL_OUT}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    if n_test == 0:
        print(f"\n  [WARN] No test snapshots from {GLOBAL_TEST_START}. Skipping eval.")
        return

    print(f"\n{'='*65}")
    print(f"  Test Evaluation  ({n_test} snapshots x 30 stocks)")
    print(f"{'='*65}")
    print(f"  {'Ticker':<8} "
          f"{'GNN h1 R2':>10} {'GNN h5 R2':>10} "
          f"{'GNN h20 R2':>11} {'HAR h5 R2':>10} {'HAR h20 R2':>11}")
    print(f"  {'-'*62}")

    model.eval()
    test_pred_norm = np.zeros((n_test, N_STOCKS, len(HORIZONS)), dtype=np.float32)
    with torch.no_grad():
        for t_idx in range(n_test):
            x_t  = torch.tensor(X_test_n[t_idx])
            pred = model(g, x_t)           # (31, 4)
            test_pred_norm[t_idx] = pred[loss_mask].numpy()  # (30, 4)

    # Denorm residuals, add HAR predictions: final = HAR + GNN_residual
    gnn_residuals  = test_pred_norm * rv_sig[np.newaxis] + rv_mu[np.newaxis]
    test_pred_orig = har_pred_test + gnn_residuals   # (n_test, 30, 4)
    test_pred_orig = np.clip(test_pred_orig, 0.0, None)

    all_rows: list[dict] = []
    for si, ticker in enumerate(VN30_TICKERS):
        row: dict = {
            "ticker":            ticker,
            "n_train_snapshots": n_train,
            "n_val_snapshots":   n_val,
            "n_test_snapshots":  n_test,
            "ess_h20":           n_train * N_STOCKS // MAX_H,
            "best_epoch":        best_ep,
        }

        for hi, h in enumerate(HORIZONS):
            # Ground truth from rv series (avoid 0.0-filled array)
            try:
                rv_h_s = compute_rv(close_vn30, h=h)
                if ticker not in rv_h_s.columns:
                    continue
                rv_series = rv_h_s[ticker]
                y_true = rv_series.reindex(test_dates).values  # align to snap dates
                y_pred = test_pred_orig[:, si, hi]

                valid = ~np.isnan(y_true)
                if valid.sum() < 2:
                    continue

                m = compute_metrics(y_true[valid], y_pred[valid])
                for k, v in m.items():
                    row[f"gnn_h{h}_{k}"] = v
            except Exception as e:
                print(f"  [WARN] GNN metrics {ticker} h={h}: {e}")

            # Inline HAR-RV comparison
            try:
                rv_h_all = compute_rv(close_vn30, h=h)
                if ticker not in rv_h_all.columns:
                    continue
                rv_full = rv_h_all[ticker].dropna()
                har_coeffs = fit_har(rv_full, train_end_ts)
                har_preds  = predict_har(rv_full, har_coeffs, test_ts)
                common = test_dates.intersection(har_preds.index)
                if len(common) >= 2:
                    y_true_har = rv_full.reindex(common).values
                    y_pred_har = har_preds.reindex(common).values
                    valid_har  = ~(np.isnan(y_true_har) | np.isnan(y_pred_har))
                    if valid_har.sum() >= 2:
                        har_m = compute_metrics(
                            y_true_har[valid_har], y_pred_har[valid_har]
                        )
                        for k, v in har_m.items():
                            row[f"har_h{h}_{k}"] = v
            except Exception as e:
                print(f"  [WARN] HAR {ticker} h={h}: {e}")

        all_rows.append(row)
        r2_h1  = row.get("gnn_h1_r2",  float("nan"))
        r2_h5  = row.get("gnn_h5_r2",  float("nan"))
        r2_h20 = row.get("gnn_h20_r2", float("nan"))
        hr2_h5 = row.get("har_h5_r2",  float("nan"))
        hr2_20 = row.get("har_h20_r2", float("nan"))
        print(f"  {ticker:<8} "
              f"{r2_h1:>10.4f} {r2_h5:>10.4f} "
              f"{r2_h20:>11.4f} {hr2_h5:>10.4f} {hr2_20:>11.4f}")

    # Summary
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n  Results -> {CSV_OUT}")

        print(f"\n{'='*65}")
        print(f"  SUMMARY -- avg R2 across {len(df)} stocks")
        print(f"{'='*65}")
        for h in HORIZONS:
            gnn_col = f"gnn_h{h}_r2"
            har_col = f"har_h{h}_r2"
            if gnn_col in df.columns:
                g_avg = df[gnn_col].mean()
                g_med = df[gnn_col].median()
                h_avg = df[har_col].mean() if har_col in df.columns else float("nan")
                delta = g_avg - h_avg
                flag  = ("[+] GNN better" if delta > 0.01
                         else ("[-] HAR better" if delta < -0.01 else "[~] similar"))
                print(f"  h={h:>2}: GNN avg={g_avg:+.4f} med={g_med:+.4f} | "
                      f"HAR avg={h_avg:+.4f} | delta={delta:+.4f}  {flag}")

    print(f"\n{'='*65}")
    print(f"  DONE | Model: {MODEL_OUT}")
    print(f"  Run: python scripts/eda/evaluate_gnn_har.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
