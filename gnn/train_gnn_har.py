"""
GNN+HAR SISO training -- one model per horizon, VN30 only (no VNINDEX).

For each h in [1, 5, 10, 20]:
  - Build snapshots: stride=MAX_H train/val, stride=1 test
  - Train GNNHARModel on z-scored HAR residuals
  - Per-stock evaluation: R2, MAE, RMSE, QLIKE

Output:
  models/gnn_har/model_h{h}.pt       x4
  results/gnn_har_curves_h{h}.png    x4
  results/gnn_har_siso_results.csv   (120 rows: 30 stocks x 4 horizons)

Constraints:
  R1: SISO (one model per horizon, HORIZONS=[1,5,10,20] configurable)
  R2: loss printed every epoch + learning curve PNG per horizon
  R3: data split printed before each horizon training
  R4: stride=MAX_H for train/val snapshots
  R6: test from 2026-01-01, train/val 80/20 split from pre-2026
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
from gnn.build_graph import VN30_TICKERS
from gnn.har_model import GNNHARModel
from gnn.har_graph import build_static_graph_30, build_snapshots_siso
from baselines.har_rv_baseline import fit_har, predict_har

# ─────────────────────────── CONFIG ───────────────────────────────────────────
HORIZONS              = [1, 5, 10, 20]
MAX_H                 = max(HORIZONS)
STRIDE_H              = {1: 5, 5: 5, 10: 10, 20: 20}   # T2-A: per-horizon stride, min=5
GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

# COVID exclusion (ablation): set True to drop March-Sept 2020 from train snapshots.
# HAR OLS still fits on full pre-2026 period -> conservative (GNN gets cleaner data).
# Set True only for ablation; main results use False.
EXCLUDE_COVID        = True
COVID_EXCLUDE_START  = "2020-03-01"
COVID_EXCLUDE_END    = "2020-09-30"

IN_CHANNELS  = 3
HIDDEN       = 16    # so neurons an; nho vi ESS thap (chi 149 snaps, 657 params)
# Dropout per-horizon: h=1 noisier (harder to fit) -> dung dropout cao hon de giam overfit.
# h=20 rat it overfit (val ~ train) -> dropout nho tranh regularize qua muc.
DROPOUT      = {1: 0.2, 5: 0.2, 10: 0.1, 20: 0.05}
EPOCHS       = 500
LR           = 1e-3
# weight_decay=1e-3: L2 regularization (phat cac trong so lon).
# Tuong duong voi Gaussian prior (mu=0, sigma^2=1/wd) tren cac tham so.
# Dung manh hon 1e-4 vi ty le obs/param = 149snaps*30stocks / 657params = 6.8 (rat thap).
WEIGHT_DECAY = 1e-3
PATIENCE     = 50    # dung som neu val loss khong cai thien sau 50 epochs lien tiep
SEED         = 42

N_STOCKS = len(VN30_TICKERS)   # 30

_root       = Path(__file__).parent.parent
CONFIG_PATH = _root / "config.yaml"
RESULTS_DIR = _root / "results"

# Khi EXCLUDE_COVID=True, luu ket qua vao file rieng de khong ghi de ket qua chinh.
_suffix    = "_no_covid" if EXCLUDE_COVID else ""
MODELS_DIR = _root / "models" / f"gnn_har{_suffix}"
CSV_OUT    = RESULTS_DIR / f"gnn_har_siso_results{_suffix}.csv"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)
DATA_DIR       = _root / _cfg["data"]["prices_dir"]
CORR_THRESHOLD = float(_cfg["model"]["corr_threshold"])


# ─────────────────────────── DROP EDGE (T1-B) ────────────────────────────────
DROP_EDGE_P = 0.2   # fraction of edges randomly dropped each training step

def drop_edges(g, p: float):
    """
    DropEdge (Rong et al. 2020): ngau nhien loai bo moi edge voi xac suat p.

    Tuong tu Dropout nhung hoat dong o muc do graph thay vi muc do neuron.
    Tac dung: (1) chinh quy hoa -> giam overfit; (2) aug data -> model robust hon
    khi mot so tuong quan stocks bi thay doi.
    Voi N=30 nodes va few edges, DropEdge=0.2 = giu lai 80% edges moi epoch.
    """
    if p <= 0.0:
        return g
    import dgl
    # Tao mask ngau nhien: True = giu edge, False = xoa edge
    mask  = torch.rand(g.num_edges()) > p
    src, dst = g.edges()
    return dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes())


# ─────────────────────────── METRICS ──────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    eps  = 1e-8
    ss_r = float(np.sum((y_true - y_pred) ** 2))
    ss_t = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = float(1.0 - ss_r / (ss_t + eps))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return dict(r2=r2, mae=mae, rmse=rmse)


# ─────────────────────────── HAR PREDICTIONS ──────────────────────────────────
def _har_pred_for_dates(
    rv_h_df: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex,
    train_end_ts: pd.Timestamp,
) -> np.ndarray:
    """
    Compute OLS-HAR predictions at each snapshot date for all 30 stocks.
    rv_h_df: pre-computed compute_rv(close_vn30, h=h).
    Returns (n_dates, 30) float32. Coefficients fit on data <= train_end_ts.
    """
    n   = len(snapshot_dates)
    out = np.zeros((n, N_STOCKS), dtype=np.float32)
    if n == 0:
        return out
    start_dt = snapshot_dates[0]
    for si, ticker in enumerate(VN30_TICKERS):
        if ticker not in rv_h_df.columns:
            continue
        try:
            rv_full    = rv_h_df[ticker].dropna()
            coeffs     = fit_har(rv_full, train_end_ts)
            har_series = predict_har(rv_full, coeffs, start_dt)
            aligned    = har_series.reindex(snapshot_dates).fillna(0.0)
            out[:, si] = aligned.values.astype(np.float32)
        except Exception:
            pass
    return out


# ─────────────────────────── SPLIT PRINT (R3) ─────────────────────────────────
def print_split(h, train_dates, val_dates, test_dates, n_train, n_val, n_test, stride):
    ess = n_train * N_STOCKS // h
    print(f"\n{'='*62}")
    print(f"  DATA SPLIT  h={h}  stride={stride}")
    print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()}"
          f"  ({n_train} snaps x {N_STOCKS} stocks)")
    print(f"  Val  : {val_dates[0].date()} -> {val_dates[-1].date()}"
          f"  ({n_val} snaps)")
    print(f"  Test : {test_dates[0].date()} -> {test_dates[-1].date()}"
          f"  ({n_test} snaps)" if n_test > 0 else "  Test : (none)")
    print(f"  ESS_h{h} = {n_train}x{N_STOCKS}/{h} = {ess}")
    print(f"{'='*62}\n")


# ─────────────────────────── LEARNING CURVE (R2) ──────────────────────────────
def save_learning_curve(train_hist: list, val_hist: list, h: int, path: Path):
    ep = range(1, len(train_hist) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, train_hist, label="Train", color="steelblue")
    ax.plot(ep, val_hist,   label="Val",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (z-scored HAR residual)")
    ax.set_title(f"GNN+HAR SISO  h={h}  stride={STRIDE_H[h]}  (DropEdge={DROP_EDGE_P}, wd={WEIGHT_DECAY})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Curve -> {path.name}")


# ─────────────────────────── MAIN ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import random as _random; _random.seed(SEED)
    import dgl as _dgl_seed; _dgl_seed.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  GNN+HAR SISO  |  VN30 30 stocks, no VNINDEX")
    print(f"  Horizons: {HORIZONS}  |  HIDDEN={HIDDEN}  DROPOUT={DROPOUT}")
    print(f"  EPOCHS={EPOCHS}  PATIENCE={PATIENCE}  STRIDE_H={STRIDE_H}")
    print(f"{'='*62}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    close_vn30   = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    log_ret_vn30 = compute_log_returns(close_vn30)
    print(f"[Data] {close_vn30.shape[0]} dates x {close_vn30.shape[1]} stocks")

    test_ts = pd.Timestamp(GLOBAL_TEST_START)

    # ── Build static graph once from MAX_H training period ─────────────────────
    # Use MAX_H snapshots to get train_end_ts (most conservative boundary).
    _X, _, _dates = build_snapshots_siso(
        close_vn30, log_ret_vn30, MAX_H, MAX_H,
        date_end=test_ts - pd.Timedelta(days=1),
    )
    _n_tr        = int(len(_dates) * TRAIN_VAL_SPLIT_RATIO)
    _train_end   = _dates[_n_tr - 1]
    del _X, _dates

    print(f"[Graph] Building 30-node static graph  train_end={_train_end.date()} ...")
    g = build_static_graph_30(log_ret_vn30, _train_end, CORR_THRESHOLD)
    print(f"  Nodes: {g.num_nodes()} | Edges: {g.num_edges()}\n")

    all_rows: list[dict] = []

    # ── SISO loop: one model per horizon ──────────────────────────────────────
    for h in HORIZONS:
        print(f"\n{'#'*62}")
        print(f"  HORIZON  h = {h}")
        print(f"{'#'*62}")

        # Pre-compute rv for this horizon (reused for HAR + metrics)
        rv_h_df = compute_rv(close_vn30, h=h)

        # Build snapshots
        X_pre, y_pre, dates_pre = build_snapshots_siso(
            close_vn30, log_ret_vn30, h, STRIDE_H[h],
            date_end=test_ts - pd.Timedelta(days=1),
        )
        n_pre   = len(dates_pre)
        n_train = int(n_pre * TRAIN_VAL_SPLIT_RATIO)
        n_val   = n_pre - n_train
        if n_train == 0:
            print(f"  [WARN] no train data for h={h}, skipping")
            continue

        train_dates  = dates_pre[:n_train]
        val_dates    = dates_pre[n_train:]
        X_train, y_train = X_pre[:n_train], y_pre[:n_train]
        X_val,   y_val   = X_pre[n_train:], y_pre[n_train:]
        train_end_ts = train_dates[-1]

        # COVID ablation: drop extreme-regime snapshots from train only
        if EXCLUDE_COVID:
            c_start = pd.Timestamp(COVID_EXCLUDE_START)
            c_end   = pd.Timestamp(COVID_EXCLUDE_END)
            mask    = ~((train_dates >= c_start) & (train_dates <= c_end))
            n_drop  = (~mask).sum()
            if n_drop > 0:
                print(f"  [COVID] dropping {n_drop} train snapshots "
                      f"({c_start.date()} to {c_end.date()})")
                X_train    = X_train[mask]
                y_train    = y_train[mask]
                train_dates = train_dates[mask]
            n_train = len(train_dates)

        X_test, y_test, test_dates = build_snapshots_siso(
            close_vn30, log_ret_vn30, h, stride=1,
            date_start=test_ts,
        )
        n_test = len(test_dates)

        # R3
        print_split(h, train_dates, val_dates, test_dates, n_train, n_val, n_test, STRIDE_H[h])

        # ── HAR Residual Training (Rule 8) ────────────────────────────────────
        # Thay vi du doan RV truc tiep, GNN du doan PHAN DU cua HAR:
        #   y_residual = y_actual - y_HAR
        # Neu GNN output = 0, final_pred = HAR -> dam bao GNN >= HAR (floor guarantee).
        # Hoc phan du de hon hoc gia tri tuyet doi vi:
        #   (1) Scale nho hon (phan du ~ 0 trung binh, bien do thap)
        #   (2) Model chi can sua sai so cua HAR, khong can tai hoc cau truc co ban
        print(f"  [HAR] fitting h={h} ...")
        har_train = _har_pred_for_dates(rv_h_df, train_dates, train_end_ts)
        har_val   = _har_pred_for_dates(rv_h_df, val_dates,   train_end_ts)
        har_test  = (_har_pred_for_dates(rv_h_df, test_dates, train_end_ts)
                     if n_test > 0 else np.zeros((0, N_STOCKS), np.float32))

        # y_res = y_actual - y_HAR: phan du can du doan. Shape: (n_snapshots, 30)
        y_res_train = (y_train - har_train).astype(np.float32)
        y_res_val   = (y_val   - har_val  ).astype(np.float32)

        # ── Z-score normalization (per stock, computed from train only) ────────
        # Tai sao phai z-score?
        # Cac stocks co muc vol rat khac nhau: GAS ~0.02/ngay, SSB ~0.006/ngay.
        # Neu train truc tiep tren raw values, loss bi thong tri boi cac stocks vol cao
        # (GNN tap trung sua sai so lon tuyet doi, bo qua stocks vol thap).
        # Z-score (mean=0, std=1 per stock) dam bao moi stock dong gop bang nhau vao loss.
        #
        # QUAN TRONG: chi dung train data de tinh mu/sigma; ap dung len val/test.
        # Dung val/test data de tinh se la "data leakage" (biet truoc phan phoi tuong lai).
        feat_mu  = X_train.mean(axis=0)     # (30, 3): trung binh features theo stock
        feat_sig = X_train.std(axis=0) + 1e-8  # +eps tranh chia cho 0
        rv_mu    = y_res_train.mean(axis=0)  # (30,): trung binh residual ~= 0
        rv_sig   = y_res_train.std(axis=0) + 1e-8

        def norm_x(X: np.ndarray) -> np.ndarray:
            # z-score features: (X - mean) / std, shape giu nguyen (n, 30, 3)
            return ((X - feat_mu[np.newaxis]) / feat_sig[np.newaxis]).astype(np.float32)

        def norm_y(yr: np.ndarray) -> np.ndarray:
            # z-score residuals: (residual - mean_residual) / std_residual
            return ((yr - rv_mu[np.newaxis]) / rv_sig[np.newaxis]).astype(np.float32)

        X_train_n = norm_x(X_train)
        y_train_n = norm_y(y_res_train)    # (n_tr, 30)
        X_val_n   = norm_x(X_val)
        y_val_n   = norm_y(y_res_val)
        X_test_n  = (norm_x(X_test) if n_test > 0
                     else np.empty((0, N_STOCKS, 3), np.float32))

        # ── Model + optimizer ─────────────────────────────────────────────────
        h_dropout = DROPOUT[h]   # T1-C: per-horizon dropout
        model = GNNHARModel(IN_CHANNELS, HIDDEN, h_dropout)
        # AdamW = Adam + weight decay (L2 regularization).
        # Adam: adaptive learning rate per parameter (momentum + RMSProp).
        # weight_decay: them L2 penalty truc tiep vao tham so (khac Adam co ban
        # where decay bi nhiem vao gradient update).
        opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # MSE tren z-scored residuals: moi stock dong gop bang nhau (sau z-score).
        crit  = nn.MSELoss()
        print(f"  Model params: {model.count_params()}  dropout={h_dropout}  "
              f"weight_decay={WEIGHT_DECAY}  drop_edge_p={DROP_EDGE_P}")

        Xv = [torch.from_numpy(X_val_n[t]) for t in range(n_val)]
        yv = [torch.from_numpy(y_val_n[t]) for t in range(n_val)]

        train_hist, val_hist                           = [], []
        best_val, best_ep, best_sd, patience_cnt = float("inf"), 0, None, 0

        print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Val':>8}")
        print(f"  {'-'*28}")

        for ep in range(1, EPOCHS + 1):
            model.train()
            perm         = np.random.permutation(n_train)
            dropped_gs   = [drop_edges(g, DROP_EDGE_P) for _ in range(n_train)]
            ep_loss = 0.0
            for i, t_idx in enumerate(perm):
                x_t   = torch.from_numpy(X_train_n[t_idx])   # (30, 3)
                y_t   = torch.from_numpy(y_train_n[t_idx])   # (30,)
                g_t   = dropped_gs[i]                         # T1-B: stochastic graph
                pred  = model(g_t, x_t)                   # (30,)
                loss = crit(pred, y_t)
                opt.zero_grad()
                loss.backward()
                # Gradient clipping: neu L2-norm cua tat ca gradients > 1.0, scale chung
                # xuong. Ngan gradient explosion (gradients tang exponentially qua cac
                # layer), dac biet quan trong trong nhung epoch dau khi model chua on dinh.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                ep_loss += loss.item()
            ep_loss /= n_train

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for t_idx in range(n_val):
                    va_loss += crit(model(g, Xv[t_idx]), yv[t_idx]).item()
            va_loss /= n_val

            train_hist.append(ep_loss)
            val_hist.append(va_loss)

            # R2: print every epoch
            print(f"  Epoch {ep:3d}/{EPOCHS} | Train: {ep_loss:.4f} | Val: {va_loss:.4f}")

            if va_loss < best_val:
                best_val, best_ep = va_loss, ep
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  [early stop] ep={ep}  best val={best_val:.4f} @ ep {best_ep}")
                    break

        if best_sd is not None:
            model.load_state_dict(best_sd)
        else:
            print("[WARN] best_sd is None; model weights unchanged")

        # R2: learning curve
        save_learning_curve(
            train_hist, val_hist, h,
            RESULTS_DIR / f"gnn_har_curves_h{h}{_suffix}.png",
        )

        # Save model checkpoint
        torch.save({
            "state_dict":    model.state_dict(),
            "h":             h,
            "in_channels":   IN_CHANNELS,
            "hidden":        HIDDEN,
            "dropout":       DROPOUT,
            "feat_mu":       feat_mu,
            "feat_sig":      feat_sig,
            "rv_mu":         rv_mu,
            "rv_sig":        rv_sig,
            "stock_order":   VN30_TICKERS,
            "best_epoch":    best_ep,
            "best_val_loss": best_val,
            "train_end_ts":  str(train_end_ts.date()),
        }, MODELS_DIR / f"model_h{h}.pt")
        print(f"  Model -> models/gnn_har/model_h{h}.pt")

        # ── Test evaluation ───────────────────────────────────────────────────
        if n_test == 0:
            print(f"  [WARN] no test snapshots for h={h}")
            continue

        model.eval()
        pred_norm = np.zeros((n_test, N_STOCKS), np.float32)
        with torch.no_grad():
            for t_idx in range(n_test):
                pred_norm[t_idx] = model(g, torch.tensor(X_test_n[t_idx])).numpy()

        # Buoc 1: undo z-score tren residual (dua ve don vi goc)
        # gnn_res[t,i] = pred_norm[t,i] * rv_sig[i] + rv_mu[i]
        gnn_res = pred_norm * rv_sig[np.newaxis] + rv_mu[np.newaxis]
        # Buoc 2: cong voi HAR prediction de duoc RV du doan cuoi
        # Buoc 3: clip >= 0 vi RV la do lenh chuan (std), luon >= 0.
        # Neu GNN du doan am thi clip ve 0 (tuong duong noi "khong co volatility").
        final_pred = np.clip(har_test + gnn_res, 0.0, None)   # (n_test, 30)

        # ── Per-stock metrics ─────────────────────────────────────────────────
        print(f"\n  Per-stock results  h={h}  ({n_test} test snapshots):")
        print(f"  {'Ticker':<8} {'GNN_R2':>8} {'HAR_R2':>8} {'GNN_MAE':>9} "
              f"{'HAR_MAE':>9} {'GNN_RMSE':>10} {'delta_R2':>9}")
        print(f"  {'-'*70}")

        # Pre-compute HAR coefficients once per ticker (reused in per-stock loop)
        har_coeffs_cache = {}
        for ticker in VN30_TICKERS:
            if ticker in rv_h_df.columns:
                try:
                    har_coeffs_cache[ticker] = fit_har(rv_h_df[ticker].dropna(), train_end_ts)
                except Exception:
                    pass

        for si, ticker in enumerate(VN30_TICKERS):
            row = {"ticker": ticker, "h": h, "n_test": n_test, "best_ep": best_ep}

            # Ground-truth from rv_h_df (more accurate than y_test fill)
            y_true = rv_h_df[ticker].reindex(test_dates).values if ticker in rv_h_df.columns else None
            if y_true is not None:
                valid = ~np.isnan(y_true)
                if valid.sum() >= 2:
                    m = compute_metrics(y_true[valid], final_pred[valid, si])
                    for k, v in m.items():
                        row[f"gnn_{k}"] = v

            # HAR baseline metrics (use cached coefficients)
            try:
                rv_full    = rv_h_df[ticker].dropna()
                har_coeffs = har_coeffs_cache.get(ticker)
                if har_coeffs is None:
                    raise ValueError("no cached coeffs")
                har_preds  = predict_har(rv_full, har_coeffs, test_ts)
                common     = test_dates.intersection(har_preds.index)
                if len(common) >= 2:
                    yt = rv_full.reindex(common).values
                    yp = har_preds.reindex(common).values
                    vm = ~(np.isnan(yt) | np.isnan(yp))
                    if vm.sum() >= 2:
                        m_har = compute_metrics(yt[vm], yp[vm])
                        for k, v in m_har.items():
                            row[f"har_{k}"] = v
            except Exception:
                pass

            row["delta_r2"] = row.get("gnn_r2", float("nan")) - row.get("har_r2", float("nan"))
            all_rows.append(row)

            gr2  = row.get("gnn_r2",   float("nan"))
            hr2  = row.get("har_r2",   float("nan"))
            gmae = row.get("gnn_mae",  float("nan"))
            hmae = row.get("har_mae",  float("nan"))
            grms = row.get("gnn_rmse", float("nan"))
            d    = row.get("delta_r2", float("nan"))
            flag = "[+]" if d > 0.01 else ("[-]" if d < -0.01 else "[~]")
            print(f"  {ticker:<8} {gr2:>8.4f} {hr2:>8.4f} {gmae:>9.5f} "
                  f"{hmae:>9.5f} {grms:>10.5f} {d:>+9.4f} {flag}")

        # Horizon summary
        h_rows = [r for r in all_rows if r.get("h") == h]
        if h_rows:
            avg_gr2 = float(np.nanmean([r.get("gnn_r2", float("nan")) for r in h_rows]))
            avg_hr2 = float(np.nanmean([r.get("har_r2", float("nan")) for r in h_rows]))
            n_beat  = sum(1 for r in h_rows if (r.get("delta_r2") or 0.0) > 0)
            print(f"\n  h={h} avg:  GNN R2={avg_gr2:+.4f}  HAR R2={avg_hr2:+.4f}"
                  f"  delta={avg_gr2-avg_hr2:+.4f}  GNN>HAR: {n_beat}/{N_STOCKS} stocks")

    # ── Save combined CSV ──────────────────────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n  Results -> {CSV_OUT}  ({len(df)} rows)")

        # Final cross-horizon summary (no averaging across h)
        print(f"\n{'='*62}")
        print(f"  FINAL SUMMARY -- avg R2 per horizon (separately)")
        print(f"{'='*62}")
        print(f"  {'h':>4} | {'GNN R2':>8} | {'HAR R2':>8} | {'delta':>8} | {'GNN>HAR':>8}")
        print(f"  {'-'*52}")
        for h in HORIZONS:
            sub = df[df.h == h]
            if sub.empty:
                continue
            g_r2 = sub["gnn_r2"].mean() if "gnn_r2" in sub.columns else float("nan")
            h_r2 = sub["har_r2"].mean() if "har_r2" in sub.columns else float("nan")
            d    = g_r2 - h_r2
            nb   = int((sub["delta_r2"] > 0).sum()) if "delta_r2" in sub.columns else 0
            print(f"  h={h:>2} | {g_r2:>+8.4f} | {h_r2:>+8.4f} | {d:>+8.4f} | {nb:>4}/{N_STOCKS}")

    print(f"\n{'='*62}")
    print(f"  DONE  |  Models -> models/gnn_har/")
    print(f"        |  Curves -> results/gnn_har_curves_h*.png")
    print(f"        |  CSV    -> {CSV_OUT.name}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
