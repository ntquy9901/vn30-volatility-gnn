"""
HAR snapshot builder and static graph constructor for GNN+HAR.

Graph: 1 static graph from train period (Pearson on full train log_returns).
Features: same type as HAR baseline -- rolling std (no leakage, ends at t).
  rv_d[t] = |log_return[t]|                   -- daily abs return (h=1 proxy)
  rv_w[t] = std(log_return[t-4..t], ddof=1)   -- 5-day rolling std
  rv_m[t] = std(log_return[t-19..t], ddof=1)  -- 20-day rolling std
  Target  = std(log_return[t+1..t+H])         -- starts at t+1 -> no leakage

Feature type now matches target type (both rolling std), same as HAR features.
HAR uses rv.shift(1).rolling(k).mean(); GNN uses compute_past_rv directly.

Usage:
  from gnn.har_graph import build_static_graph, build_snapshots
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.build_graph import build_graph, GraphData, ALL_NODES, VN30_TICKERS
from src.volatility_labels import compute_rv, compute_log_returns, compute_past_rv

warnings.filterwarnings("ignore")

N_NODES    = 31   # VNINDEX + 30 VN30
N_FEATURES = 3    # [rv_d, rv_w, rv_m]


def build_static_graph(
    log_returns: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    corr_threshold: float = 0.4,
) -> GraphData:
    """
    Build 1 static DGL graph using Pearson over the entire train period.
    corr_window set to full train length so all train data is used.
    """
    train_lr   = log_returns[log_returns.index <= train_end_ts]
    corr_window = len(train_lr)
    return build_graph(
        log_returns,
        end_date=train_end_ts,
        corr_window=corr_window,
        corr_threshold=corr_threshold,
    )


def _compute_har_features(log_returns: pd.DataFrame) -> tuple:
    """
    Returns (rv_d, rv_w, rv_m) DataFrames indexed same as log_returns.
    Features use rolling std -- same type as RV targets (fixes feature-target mismatch).
    rv_d = |log_return[t]|           -- daily abs return, h=1 proxy (rolling(1).std ddof=1 is NaN)
    rv_w = compute_past_rv(lr, h=5)  -- 5-day rolling std, ddof=1
    rv_m = compute_past_rv(lr, h=20) -- 20-day rolling std, ddof=1
    No lookahead: all features end at t, target starts at t+1.
    """
    rv_d = log_returns.abs()
    rv_w = compute_past_rv(log_returns, h=5)
    rv_m = compute_past_rv(log_returns, h=20)
    return rv_d, rv_w, rv_m


def build_snapshots(
    close: pd.DataFrame,
    log_returns: pd.DataFrame,
    horizons: list,
    stride: int,
    date_start: pd.Timestamp | None = None,
    date_end:   pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build GNN snapshots for a date range.

    Returns:
      X     : (n_snaps, 31, 3)  float32 -- node features per snapshot
      y     : (n_snaps, 30, 4)  float32 -- RV targets per stock per horizon
      dates : DatetimeIndex of length n_snaps
    """
    rv_d, rv_w, rv_m = _compute_har_features(log_returns)
    max_h = max(horizons)

    # Build target DataFrames for each horizon
    rv_targets = {}
    for h in horizons:
        rv_targets[h] = compute_rv(close, h=h)

    # Find valid dates: rv_m non-NaN for first VN30 stock AND rv_h_max non-NaN
    anchor = VN30_TICKERS[0]
    feat_ok = rv_m[anchor].notna() if anchor in rv_m.columns else pd.Series(True, index=log_returns.index)
    tgt_ok  = rv_targets[max_h][anchor].notna() if anchor in rv_targets[max_h].columns else pd.Series(True, index=close.index)

    valid_mask = feat_ok & tgt_ok
    if date_start is not None:
        valid_mask = valid_mask & (log_returns.index >= date_start)
    if date_end is not None:
        valid_mask = valid_mask & (log_returns.index <= date_end)

    valid_dates_all = log_returns.index[valid_mask]
    # Subsample by stride
    snapshot_dates = valid_dates_all[::stride]

    n_snaps  = len(snapshot_dates)
    n_stocks = len(VN30_TICKERS)
    n_horiz  = len(horizons)

    X = np.zeros((n_snaps, N_NODES, N_FEATURES), dtype=np.float32)
    y = np.zeros((n_snaps, n_stocks, n_horiz),   dtype=np.float32)

    for i, date in enumerate(snapshot_dates):
        # Node features (31 nodes)
        for j, node in enumerate(ALL_NODES):
            if node not in log_returns.columns:
                continue
            try:
                X[i, j, 0] = float(rv_d.loc[date, node]) if not pd.isna(rv_d.loc[date, node]) else 0.0
                X[i, j, 1] = float(rv_w.loc[date, node]) if not pd.isna(rv_w.loc[date, node]) else 0.0
                X[i, j, 2] = float(rv_m.loc[date, node]) if not pd.isna(rv_m.loc[date, node]) else 0.0
            except (KeyError, ValueError):
                pass

        # Targets (30 VN30 stocks)
        for k, h in enumerate(horizons):
            for si, ticker in enumerate(VN30_TICKERS):
                if ticker not in rv_targets[h].columns:
                    continue
                try:
                    val = rv_targets[h].loc[date, ticker]
                    y[i, si, k] = float(val) if not pd.isna(val) else 0.0
                except (KeyError, ValueError):
                    pass

    return X, y, snapshot_dates
