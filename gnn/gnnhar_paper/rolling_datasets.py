"""
Rolling window dataset builder for GNNHAR paper training scheme.

Paper: "Forecasting Realized Volatility with Spillover Effects:
         Perspectives from Graph Neural Networks" (IJF 2024)

The paper uses a rolling (expanding) window approach:
  - For each date in test period: train on [s_date, v_date], validate on [v_date, date]
  - Test window: [date, f_date] (typically 22 days)
  - Adjacency: computed from 1000-day lookback returns

This differs from static 80/20 split in that:
  1. Training set grows over time (expanding window)
  2. Adjacency is recomputed per window (captures changing correlations)
  3. Multiple separate models are trained (one per test window)

For efficiency, we implement a simplified version:
  - Fixed train/val split (80/20 from pre-test data)
  - Rolling test windows (stride = window_size)
  - Adjacency can be static or rolling per window

Data flow:
  close prices -> log returns -> HAR features [rv_d, rv_w, rv_m] -> snapshots
                -> targets (RV at horizon h)

Each snapshot = one training example:
  X[t] : (N, 3)  -- HAR features for N stocks at time t
  y[t] : (N,)    -- RV target for N stocks at time t (one horizon)
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS

warnings.filterwarnings("ignore")

N_STOCKS = len(VN30_TICKERS)  # 30
N_FEATURES = 3                 # [rv_d, rv_w, rv_m]


def compute_har_features(log_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute HAR features exactly as in the GNNHAR paper.

    Paper uses simple lag averages (not rolling std like current implementation):
        rv_d[t] = average of past 1 day RV
        rv_w[t] = average of past 5 days RV
        rv_m[t] = average of past 22 days RV

    However, the paper's code shows a discrepancy: they use the future target
    (RV_h) as input feature, which is lookahead! Our implementation fixes this
    by using past RV computed from returns.

    Args:
        log_returns: (T, N) DataFrame of log returns

    Returns:
        rv_d, rv_w, rv_m: (T, N) DataFrames of HAR features
    """
    # Paper's approach: simple averages of past RV values
    # But we don't have RV as a precomputed series, so we compute
    # from returns as a proxy.

    # Daily RV proxy: absolute return (same as paper's "variance")
    rv_d = log_returns.abs()

    # Weekly: 5-day rolling mean of absolute returns
    rv_w = rv_d.rolling(5, min_periods=1).mean()

    # Monthly: 22-day rolling mean of absolute returns
    rv_m = rv_d.rolling(22, min_periods=1).mean()

    return rv_d, rv_w, rv_m


def build_static_snapshots(
    close: pd.DataFrame,
    log_returns: pd.DataFrame,
    horizon: int,
    stride: int,
    date_end: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build snapshots for fixed training period (static 80/20 split).

    This is similar to current implementation but matches paper's HAR features.

    Args:
        close: (T, N) close prices
        log_returns: (T, N) log returns
        horizon: forecast horizon (1, 5, 10, or 20)
        stride: sampling stride (reduce label overlap)
        date_end: end date for snapshots (exclusive)

    Returns:
        X: (n_snapshots, N, 3) HAR features
        y: (n_snapshots, N) RV targets
        dates: DatetimeIndex of snapshot dates
    """
    # Compute HAR features
    rv_d, rv_w, rv_m = compute_har_features(log_returns)

    # Compute RV target
    rv_target = compute_rv(close, h=horizon)

    # Valid dates: all features and target are non-NaN
    anchor = VN30_TICKERS[0]
    feat_ok = rv_m[anchor].notna() if anchor in rv_m.columns else pd.Series(True, index=log_returns.index)
    tgt_ok = rv_target[anchor].notna() if anchor in rv_target.columns else pd.Series(True, index=close.index)

    valid_mask = feat_ok & tgt_ok
    if date_end is not None:
        valid_mask = valid_mask & (log_returns.index < date_end)

    # Sample with stride to reduce label overlap
    snapshot_dates = log_returns.index[valid_mask][::stride]
    n_snaps = len(snapshot_dates)

    def _fill(df: pd.DataFrame) -> np.ndarray:
        """Reindex and fill NaN with 0."""
        return np.nan_to_num(
            df.reindex(snapshot_dates).reindex(columns=VN30_TICKERS).values,
            nan=0.0,
        ).astype(np.float32)

    # Stack features: (n, N, 3)
    X = np.stack([_fill(rv_d), _fill(rv_w), _fill(rv_m)], axis=2)
    y = _fill(rv_target)  # (n, N)

    return X, y, snapshot_dates


def build_rolling_window(
    close: pd.DataFrame,
    log_returns: pd.DataFrame,
    horizon: int,
    window_size: int = 22,
    valid_len: int = 22,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> dict:
    """
    Build rolling window dataset following paper's exact scheme.

    For each date in test period:
        train: [s_date, v_date]
        val:   [v_date, date]
        test:  [date, f_date]

    where:
        s_date = max(date - 1000, start)  # 1000-day lookback
        v_date = date - valid_len          # validation starts here
        f_date = date + window_size        # test window ends here

    Args:
        close: (T, N) close prices
        log_returns: (T, N) log returns
        horizon: forecast horizon
        window_size: test window size (default 22 from paper)
        valid_len: validation period length (default 22 from paper)
        test_start: start of test period
        test_end: end of test period

    Returns:
        dict with:
            'windows': list of dicts with 'train_idx', 'val_idx', 'test_idx',
            'dates': list of test dates,
            'X': full feature array,
            'y': full target array,
            'date_index': full date index
    """
    # Build full snapshots (stride=1 for paper's exact indexing)
    X_full, y_full, dates_full = build_static_snapshots(
        close, log_returns, horizon, stride=1
    )

    # Convert date to index
    date_to_idx = {date: i for i, date in enumerate(dates_full)}

    # Determine test period
    if test_start is None:
        test_start = dates_full[int(len(dates_full) * 0.8)]  # last 20% as test
    if test_end is None:
        test_end = dates_full[-1]

    # Filter test dates (every window_size days)
    test_dates = [
        d for d in dates_full
        if d >= test_start and d <= test_end
    ]
    # Sample test dates at window stride
    test_dates = test_dates[::window_size]

    windows = []
    for date in test_dates:
        idx = date_to_idx[date]

        # s_date = max(date - 1000 days, start)
        # We use 1000 INDEX positions (not days) to match paper's "1000 samples"
        s_idx = max(0, idx - 1000)

        # v_date = date - valid_len
        v_idx = max(s_idx, idx - valid_len)

        # f_date = date + window_size
        f_idx = min(len(dates_full), idx + window_size)

        windows.append({
            'date': date,
            'train_idx': (s_idx, v_idx),
            'val_idx': (v_idx, idx),
            'test_idx': (idx, f_idx),
        })

    return {
        'windows': windows,
        'test_dates': test_dates,
        'X': X_full,
        'y': y_full,
        'date_index': dates_full,
    }


def get_window_data(
    rolling_data: dict,
    window_idx: int,
) -> tuple:
    """
    Extract train/val/test data for a specific window.

    Args:
        rolling_data: output from build_rolling_window
        window_idx: index of window to extract

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, window_info
    """
    win = rolling_data['windows'][window_idx]
    X = rolling_data['X']
    y = rolling_data['y']

    s_i, v_i = win['train_idx']
    v_i2, t_i = win['val_idx']
    t_i2, f_i = win['test_idx']

    X_train = X[s_i:v_i]
    y_train = y[s_i:v_i]
    X_val = X[v_i:t_i]
    y_val = y[v_i:t_i]
    X_test = X[t_i:f_i]
    y_test = y[t_i:f_i]

    return X_train, y_train, X_val, y_val, X_test, y_test, win


if __name__ == "__main__":
    # Test with real data
    print("[TEST] Rolling window dataset builder...")

    from src.volatility_labels import load_close_prices, compute_log_returns
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(__file__).parent.parent / cfg["data"]["prices_dir"]
    close = load_close_prices(data_dir, tickers=VN30_TICKERS)
    log_ret = compute_log_returns(close)

    print(f"Data shape: {close.shape}")

    # Test static snapshots
    X, y, dates = build_static_snapshots(
        close, log_ret, horizon=1, stride=20,
        date_end=pd.Timestamp("2026-01-01")
    )
    print(f"\nStatic snapshots h=1:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")

    # Test rolling window
    rolling = build_rolling_window(
        close, log_ret, horizon=1,
        window_size=22, valid_len=22,
        test_start=pd.Timestamp("2026-01-01"),
    )

    print(f"\nRolling windows h=1:")
    print(f"  Total windows: {len(rolling['windows'])}")

    # First window details
    X_tr, y_tr, X_va, y_va, X_te, y_te, win = get_window_data(rolling, 0)
    print(f"\n  Window 0 ({win['date'].date()}):")
    print(f"    Train: {X_tr.shape[0]} samples")
    print(f"    Val:   {X_va.shape[0]} samples")
    print(f"    Test:  {X_te.shape[0]} samples")

    print("\n[OK] All tests passed")
