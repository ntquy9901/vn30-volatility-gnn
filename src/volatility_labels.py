"""
Compute realized volatility labels for VN30 stocks.

RV_t(h) = std(log-returns from t+1 to t+h, ddof=1)

Usage:
    python src/volatility_labels.py
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def load_close_prices(prices_dir: str | Path, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Load close prices from per-ticker CSV files.

    Args:
        prices_dir: Directory containing *_ohlcv.csv files.
        tickers: Subset of tickers to load. None = all files found.

    Returns:
        DataFrame shape (n_dates, n_tickers), DatetimeIndex, sorted ascending.
    """
    prices_dir = Path(prices_dir)
    csvs = sorted(prices_dir.glob("*_ohlcv.csv"))

    if tickers is not None:
        ticker_set = set(tickers)
        csvs = [f for f in csvs if f.stem.replace("_ohlcv", "") in ticker_set]

    frames: dict[str, pd.Series] = {}
    for csv in csvs:
        ticker = csv.stem.replace("_ohlcv", "")
        df = pd.read_csv(csv, parse_dates=["date"], index_col="date")
        frames[ticker] = df["close"]

    prices = pd.DataFrame(frames).sort_index()
    return prices


def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    log-return r_t = log(P_t / P_{t-1}). First row is NaN.
    """
    return np.log(close / close.shift(1))


def compute_rv(close: pd.DataFrame, h: int = 20) -> pd.DataFrame:
    """
    Realized volatility: RV_t(h) = std(r_{t+1}, ..., r_{t+h}, ddof=1).

    Implementation:
        rolling(h).std() at index t = std of window [t-h+1, t]
        .shift(-h)        at index t = that value from index t+h
                                     = std of window [t+1, t+h]  ✓

    Returns NaN for the last h rows (no sufficient future data).
    NaN for the first row of log-returns (no prior price).

    Args:
        close: Close-price DataFrame (n_dates, n_stocks).
        h: Forecast horizon in trading days (default 20 ≈ 1 month HOSE).

    Returns:
        RV DataFrame, same shape as close.
    """
    log_ret = compute_log_returns(close)
    if h == 1:
        # std(1 value, ddof=1) is undefined; |r_{t+1}| is the standard h=1 proxy
        return log_ret.abs().shift(-1)
    rv = log_ret.rolling(h, min_periods=h).std(ddof=1).shift(-h)
    return rv


def compute_past_rv(log_returns: pd.DataFrame, h: int = 20) -> pd.DataFrame:
    """
    Past RV at date t = std(log-returns[t-h+1 .. t], ddof=1).
    Fully available at time t — no lookahead.
    """
    return log_returns.rolling(h, min_periods=h).std(ddof=1)


def get_rv_node_features(
    log_returns: pd.DataFrame,
    window_end: pd.Timestamp,
    all_nodes: list,
    h: int = 20,
) -> torch.Tensor:
    """
    Lagged RV node features [log(RV_d), log(RV_w), log(RV_m)] for each node.

    RV_d = past-h RV at window_end          (daily)
    RV_w = mean of past 5  RV values        (weekly)
    RV_m = mean of past 20 RV values        (monthly)

    Log-transformed so scale matches log-RV training target.

    Returns:
        Tensor shape (len(all_nodes), 3), float32.
    """
    rv_past = compute_past_rv(log_returns, h)
    n = len(all_nodes)
    feat = np.zeros((n, 3), dtype=np.float32)

    for idx, ticker in enumerate(all_nodes):
        if ticker not in rv_past.columns:
            continue
        series = rv_past[ticker]
        avail  = series[series.index <= window_end].dropna()
        if len(avail) == 0:
            continue
        rv_d = float(avail.iloc[-1])
        rv_w = float(avail.iloc[-5:].mean())  if len(avail) >= 5  else rv_d
        rv_m = float(avail.iloc[-20:].mean()) if len(avail) >= 20 else rv_d

        feat[idx, 0] = np.log(max(rv_d, 1e-8))
        feat[idx, 1] = np.log(max(rv_w, 1e-8))
        feat[idx, 2] = np.log(max(rv_m, 1e-8))

    return torch.tensor(feat, dtype=torch.float)


def get_extended_rv_features(
    log_returns: pd.DataFrame,
    window_end: pd.Timestamp,
    all_nodes: list,
    h: int = 20,
) -> torch.Tensor:
    """
    Extended RV node features — 6 dimensions per node:
      [0] log(RV_d)        — past-h RV (daily)
      [1] log(RV_w)        — 5-day avg RV (weekly)
      [2] log(RV_m)        — 20-day avg RV (monthly)
      [3] log(RV_q)        — 60-day avg RV (quarterly)
      [4] corr_vnindex     — rolling 60-day Pearson corr with VNINDEX returns
      [5] jump_ratio       — max(RV_d - RV_w, 0) / RV_d, proportion from jumps

    Returns:
        Tensor shape (len(all_nodes), 6), float32.
    """
    rv_past = compute_past_rv(log_returns, h)
    n       = len(all_nodes)
    feat    = np.zeros((n, 6), dtype=np.float32)

    # Pre-compute VNINDEX return series once
    vi_col = log_returns["VNINDEX"] if "VNINDEX" in log_returns.columns else None

    for idx, ticker in enumerate(all_nodes):
        if ticker not in rv_past.columns:
            continue
        series = rv_past[ticker]
        avail  = series[series.index <= window_end].dropna()
        if len(avail) == 0:
            continue

        rv_d = float(avail.iloc[-1])
        rv_w = float(avail.iloc[-5:].mean())  if len(avail) >= 5  else rv_d
        rv_m = float(avail.iloc[-20:].mean()) if len(avail) >= 20 else rv_d
        rv_q = float(avail.iloc[-60:].mean()) if len(avail) >= 60 else rv_m

        # Jump ratio: fraction of daily RV unexplained by recent average
        jump_ratio = max(rv_d - rv_w, 0.0) / max(rv_d, 1e-8)

        # Correlation with VNINDEX (systematic risk exposure)
        corr_vi = 0.0
        if vi_col is not None and ticker in log_returns.columns:
            df_pair = pd.concat([log_returns[ticker], vi_col], axis=1).dropna()
            df_pair = df_pair[df_pair.index <= window_end].iloc[-60:]
            if len(df_pair) >= 10:
                c = float(df_pair.iloc[:, 0].corr(df_pair.iloc[:, 1]))
                corr_vi = 0.0 if np.isnan(c) else c

        feat[idx, 0] = np.log(max(rv_d, 1e-8))
        feat[idx, 1] = np.log(max(rv_w, 1e-8))
        feat[idx, 2] = np.log(max(rv_m, 1e-8))
        feat[idx, 3] = np.log(max(rv_q, 1e-8))
        feat[idx, 4] = corr_vi
        feat[idx, 5] = jump_ratio

    return torch.tensor(feat, dtype=torch.float)


# ── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")

    PRICES_DIR = Path("data/raw/prices")
    H = 20

    VN30_TICKERS = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SHB", "SSB",
        "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM",
    ]

    print("Loading close prices (30 VN30 stocks)...")
    close = load_close_prices(PRICES_DIR, tickers=VN30_TICKERS)
    print(f"  Shape: {close.shape}  |  Date range: {close.index[0].date()} -> {close.index[-1].date()}")

    # Check for missing tickers
    missing = [t for t in VN30_TICKERS if t not in close.columns]
    if missing:
        print(f"  WARNING: Missing tickers: {missing}")
    else:
        print(f"  All 30 tickers loaded OK")

    # NaN audit
    nan_frac = close.isna().mean()
    max_nan = nan_frac.max()
    print(f"  Max NaN fraction in any ticker: {max_nan:.4f}")
    if max_nan > 0.05:
        print(f"  WARNING: High NaN fraction in: {nan_frac[nan_frac > 0.05].index.tolist()}")

    print(f"\nComputing log-returns...")
    log_ret = compute_log_returns(close)
    extreme = (log_ret.abs() > 0.15).sum().sum()
    print(f"  Log-returns outside +-15%: {extreme} cells")

    print(f"\nComputing RV labels (h={H})...")
    rv = compute_rv(close, h=H)
    print(f"  RV shape: {rv.shape}")

    # Sample: VCB
    vcb_rv = rv["VCB"].dropna()
    print(f"\nVCB RV(20) stats (non-NaN):")
    print(f"  Count : {len(vcb_rv)}")
    print(f"  Mean  : {vcb_rv.mean():.6f}  ({vcb_rv.mean()*100:.4f}%/day)")
    print(f"  Std   : {vcb_rv.std():.6f}")
    print(f"  Min   : {vcb_rv.min():.6f}")
    print(f"  Max   : {vcb_rv.max():.6f}")
    print(f"  Annualized mean: {vcb_rv.mean() * (252**0.5) * 100:.2f}%/year")

    # Train/test split check
    train_end = pd.Timestamp("2025-12-31")
    test_start = pd.Timestamp("2026-01-01")
    rv_train = rv[rv.index <= train_end]
    rv_test  = rv[rv.index >= test_start]
    print(f"\nTrain RV windows: {len(rv_train.dropna(how='all'))}")
    print(f"Test  RV windows: {len(rv_test.dropna(how='all'))}")
    print(f"  Test date range: {rv_test.dropna(how='all').index[0].date()} -> {rv_test.dropna(how='all').index[-1].date()}")

    # Verify no look-ahead: label at train_end uses returns up to train_end + h
    last_train_label_date = rv_train.dropna(how="all").index[-1]
    print(f"\nLast date with valid train label: {last_train_label_date.date()}")
    print(f"  (uses returns up to approx {last_train_label_date + pd.offsets.BDay(H)})")

    print("\nvolatility_labels.py OK.")
