"""
GARCH(1,1) baseline: fit per stock on train data, predict RV on test window.

For each stock, fits GARCH(1,1) on log-returns up to train_end, then
produces rolling 1-step-ahead conditional volatility forecasts for the
test period by updating the variance filter recursively (no re-fit).

Predicted RV at date t = sqrt of conditional variance h_t, which proxies
the expected absolute daily move.  Compared against realized std(h=20).

Usage:
    python baselines/garch_baseline.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS


def fit_garch_predictions(
    log_ret: pd.Series,
    rv_target: pd.Series,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.Series:
    """
    Fit GARCH(1,1) once on training data, then produce rolling sigma forecasts
    for the test period using fixed parameters (no re-fit).

    Returns:
        pd.Series of predicted conditional sigma (proxy for RV), indexed by
        test dates where rv_target is defined.
    """
    from arch import arch_model

    scale = 100.0  # % scale for numerical stability
    ret_all = log_ret.dropna() * scale

    train_ret = ret_all[ret_all.index <= train_end]
    if len(train_ret) < 50:
        return pd.Series(dtype=float, name=log_ret.name)

    # Fit once on training data
    am = arch_model(train_ret, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)

    # Rolling forecast from start of test period using fixed params
    # arch supports this via forecast(start=...) after fitting on full series up to train_end
    # We use the full series through test so the variance filter can update, but params are fixed
    ret_full = ret_all.copy()

    am_full = arch_model(ret_full, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fix parameters to training estimates and compute variance path
        params = res.params
        resid  = ret_full.values
        omega  = params["omega"]
        alpha1 = params["alpha[1]"]
        beta1  = params["beta[1]"]

    # Compute recursive conditional variance over entire series
    n      = len(resid)
    h      = np.zeros(n)
    h[0]   = np.var(resid[:50])  # warm-up using first 50 obs
    for t in range(1, n):
        h[t] = omega + alpha1 * resid[t - 1] ** 2 + beta1 * h[t - 1]

    sigma_series = pd.Series(
        np.sqrt(np.maximum(h, 1e-10)) / scale,
        index=ret_full.index,
        name=log_ret.name,
    )

    # Keep only test dates that have valid rv_target
    test_dates = rv_target.index[
        (rv_target.index >= test_start) & (~rv_target.isna())
    ]
    return sigma_series.reindex(test_dates).dropna()


def run_garch_baseline(
    prices_dir: str,
    train_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    horizon: int = 20,
    tickers: list[str] | None = None,
) -> dict[str, pd.Series]:
    """
    Run GARCH(1,1) baseline for all VN30 stocks.

    Returns:
        dict mapping ticker -> pd.Series of predicted sigma for test period.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    train_end_ts  = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    close   = load_close_prices(prices_dir, tickers=tickers)
    log_ret = compute_log_returns(close)
    rv_all  = compute_rv(close, h=horizon)

    results = {}
    for ticker in tickers:
        if ticker not in log_ret.columns:
            continue
        try:
            preds = fit_garch_predictions(
                log_ret[ticker], rv_all[ticker],
                train_end_ts, test_start_ts,
            )
            results[ticker] = preds
        except Exception as e:
            print(f"  GARCH {ticker} FAILED: {e}")

    return results


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    prices_dir = cfg["data"]["prices_dir"]
    train_end  = cfg["data"]["train_end"]
    test_start = cfg["data"]["test_start"]
    horizon    = cfg["model"]["horizon"]

    test_tickers = ["VCB", "HPG", "FPT"]
    print(f"GARCH(1,1) baseline — {len(test_tickers)} tickers")
    results = run_garch_baseline(
        prices_dir, train_end=train_end, test_start=test_start,
        horizon=horizon, tickers=test_tickers,
    )

    print(f"\nResults:")
    for t, s in results.items():
        print(f"  {t}: {len(s)} predictions, mean={s.mean():.4f}, std={s.std():.4f}")
    print("\ngarch_baseline.py smoke test OK.")
