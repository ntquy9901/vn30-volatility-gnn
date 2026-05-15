"""
HAR-RV baseline: Corsi (2009) Heterogeneous Autoregressive model for Realized Volatility.

Model: RV_t = α + β_d·RV_{t-1} + β_w·RV_{t-5}^(5) + β_m·RV_{t-22}^(22) + ε
where RV^(k) = mean(RV_{t-1},...,RV_{t-k}) (weekly and monthly averages)

Fitted via OLS per stock on training data.  No re-fit on test (true out-of-sample).

Usage:
    python baselines/har_rv_baseline.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_rv
from gnn.build_graph import VN30_TICKERS


def build_har_features(rv: pd.Series) -> pd.DataFrame:
    """
    Build HAR feature matrix: [RV_{t-1}, RV^(5)_{t}, RV^(22)_{t}].
    Index aligns with rv (prediction target aligned separately).
    """
    rv_d  = rv.shift(1)                                  # daily lag
    rv_w  = rv.shift(1).rolling(5,  min_periods=5).mean()  # 5-day avg ending at t-1
    rv_m  = rv.shift(1).rolling(22, min_periods=22).mean() # 22-day avg ending at t-1

    features = pd.DataFrame({
        "const": 1.0,
        "RV_d":  rv_d,
        "RV_w":  rv_w,
        "RV_m":  rv_m,
    }, index=rv.index)
    return features


def fit_har(
    rv: pd.Series,
    train_end: pd.Timestamp,
) -> np.ndarray:
    """
    OLS fit on training data.  Returns coefficient vector [α, β_d, β_w, β_m].
    """
    features = build_har_features(rv)
    df = pd.concat([features, rv.rename("target")], axis=1).dropna()
    train = df[df.index <= train_end]

    X = train[["const", "RV_d", "RV_w", "RV_m"]].values
    y = train["target"].values
    # OLS: β = (X'X)^{-1} X'y
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def predict_har(
    rv: pd.Series,
    coeffs: np.ndarray,
    test_start: pd.Timestamp,
) -> pd.Series:
    """
    Out-of-sample predictions on test set (no re-fitting).
    """
    features = build_har_features(rv)
    df = pd.concat([features, rv.rename("target")], axis=1)
    test = df[df.index >= test_start].dropna()

    X    = test[["const", "RV_d", "RV_w", "RV_m"]].values
    pred = X @ coeffs
    pred = np.maximum(pred, 0.0)   # RV cannot be negative
    return pd.Series(pred, index=test.index, name=rv.name)


def run_har_baseline(
    prices_dir: str,
    train_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    horizon: int = 20,
    tickers: list[str] | None = None,
) -> dict[str, pd.Series]:
    """
    Run HAR-RV baseline for all VN30 stocks.

    Returns:
        dict mapping ticker -> pd.Series of predictions aligned with test RV.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    train_end_ts  = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    close  = load_close_prices(prices_dir, tickers=tickers)
    rv_all = compute_rv(close, h=horizon)

    results = {}
    for ticker in tickers:
        if ticker not in rv_all.columns:
            continue
        try:
            coeffs = fit_har(rv_all[ticker], train_end_ts)
            preds  = predict_har(rv_all[ticker], coeffs, test_start_ts)
            results[ticker] = preds
        except Exception as e:
            print(f"  HAR {ticker} FAILED: {e}")

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

    print("HAR-RV baseline smoke test")
    results = run_har_baseline(
        prices_dir, train_end=train_end, test_start=test_start, horizon=horizon,
    )

    print(f"\nResults summary ({len(results)} stocks):")
    for t, s in list(results.items())[:5]:
        print(f"  {t}: {len(s)} test points, mean={s.mean():.4f}, std={s.std():.4f}")

    # Verify coefficients on VCB
    close  = load_close_prices(prices_dir, tickers=["VCB"])
    rv_vcb = compute_rv(close, h=horizon)["VCB"]
    c = fit_har(rv_vcb, pd.Timestamp(train_end))
    print(f"\nVCB HAR coefficients: alpha={c[0]:.4f}, beta_d={c[1]:.4f}, beta_w={c[2]:.4f}, beta_m={c[3]:.4f}")
    print("\nhar_rv_baseline.py smoke test OK.")
