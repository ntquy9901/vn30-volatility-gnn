"""
Evaluation metrics for volatility forecasting.

Metrics implemented:
  - MAE   : Mean Absolute Error
  - RMSE  : Root Mean Squared Error
  - R2    : Coefficient of determination (Mincer-Zarnowitz R²)
  - QLIKE : Patton (2011) quasi-likelihood loss for volatility
  - DM    : Diebold-Mariano test (two-sided, Harvey et al. 1997 correction)
  - MZ    : Mincer-Zarnowitz regression (intercept, slope, R²)

All functions accept 1-D numpy arrays or pandas Series.

Usage:
    python evaluation/metrics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import NamedTuple


def _to_array(*args) -> list[np.ndarray]:
    return [np.asarray(a, dtype=np.float64).ravel() for a in args]


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def pearson_r(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true, y_pred)
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def qlike(y_true, y_pred) -> float:
    """
    QLIKE loss (Patton 2011): L = sigma_hat/RV - log(sigma_hat/RV) - 1
    where sigma_hat = predicted RV, RV = realized volatility.
    Both inputs are in daily std units (not variance).
    Clips predictions below 1e-8 to avoid log(0).
    """
    y_true, y_pred = _to_array(y_true, y_pred)
    var_true = np.maximum(y_true ** 2, 1e-10)
    var_pred = np.maximum(y_pred ** 2, 1e-10)
    return float(np.mean(var_pred / var_true - np.log(var_pred / var_true) - 1.0))


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

class DMResult(NamedTuple):
    statistic: float
    p_value:   float
    better:    str   # "model1" | "model2" | "equal"


def diebold_mariano(
    y_true,
    y_pred1,
    y_pred2,
    loss: str = "qlike",
    h: int = 1,
) -> DMResult:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction.

    H0: Equal predictive accuracy between model1 and model2.

    Args:
        y_true : Realized volatility (1-D).
        y_pred1: Predictions from model1 (GNN / proposed model).
        y_pred2: Predictions from baseline model2.
        loss   : "mse", "mae", or "qlike".
        h      : Forecast horizon for autocorrelation correction.

    Returns:
        DMResult(statistic, p_value, better) where better indicates which model
        is more accurate (lower loss), or "equal" if H0 not rejected at 5%.
    """
    y_true, y_pred1, y_pred2 = _to_array(y_true, y_pred1, y_pred2)

    if loss == "mse":
        d = (y_true - y_pred1) ** 2 - (y_true - y_pred2) ** 2
    elif loss == "mae":
        d = np.abs(y_true - y_pred1) - np.abs(y_true - y_pred2)
    elif loss == "qlike":
        d = _qlike_loss_vec(y_true, y_pred1) - _qlike_loss_vec(y_true, y_pred2)
    else:
        raise ValueError(f"Unknown loss: {loss!r}. Use 'mse', 'mae', or 'qlike'.")

    n       = len(d)
    d_bar   = np.mean(d)

    # Newey-West variance with h-1 lags
    gamma0 = np.var(d, ddof=0)
    gamma  = gamma0
    for k in range(1, h):
        cov_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma += 2.0 * cov_k
    var_d = max(gamma / n, 1e-20)

    # Harvey et al. (1997) small-sample correction
    # Multiply by sqrt((n + 1 - 2h + h(h-1)/n) / n)
    correction = np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    dm_stat    = d_bar / np.sqrt(var_d) * correction

    # Two-sided t-test with n-1 degrees of freedom
    p_val = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1))

    if p_val > 0.05:
        better = "equal"
    elif dm_stat < 0:
        better = "model1"  # model1 has lower loss
    else:
        better = "model2"

    return DMResult(statistic=float(dm_stat), p_value=float(p_val), better=better)


def _qlike_loss_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    var_true = np.maximum(y_true ** 2, 1e-10)
    var_pred = np.maximum(y_pred ** 2, 1e-10)
    return var_pred / var_true - np.log(var_pred / var_true) - 1.0


# ── Mincer-Zarnowitz regression ───────────────────────────────────────────────

class MZResult(NamedTuple):
    intercept: float
    slope:     float
    r2:        float
    p_intercept_zero: float  # p-value for H0: intercept=0
    p_slope_one:      float  # p-value for H0: slope=1


def mincer_zarnowitz(y_true, y_pred) -> MZResult:
    """
    Mincer-Zarnowitz (1969) regression: RV = a + b * RV_hat + eps
    Efficient forecast: a=0, b=1.

    Returns intercept, slope, R², and p-values for H0: a=0 and H0: b=1.
    """
    y_true, y_pred = _to_array(y_true, y_pred)
    n = len(y_true)

    X = np.column_stack([np.ones(n), y_pred])
    # OLS
    coeffs, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
    a, b = coeffs

    y_hat = X @ coeffs
    resid = y_true - y_hat
    s2    = np.sum(resid ** 2) / (n - 2)

    XtX_inv = np.linalg.pinv(X.T @ X)
    se      = np.sqrt(s2 * np.diag(XtX_inv))

    # t-stats and p-values
    t_a = a / se[0]
    t_b = (b - 1.0) / se[1]
    p_a = 2.0 * (1.0 - stats.t.cdf(abs(t_a), df=n - 2))
    p_b = 2.0 * (1.0 - stats.t.cdf(abs(t_b), df=n - 2))

    r2 = r2_score(y_true, y_hat)

    return MZResult(
        intercept=float(a),
        slope=float(b),
        r2=r2,
        p_intercept_zero=float(p_a),
        p_slope_one=float(p_b),
    )


# ── Aggregate evaluation ──────────────────────────────────────────────────────

def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    name: str = "model",
) -> dict:
    """
    Compute all metrics for a single model vs realized volatility.

    Returns dict suitable for pd.DataFrame row.
    """
    mz = mincer_zarnowitz(y_true, y_pred)
    return {
        "model":    name,
        "MAE":      mae(y_true, y_pred),
        "RMSE":     rmse(y_true, y_pred),
        "R2":       r2_score(y_true, y_pred),
        "QLIKE":    qlike(y_true, y_pred),
        "Pearson_r": pearson_r(y_true, y_pred),
        "MZ_alpha": mz.intercept,
        "MZ_beta":  mz.slope,
        "MZ_R2":    mz.r2,
        "MZ_p_alpha0": mz.p_intercept_zero,
        "MZ_p_beta1":  mz.p_slope_one,
    }


def compare_models(
    y_true: pd.Series | np.ndarray,
    predictions: dict[str, pd.Series | np.ndarray],
    dm_reference: str | None = None,
    dm_loss: str = "qlike",
) -> pd.DataFrame:
    """
    Compute metric table and optional DM test results for all models.

    Args:
        y_true      : Ground-truth realized volatility.
        predictions : {model_name: predictions} dict.
        dm_reference: Name of the reference model for DM test (e.g., 'GNN').
                      If None, DM test is skipped.
        dm_loss     : Loss function for DM test ('qlike', 'mse', 'mae').

    Returns:
        DataFrame with one row per model.
    """
    rows = []
    for name, y_pred in predictions.items():
        # Align on common valid index
        yt, yp = _to_array(y_true, y_pred)
        valid  = ~(np.isnan(yt) | np.isnan(yp))
        row    = evaluate_model(yt[valid], yp[valid], name=name)

        if dm_reference is not None and name != dm_reference:
            ref_pred = predictions[dm_reference]
            _, ref_p = _to_array(y_true, ref_pred)
            valid_dm = ~(np.isnan(yt) | np.isnan(yp) | np.isnan(ref_p))
            dm       = diebold_mariano(
                yt[valid_dm], ref_p[valid_dm], yp[valid_dm], loss=dm_loss
            )
            row["DM_stat"]  = dm.statistic
            row["DM_pval"]  = dm.p_value
            row["DM_better"] = dm.better
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 100

    # Simulate: true RV ~ abs(Normal), model1 ≈ truth + small noise, model2 = worse
    rv_true  = np.abs(np.random.normal(0.015, 0.005, n))
    pred1    = rv_true + np.random.normal(0, 0.002, n)   # good model
    pred2    = rv_true + np.random.normal(0.005, 0.004, n)  # worse model

    print("=== Individual metrics ===")
    m1 = evaluate_model(rv_true, pred1, "GNN")
    m2 = evaluate_model(rv_true, pred2, "GARCH")
    for k, v in m1.items():
        if k != "model":
            print(f"  GNN   {k}: {v:.4f}")
    print()
    for k, v in m2.items():
        if k != "model":
            print(f"  GARCH {k}: {v:.4f}")

    print("\n=== DM test (GNN vs GARCH, QLIKE) ===")
    dm = diebold_mariano(rv_true, pred1, pred2, loss="qlike")
    print(f"  stat={dm.statistic:.3f}, p={dm.p_value:.4f}, better={dm.better!r}")

    print("\n=== compare_models table ===")
    df = compare_models(
        rv_true,
        {"GNN": pred1, "GARCH": pred2},
        dm_reference="GNN",
        dm_loss="qlike",
    )
    print(df[["MAE", "RMSE", "R2", "QLIKE", "Pearson_r", "DM_stat", "DM_pval"]].to_string())
    print("\nmetrics.py smoke test OK.")
