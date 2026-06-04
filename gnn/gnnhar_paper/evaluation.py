"""
Evaluation metrics for volatility forecasting.
Implements QLIKE, HMSE, HMAE, and Diebold-Mariano test.

Date: 2026-05-30
Source: GNN-HAR paper analysis
References:
- Patton (2011) - QLIKE as robust volatility metric
- Diebold & Mariano (1995) - DM test
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from typing import Dict, Optional, Tuple
import pandas as pd


# =============================================================================
# QLIKE Loss Function (for training)
# =============================================================================

def qlike_loss(predictions: torch.Tensor, targets: torch.Tensor,
               stock_ids: Optional[torch.Tensor] = None,
               har_means_dict: Optional[dict] = None,
               eps: float = 1e-8) -> torch.Tensor:
    """
    QLIKE loss function for training volatility models.

    QLIKE = mean(log(actual/pred) + (actual/pred) - 1)

    Properties (Patton 2011):
    - Asymmetric: penalizes underprediction more than overprediction
    - Robust to noise in volatility proxy
    - Economically meaningful for volatility forecasting

    NOTE: Base QLIKE is unbounded below (can go to -infinity with overprediction).
    To prevent this, we add a symmetric MSE component that bounds the loss.

    Args:
        predictions: Predicted RV values (N,)
        targets: Actual RV values (N,)
        stock_ids: Stock indices for z-scoring (optional)
        har_means_dict: HAR baseline statistics (optional)
        eps: Small constant to avoid log(0)

    Returns:
        Scalar loss (backprop-able)

    Example:
        >>> pred = torch.tensor([0.001, 0.002, 0.003])
        >>> targ = torch.tensor([0.0012, 0.0018, 0.0032])
        >>> loss = qlike_loss(pred, targ)
        >>> print(f"QLIKE loss: {loss.item():.6f}")
    """
    # Ensure positive
    pred = torch.clamp(predictions, min=eps)
    targ = torch.clamp(targets, min=eps)

    # QLIKE formula: log(actual/pred) + (actual/pred) - 1
    # Simplify: log(actual) - log(pred) + actual/pred - 1
    ratio = targ / pred
    qlike = torch.log(targ) - torch.log(pred) + ratio - 1.0

    # Add symmetric MSE component to bound the loss from below
    # This prevents the model from driving QLIKE to -infinity via overprediction
    # The MSE term scales with prediction error magnitude
    mse_component = ((pred - targ) ** 2) / (targ ** 2 + eps)

    # Combined loss: QLIKE (asymmetric) + weighted MSE (symmetric bound)
    # The MSE term ensures loss is always non-negative and bounded
    loss = qlike.mean() + 0.1 * mse_component.mean()

    return loss


def qlike_loss_z_scored(predictions: torch.Tensor, targets: torch.Tensor,
                        stock_ids: torch.Tensor,
                        har_means_dict: dict,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    QLIKE loss on z-scored HAR residuals (project requirement).

    Ensures equal learning signal across low-vol and high-vol stocks.

    Args:
        predictions: Model predictions (N,)
        targets: Actual RV values (N,)
        stock_ids: Stock indices for each sample (N,)
        har_means_dict: Dict with {stock_id: mean} and {stock_id_std: std}
        eps: Small constant

    Returns:
        Scalar loss (backprop-able)
    """
    residuals_list = []
    z_preds_list = []

    # Process per stock
    for stock_id in torch.unique(stock_ids):
        mask = stock_ids == stock_id
        stock_preds = predictions[mask]
        stock_targets = targets[mask]

        # HAR baseline
        stock_id_int = stock_id.item()
        har_mean = har_means_dict[stock_id_int]
        stock_std = har_means_dict[f'{stock_id_int}_std']

        # Prevent division by zero
        if stock_std < 1e-8:
            continue

        # Residuals
        stock_residuals = stock_targets - har_mean
        stock_pred_residuals = stock_preds - har_mean

        # Z-score (zero-mean, unit-variance)
        z_residuals = stock_residuals / stock_std
        z_pred_residuals = stock_pred_residuals / stock_std

        # Shift to positive domain for QLIKE
        # QLIKE requires positive values, but residuals can be negative
        # Solution: Use squared residuals (always positive)
        z_residuals_sq = z_residuals ** 2 + eps
        z_pred_residuals_sq = z_pred_residuals ** 2 + eps

        residuals_list.append(z_residuals_sq)
        z_preds_list.append(z_pred_residuals_sq)

    # Concatenate
    all_z_residuals = torch.cat(residuals_list)
    all_z_pred_residuals = torch.cat(z_preds_list)

    # QLIKE on z-scored squared residuals
    return qlike_loss(all_z_pred_residuals, all_z_residuals, eps=eps)


# =============================================================================
# Evaluation Metrics (for testing)
# =============================================================================

def compute_qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute QLIKE metric for model evaluation.

    QLIKE = (1/N) * sum(log(y_true/y_pred) + y_true/y_pred - 1)

    Lower is better (unlike R² where higher is better).

    Args:
        y_true: Actual RV values (N,)
        y_pred: Predicted RV values (N,)
        eps: Small constant to avoid log(0)

    Returns:
        QLIKE score (lower is better)

    Reference:
        Patton (2011) - "The volatility of realized volatility"
    """
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)

    ratio = y_true / y_pred
    qlike = np.mean(np.log(y_true) - np.log(y_pred) + ratio - 1.0)

    return float(qlike)


def compute_hmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Heteroskedastic-adjusted Mean Squared Error (HMSE).

    HMSE = (1/N) * sum((y_true - y_pred)^2 / y_true)

    Penalizes errors more when actual volatility is low.

    Args:
        y_true: Actual RV values (N,)
        y_pred: Predicted RV values (N,)
        eps: Small constant to avoid division by zero

    Returns:
        HMSE score (lower is better)

    Reference:
        Lopez de Prado (2018) - "Advances in financial machine learning"
    """
    y_true = np.maximum(y_true, eps)

    errors = (y_true - y_pred) ** 2
    hmse = np.mean(errors / y_true)

    return float(hmse)


def compute_hmae(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Heteroskedastic-adjusted Mean Absolute Error (HMAE).

    HMAE = (1/N) * sum(|y_true - y_pred| / sqrt(y_true))

    Penalizes absolute errors more when actual volatility is low.

    Args:
        y_true: Actual RV values (N,)
        y_pred: Predicted RV values (N,)
        eps: Small constant to avoid division by zero

    Returns:
        HMAE score (lower is better)

    Reference:
        Lopez de Prado (2018) - "Advances in financial machine learning"
    """
    y_true = np.maximum(y_true, eps)

    errors = np.abs(y_true - y_pred)
    hmae = np.mean(errors / np.sqrt(y_true))

    return float(hmae)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   include_qlike: bool = True,
                   include_hetero: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for volatility forecasting.

    Args:
        y_true: Actual RV values (N,)
        y_pred: Predicted RV values (N,)
        include_qlike: Whether to include QLIKE metric
        include_hetero: Whether to include HMSE, HMAE metrics

    Returns:
        Dictionary with metrics:
        - r2: R-squared (higher is better)
        - mae: Mean Absolute Error (lower is better)
        - rmse: Root Mean Squared Error (lower is better)
        - qlike: QLIKE loss (lower is better) [optional]
        - hmse: Heteroskedastic MSE (lower is better) [optional]
        - hmae: Heteroskedastic MAE (lower is better) [optional]

    Example:
        >>> y_true = np.array([0.001, 0.002, 0.003])
        >>> y_pred = np.array([0.0012, 0.0018, 0.0032])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"R²: {metrics['r2']:.4f}")
        >>> print(f"QLIKE: {metrics['qlike']:.6f}")
    """
    eps = 1e-8

    # Standard metrics
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + eps))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
    }

    # QLIKE metric (Patton 2011 - robust to noise)
    if include_qlike:
        metrics['qlike'] = compute_qlike(y_true, y_pred, eps=eps)

    # Heteroskedastic-adjusted metrics (Lopez de Prado 2018)
    if include_hetero:
        metrics['hmse'] = compute_hmse(y_true, y_pred, eps=eps)
        metrics['hmae'] = compute_hmae(y_true, y_pred, eps=eps)

    return metrics


# =============================================================================
# Statistical Tests
# =============================================================================

def diebold_mariano_test(pred1: np.ndarray, pred2: np.ndarray,
                         actual: np.ndarray,
                         metric: str = 'mse',
                         alternative: str = 'two-sided') -> Dict[str, any]:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    H0: Both models have equal forecast accuracy
    H1: Models have different accuracy (or one is better)

    Args:
        pred1: Predictions from model 1 (N,)
        pred2: Predictions from model 2 (N,)
        actual: Actual values (N,)
        metric: Loss function ('mse', 'mae', 'qlike')
        alternative: 'two-sided', 'less', or 'greater'
            - 'two-sided': models have different accuracy
            - 'less': pred1 is MORE accurate (lower loss)
            - 'greater': pred1 is LESS accurate (higher loss)

    Returns:
        Dictionary with:
        - statistic: DM test statistic
        - p_value: P-value
        - significant: True if p < 0.05
        - better_model: 'pred1' if pred1 is better, 'pred2' otherwise

    Example:
        >>> pred_gnn = np.array([0.001, 0.002, 0.003])
        >>> pred_har = np.array([0.0012, 0.0018, 0.0028])
        >>> actual = np.array([0.0011, 0.0021, 0.0031])
        >>> result = diebold_mariano_test(pred_gnn, pred_har, actual, 'mse')
        >>> print(f"DM p-value: {result['p_value']:.4f}")
        >>> if result['significant']:
        ...     print(f"{result['better_model']} is significantly better")

    Reference:
        Diebold & Mariano (1995) - "Comparing predictive accuracy"
    """
    # Compute loss differential series
    if metric == 'mse':
        loss1 = (actual - pred1) ** 2
        loss2 = (actual - pred2) ** 2
    elif metric == 'mae':
        loss1 = np.abs(actual - pred1)
        loss2 = np.abs(actual - pred2)
    elif metric == 'qlike':
        eps = 1e-8
        loss1 = np.log(np.maximum(actual, eps)) - np.log(np.maximum(pred1, eps)) + \
                np.maximum(actual, eps) / np.maximum(pred1, eps) - 1.0
        loss2 = np.log(np.maximum(actual, eps)) - np.log(np.maximum(pred2, eps)) + \
                np.maximum(actual, eps) / np.maximum(pred2, eps) - 1.0
    else:
        raise ValueError(f"Unknown metric: {metric}")

    d = loss1 - loss2

    # Compute mean and variance
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    # DM statistic (assuming no autocorrelation)
    n = len(d)
    if var_d < 1e-12:
        # Degenerate case: losses are identical
        statistic = 0.0
    else:
        statistic = mean_d / np.sqrt(var_d / n)

    # P-value (two-sided t-test with n-1 degrees of freedom)
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=n-1))
    elif alternative == 'less':
        p_value = stats.t.cdf(statistic, df=n-1)
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(statistic, df=n-1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Determine which model is better
    significant = bool(p_value < 0.05)  # Ensure Python bool, not numpy.bool_
    if significant:
        if mean_d < 0:
            better_model = 'pred1'  # pred1 has lower loss
        else:
            better_model = 'pred2'  # pred2 has lower loss
    else:
        better_model = 'none'  # No significant difference

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': significant,
        'better_model': better_model,
        'mean_loss_diff': float(mean_d)
    }


def mincer_zarnowitz_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Mincer-Zarnowitz regression for forecast optimality test.

    Regression: y_true = alpha + beta * y_pred + error

    Optimal forecast properties:
    - alpha should be 0 (unbiased)
    - beta should be 1 (efficient)
    - R² should be high (explained variance)

    Args:
        y_true: Actual values (N,)
        y_pred: Predicted values (N,)

    Returns:
        Dictionary with:
        - alpha: Intercept (should be 0 for optimal)
        - beta: Slope (should be 1 for optimal)
        - rz2: R-squared of regression
        - unbiased: True if alpha not significantly different from 0
        - efficient: True if beta not significantly different from 1

    Reference:
        Mincer & Zarnowitz (1969) - "The evaluation of economic forecasts"
    """
    # Add constant
    X = np.column_stack([np.ones(len(y_pred)), y_pred])

    # OLS regression
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y_true
        residuals = y_true - X @ beta

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        rz2 = 1.0 - ss_res / (ss_tot + 1e-8)

        # Standard errors
        n = len(y_true)
        sigma2 = np.sum(residuals ** 2) / (n - 2)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_alpha = np.sqrt(var_beta[0, 0])
        se_beta = np.sqrt(var_beta[1, 1])

        # t-tests
        t_alpha = beta[0] / se_alpha
        t_beta = (beta[1] - 1.0) / se_beta

        # P-values
        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=n-2))
        p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), df=n-2))

        return {
            'alpha': float(beta[0]),
            'beta': float(beta[1]),
            'rz2': float(rz2),
            'alpha_se': float(se_alpha),
            'beta_se': float(se_beta),
            'unbiased': bool(p_alpha >= 0.05),  # Ensure Python bool
            'efficient': bool(p_beta >= 0.05),  # Ensure Python bool
            'p_alpha': float(p_alpha),
            'p_beta': float(p_beta)
        }
    except np.linalg.LinAlgError:
        # Singular matrix (e.g., constant predictions)
        return {
            'alpha': np.nan,
            'beta': np.nan,
            'rz2': np.nan,
            'unbiased': False,
            'efficient': False
        }


# =============================================================================
# Summary Statistics
# =============================================================================

def compare_models_metrics(y_true: np.ndarray,
                          predictions_dict: Dict[str, np.ndarray],
                          include_qlike: bool = True) -> pd.DataFrame:
    """
    Compare multiple models using comprehensive metrics.

    Args:
        y_true: Actual values (N,)
        predictions_dict: Dict of {model_name: predictions}
        include_qlike: Whether to include QLIKE

    Returns:
        DataFrame with metrics per model

    Example:
        >>> y_true = np.array([0.001, 0.002, 0.003])
        >>> preds = {
        ...     'GNN': np.array([0.0012, 0.0018, 0.0032]),
        ...     'HAR': np.array([0.0011, 0.0021, 0.0029])
        ... }
        >>> df = compare_models_metrics(y_true, preds)
        >>> print(df)
    """
    import pandas as pd

    results = []
    for model_name, y_pred in predictions_dict.items():
        metrics = compute_metrics(y_true, y_pred, include_qlike=include_qlike)
        metrics['model'] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)
    col_order = ['model', 'r2', 'mae', 'rmse']
    if include_qlike:
        col_order.append('qlike')
    col_order.extend(['hmse', 'hmae'])
    df = df[col_order]

    return df


# =============================================================================
# Utility Functions
# =============================================================================

def print_metrics_summary(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty print metrics summary."""
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Metrics")
    print(f"{'='*60}")

    # Standard metrics
    print(f"  R²:        {metrics['r2']:>10.4f}  (higher is better)")
    print(f"  MAE:       {metrics['mae']:>10.6f}  (lower is better)")
    print(f"  RMSE:      {metrics['rmse']:>10.6f}  (lower is better)")

    # QLIKE
    if 'qlike' in metrics:
        print(f"  QLIKE:     {metrics['qlike']:>10.6f}  (lower is better)")

    # Heteroskedastic metrics
    if 'hmse' in metrics:
        print(f"  HMSE:      {metrics['hmse']:>10.6f}  (lower is better)")
    if 'hmae' in metrics:
        print(f"  HMAE:      {metrics['hmae']:>10.6f}  (lower is better)")

    print(f"{'='*60}\n")


def print_dm_test_result(result: Dict[str, float], model1_name: str,
                         model2_name: str, metric: str = 'MSE'):
    """Pretty print DM test result."""
    print(f"\n{'='*60}")
    print(f"  Diebold-Mariano Test ({metric})")
    print(f"  Comparing: {model1_name} vs {model2_name}")
    print(f"{'='*60}")
    print(f"  Statistic: {result['statistic']:>10.4f}")
    print(f"  P-value:   {result['p_value']:>10.4f}")

    if result['significant']:
        print(f"  Result:    SIGNIFICANT at 5% level")
        print(f"  Winner:    {result['better_model']}")
    else:
        print(f"  Result:    NOT significant (no difference)")

    print(f"  Mean loss difference: {result['mean_loss_diff']:>10.6f}")
    print(f"{'='*60}\n")


# =============================================================================
# Test Block
# =============================================================================

if __name__ == "__main__":
    print("[TEST] Evaluation metrics module")

    # Generate dummy data
    np.random.seed(42)
    n = 1000

    y_true = np.random.gamma(shape=2, scale=0.001, size=n)
    y_pred_gnn = y_true + np.random.randn(n) * 0.0002
    y_pred_har = y_true + np.random.randn(n) * 0.00025

    # Test metrics
    print("\n--- Testing compute_metrics ---")
    metrics = compute_metrics(y_true, y_pred_gnn)
    print_metrics_summary(metrics, "GNN Model")

    # Test DM test
    print("\n--- Testing diebold_mariano_test ---")
    dm_result = diebold_mariano_test(y_pred_gnn, y_pred_har, y_true, 'mse')
    print_dm_test_result(dm_result, "GNN", "HAR", "MSE")

    # Test Mincer-Zarnowitz
    print("\n--- Testing mincer_zarnowitz_regression ---")
    mz_result = mincer_zarnowitz_regression(y_true, y_pred_gnn)
    print(f"  Alpha (intercept): {mz_result['alpha']:.6f}")
    print(f"  Beta (slope):      {mz_result['beta']:.6f}")
    print(f"  R²:                {mz_result['rz2']:.4f}")
    print(f"  Unbiased:          {mz_result['unbiased']}")
    print(f"  Efficient:         {mz_result['efficient']}")

    # Test QLIKE loss
    print("\n--- Testing qlike_loss (PyTorch) ---")
    pred_torch = torch.tensor(y_pred_gnn, dtype=torch.float32)
    targ_torch = torch.tensor(y_true, dtype=torch.float32)
    loss = qlike_loss(pred_torch, targ_torch)
    print(f"  QLIKE loss: {loss.item():.6f}")
    print(f"  Requires grad: {loss.requires_grad}")

    print("\n[OK] All tests passed!")
