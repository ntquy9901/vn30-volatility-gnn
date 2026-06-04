"""
Evaluate sklearn GHAR baselines for multi-stock volatility forecasting.

Compares:
1. HAR OLS (per-stock sklearn LinearRegression) - established baseline
2. sklearn GHAR with identity adjacency - should match HAR OLS
3. sklearn GHAR with Pearson adjacency - tests if graph helps
4. sklearn GHAR with GLASSO adjacency - paper's graph method

Expected results:
- HAR OLS: R2 approx 0.60-0.70 (baseline from single-stock VIC)
- GHAR (iden): Should match HAR OLS (same features, just different implementation)
- GHAR (pearson): Should beat HAR OLS if graph signal exists
- GHAR (glasso): Should beat HAR OLS if partial correlations help

Usage:
    python gnn/gnnhar_paper/evaluate_sklearn_baseline.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.ghar_sklearn import GHARSklearn
from src.volatility_labels import compute_log_returns
from gnn.build_graph import VN30_TICKERS

print("\n" + "="*70)
print("  SKLEARN GHAR BASELINE EVALUATION (MULTI-STOCK)")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZON = 5
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================

print("[Data] Loading multi-stock data...")
loader = MultiStockDataLoader(
    tickers=VN30_TICKERS,
    horizon=HORIZON,
    train_end=TRAIN_END,
    test_start=TEST_START,
)
loader.load_data()
loader.build_features()
loader.flatten_dataset()
loader.split_train_val_test()

X_train, y_train, stocks_train, dates_train, X_test, y_test, stocks_test, dates_test = loader.prepare_sklearn_data()

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Features: {X_train.shape[1]} (HAR: RV_d, RV_w, RV_m)")

# Load returns for graph construction
returns = compute_log_returns(loader.close)
print(f"  Returns for graph: {returns.shape}")

# =============================================================================
# BASELINE 1: HAR OLS (PER-STOCK LINEAR REGRESSION)
# =============================================================================

print("\n" + "="*70)
print("  BASELINE 1: HAR OLS (PER-STOCK LINEAR REGRESSION)")
print("="*70)

def fit_har_ols_per_stock(X_train, y_train, stocks_train):
    """
    Fit separate HAR OLS model per stock using sklearn LinearRegression.

    This is the established baseline (R2 approx 0.63 for VIC single-stock).
    """
    models = {}

    for stock_id in np.unique(stocks_train):
        mask = (stocks_train == stock_id)
        X_stock = X_train[mask]
        y_stock = y_train[mask]

        if len(X_stock) == 0:
            continue

        model = LinearRegression(fit_intercept=True, n_jobs=-1)
        model.fit(X_stock, y_stock)
        models[stock_id] = model

    return models

def predict_har_ols_per_stock(models, X_test, stocks_test):
    """Predict with per-stock HAR OLS models."""
    preds = np.zeros(len(X_test))

    for stock_id, model in models.items():
        mask = (stocks_test == stock_id)
        if mask.any():
            preds[mask] = model.predict(X_test[mask])

    return preds

print("\n[HAR OLS] Training per-stock LinearRegression models...")
har_ols_models = fit_har_ols_per_stock(X_train, y_train, stocks_train)
print(f"  Trained {len(har_ols_models)} stock-specific models")

print("\n[HAR OLS] Predicting on test set...")
y_pred_har_ols = predict_har_ols_per_stock(har_ols_models, X_test, stocks_test)

# Clip negatives
y_pred_har_ols = np.maximum(y_pred_har_ols, 0.0)

# Compute metrics
r2_har_ols = r2_score(y_test, y_pred_har_ols)
mae_har_ols = mean_absolute_error(y_test, y_pred_har_ols)

print(f"\n[HAR OLS] Results:")
print(f"  R2:   {r2_har_ols:+.4f}")
print(f"  MAE:  {mae_har_ols:.6f}")

# =============================================================================
# BASELINE 2: SKLEARN GHAR WITH IDENTITY ADJACENCY
# =============================================================================

print("\n" + "="*70)
print("  BASELINE 2: SKLEARN GHAR (IDENTITY ADJACENCY)")
print("="*70)

print("\n[GHAR-iden] Training sklearn GHAR with identity adjacency...")
model_iden = GHARSklearn(
    adj_method='iden',
    graph_end_date=TRAIN_END,
)
model_iden.fit(X_train, y_train, stocks_train, dates_train, returns)

print("\n[GHAR-iden] Predicting on test set...")
y_pred_iden = model_iden.predict(X_test, stocks_test, dates_test)
metrics_iden = model_iden.evaluate(y_test, y_pred_iden)

print(f"\n[GHAR-iden] Results:")
print(f"  R2:   {metrics_iden['r2']:+.4f}")
print(f"  MAE:  {metrics_iden['mae']:.6f}")

# =============================================================================
# BASELINE 3: SKLEARN GHAR WITH PEARSON ADJACENCY
# =============================================================================

print("\n" + "="*70)
print("  BASELINE 3: SKLEARN GHAR (PEARSON ADJACENCY)")
print("="*70)

print("\n[GHAR-pearson] Training sklearn GHAR with Pearson adjacency...")
model_pearson = GHARSklearn(
    adj_method='pearson',
    threshold=0.3,
    corr_window=60,
    graph_end_date=TRAIN_END,
)
model_pearson.fit(X_train, y_train, stocks_train, dates_train, returns)

print("\n[GHAR-pearson] Predicting on test set...")
y_pred_pearson = model_pearson.predict(X_test, stocks_test, dates_test)
metrics_pearson = model_pearson.evaluate(y_test, y_pred_pearson)

print(f"\n[GHAR-pearson] Results:")
print(f"  R2:   {metrics_pearson['r2']:+.4f}")
print(f"  MAE:  {metrics_pearson['mae']:.6f}")

# =============================================================================
# BASELINE 4: SKLEARN GHAR WITH GLASSO ADJACENCY
# =============================================================================

print("\n" + "="*70)
print("  BASELINE 4: SKLEARN GHAR (GLASSO ADJACENCY)")
print("="*70)

print("\n[GHAR-glasso] Training sklearn GHAR with GLASSO adjacency...")
model_glasso = GHARSklearn(
    adj_method='glasso',
    glasso_alpha=0.01,
    corr_window=60,
    graph_end_date=TRAIN_END,
)
model_glasso.fit(X_train, y_train, stocks_train, dates_train, returns)

print("\n[GHAR-glasso] Predicting on test set...")
y_pred_glasso = model_glasso.predict(X_test, stocks_test, dates_test)
metrics_glasso = model_glasso.evaluate(y_test, y_pred_glasso)

print(f"\n[GHAR-glasso] Results:")
print(f"  R2:   {metrics_glasso['r2']:+.4f}")
print(f"  MAE:  {metrics_glasso['mae']:.6f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("  SUMMARY: BASELINE COMPARISON")
print("="*70 + "\n")

print(f"Test period: {TEST_START} to 2026-05-31")
print(f"Test samples: {len(X_test)}")
print(f"Distribution shift: {(y_test.mean() - y_train.mean()) / y_train.mean() * 100:+.1f}%")
print("\nModel Performance:")
print(f"{'Model':<25} {'R2':>10} {'MAE':>12} {'Improvement':>15}")
print(f"{'-'*60}")

baseline_r2 = r2_har_ols

results = [
    ("HAR OLS (per-stock)", r2_har_ols, mae_har_ols),
    ("GHAR (identity)", metrics_iden['r2'], metrics_iden['mae']),
    ("GHAR (pearson)", metrics_pearson['r2'], metrics_pearson['mae']),
    ("GHAR (glasso)", metrics_glasso['r2'], metrics_glasso['mae']),
]

for model_name, r2, mae in results:
    improvement = r2 - baseline_r2
    print(f"{model_name:<25} {r2:>+10.4f} {mae:>12.6f} {improvement:>+15.4f}")

print(f"\n{'='*70}\n")

# Analysis
print("[Analysis] Graph Signal Detection:")
print(f"  HAR OLS baseline R2 = {baseline_r2:+.4f}")

if metrics_pearson['r2'] > baseline_r2:
    print(f"  [POSITIVE] Pearson graph improves R2 by {(metrics_pearson['r2'] - baseline_r2):+.4f}")
    print(f"  >> Cross-stock spillover signal EXISTS")
    print(f"  >> Recommend: Proceed to PyTorch GHAR and GNNHAR models")
elif metrics_pearson['r2'] > baseline_r2 - 0.05:
    print(f"  [NEUTRAL] Pearson graph similar to HAR OLS (diff = {(metrics_pearson['r2'] - baseline_r2):+.4f})")
    print(f"  >> Graph signal WEAK or ABSENT")
    print(f"  >> Consider: Try different graph construction methods or stop here")
else:
    print(f"  [NEGATIVE] Pearson graph WORSE than HAR OLS (diff = {(metrics_pearson['r2'] - baseline_r2):+.4f})")
    print(f"  >> Graph implementation issue or no spillover signal")
    print(f"  >> Action: Debug graph transformation or reconsider approach")

if metrics_glasso['r2'] > baseline_r2:
    print(f"  [POSITIVE] GLASSO graph improves R2 by {(metrics_glasso['r2'] - baseline_r2):+.4f}")

# GHAR identity should match HAR OLS
iden_diff = abs(metrics_iden['r2'] - baseline_r2)
if iden_diff < 0.01:
    print(f"\n[Validation] GHAR(identity) approx HAR OLS (diff = {iden_diff:.4f}) OK")
    print(f"  >> Implementation CORRECT (identity graph = HAR baseline)")
else:
    print(f"\n[WARNING] GHAR(identity) != HAR OLS (diff = {iden_diff:.4f})")
    print(f"  >> Implementation BUG DETECTED (should match HAR OLS)")

print(f"\n{'='*70}\n")
