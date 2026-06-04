"""
Test sklearn GHAR with RESIDUAL DESIGN (original + graph features).

This tests the CORRECT implementation from the paper:
- 'iden+pearson' creates 6 features: 3 original + 3 graph-augmented
- Model learns to balance local HAR vs cross-stock spillover

Expected: Should beat HAR OLS baseline if graph signal exists.
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
print("  TEST: SKLEARN GHAR WITH RESIDUAL DESIGN")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("[Data] Loading multi-stock data...")
loader = MultiStockDataLoader(
    tickers=VN30_TICKERS,
    horizon=5,
    train_end="2025-12-31",
    test_start="2026-01-01",
)
loader.load_data()
loader.build_features()
loader.flatten_dataset()
loader.split_train_val_test()

X_train, y_train, stocks_train, dates_train, X_test, y_test, stocks_test, dates_test = loader.prepare_sklearn_data()

returns = compute_log_returns(loader.close)

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")

# =============================================================================
# BASELINE: HAR OLS (PER-STOCK)
# =============================================================================

print(f"\n{'='*70}")
print("  BASELINE: HAR OLS (PER-STOCK LINEAR REGRESSION)")
print(f"{'='*70}")

models_per_stock = {}
for stock_id in np.unique(stocks_train):
    mask = (stocks_train == stock_id)
    if mask.sum() == 0:
        continue
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(X_train[mask], y_train[mask])
    models_per_stock[stock_id] = model

y_pred_har_ols = np.zeros_like(y_test)
for stock_id, model in models_per_stock.items():
    mask = (stocks_test == stock_id)
    if mask.any():
        y_pred_har_ols[mask] = model.predict(X_test[mask])

y_pred_har_ols = np.maximum(y_pred_har_ols, 0.0)

r2_har_ols = r2_score(y_test, y_pred_har_ols)
mae_har_ols = mean_absolute_error(y_test, y_pred_har_ols)

print(f"\nHAR OLS Results:")
print(f"  R2:   {r2_har_ols:+.4f}")
print(f"  MAE:  {mae_har_ols:.6f}")

# =============================================================================
# TEST 1: IDENTITY ONLY (BASELINE, SHOULD MATCH HAR OLS)
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 1: GHAR (IDENTITY ONLY - 3 FEATURES)")
print(f"{'='*70}")

model_iden = GHARSklearn(adj_method='iden', graph_end_date='2025-12-31')
model_iden.fit(X_train, y_train, stocks_train, dates_train, returns)
y_pred_iden = model_iden.predict(X_test, stocks_test, dates_test)
metrics_iden = model_iden.evaluate(y_test, y_pred_iden)

print(f"\nGHAR (identity only) Results:")
print(f"  R2:   {metrics_iden['r2']:+.4f}")
print(f"  MAE:  {metrics_iden['mae']:.6f}")

# =============================================================================
# TEST 2: RESIDUAL DESIGN - IDEN + PEARSON (6 FEATURES)
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 2: GHAR (RESIDUAL: IDEN + PEARSON - 6 FEATURES)")
print(f"{'='*70}")

# Test with threshold=0.7 (8% density, works best from diagnostics)
model_residual = GHARSklearn(
    adj_method='iden+pearson',
    threshold=0.7,
    graph_end_date='2025-12-31',
)
model_residual.fit(X_train, y_train, stocks_train, dates_train, returns)
y_pred_residual = model_residual.predict(X_test, stocks_test, dates_test)
metrics_residual = model_residual.evaluate(y_test, y_pred_residual)

print(f"\nGHAR (residual: iden+pearson, thresh=0.7) Results:")
print(f"  R2:   {metrics_residual['r2']:+.4f}")
print(f"  MAE:  {metrics_residual['mae']:.6f}")

# =============================================================================
# TEST 3: RESIDUAL DESIGN WITH DIFFERENT THRESHOLDS
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 3: RESIDUAL DESIGN WITH DIFFERENT PEARSON THRESHOLDS")
print(f"{'='*70}")

thresholds = [0.3, 0.5, 0.7, 0.8]
residual_results = []

for thresh in thresholds:
    model = GHARSklearn(
        adj_method='iden+pearson',
        threshold=thresh,
        graph_end_date='2025-12-31',
    )
    model.fit(X_train, y_train, stocks_train, dates_train, returns)
    y_pred = model.predict(X_test, stocks_test, dates_test)
    metrics = model.evaluate(y_test, y_pred)

    residual_results.append({
        'threshold': thresh,
        'r2': metrics['r2'],
        'mae': metrics['mae'],
    })

    print(f"Threshold {thresh}: R2 = {metrics['r2']:+.4f}, MAE = {metrics['mae']:.6f}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print("  SUMMARY: RESIDUAL DESIGN RESULTS")
print(f"{'='*70}\n")

print(f"Baseline Comparison:")
print(f"{'Model':<40} {'R2':>10} {'MAE':>12} {'Improvement':>15}")
print(f"{'-'*75}")

baseline_r2 = r2_har_ols

print(f"{'HAR OLS (per-stock)':<40} {r2_har_ols:+10.4f} {mae_har_ols:>12.6f} {0:+15.4f}")
print(f"{'GHAR (identity only)':<40} {metrics_iden['r2']:+10.4f} {metrics_iden['mae']:>12.6f} {(metrics_iden['r2']-baseline_r2):+15.4f}")
print(f"{'GHAR (iden+pearson, thresh=0.7)':<40} {metrics_residual['r2']:+10.4f} {metrics_residual['mae']:>12.6f} {(metrics_residual['r2']-baseline_r2):+15.4f}")

print(f"\nThreshold Analysis (residual design iden+pearson):")
print(f"{'Threshold':<15} {'R2':>10} {'vs Baseline':>15}")
print(f"{'-'*40}")
for res in residual_results:
    thresh = res['threshold']
    r2 = res['r2']
    diff = r2 - baseline_r2
    print(f"{thresh:<15} {r2:+10.4f} {diff:+15.4f}")

# Find best threshold
best_residual = max(residual_results, key=lambda x: x['r2'])
print(f"\nBest threshold: {best_residual['threshold']} (R2 = {best_residual['r2']:+.4f})")

print(f"\n[Analysis] Graph Signal Detection:")
if best_residual['r2'] > baseline_r2 + 0.01:
    print(f"  [POSITIVE] Residual design BEATS HAR OLS by {best_residual['r2'] - baseline_r2:.4f}")
    print(f"  >> Cross-stock spillover signal EXISTS")
    print(f"  >> Graph transformation ADDS predictive power")
    print(f"  >> Recommendation: Use threshold={best_residual['threshold']}")
elif best_residual['r2'] > baseline_r2 - 0.01:
    print(f"  [NEUTRAL] Residual design similar to HAR OLS (diff = {best_residual['r2'] - baseline_r2:.4f})")
    print(f"  >> Graph signal WEAK or marginal")
    print(f"  >> Recommendation: Stick with HAR OLS (simpler)")
else:
    print(f"  [NEGATIVE] Residual design WORSE than HAR OLS (diff = {best_residual['r2'] - baseline_r2:.4f})")
    print(f"  >> Graph signal ABSENT or implementation still has issues")
    print(f"  >> Recommendation: Use HAR OLS baseline")

print(f"\n{'='*70}\n")
