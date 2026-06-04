"""Debug sklearn GHAR transformation in detail."""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.ghar_sklearn import GHARSklearn
from src.volatility_labels import compute_log_returns
from gnn.build_graph import VN30_TICKERS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("="*70)
print("DEBUG: sklearn GHAR Transformation")
print("="*70)

# Load data
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

print(f"\n[Data shapes]")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")

# =============================================================================
# TEST 1: Verify sample ordering is consistent
# =============================================================================

print(f"\n{'='*70}")
print("TEST 1: Check if samples are ordered by date")
print(f"{'='*70}")

# Check if first 10 samples are from the same date or different
print(f"\nFirst 10 training samples:")
for i in range(10):
    print(f"  Sample {i}: stock={stocks_train[i]}, date={dates_train[i]}")

# Check if samples are grouped by date
unique_dates_in_first_100 = len(np.unique(dates_train[:100]))
print(f"\nUnique dates in first 100 samples: {unique_dates_in_first_100}")

if unique_dates_in_first_100 == 1:
    print("  [INFO] First 100 samples are from the SAME date (grouped)")
else:
    print("  [INFO] First 100 samples are from DIFFERENT dates (interleaved)")

# =============================================================================
# TEST 2: Manual HAR OLS with SAME model as GHAR (single shared model)
# =============================================================================

print(f"\n{'='*70}")
print("TEST 2: Compare HAR OLS approaches")
print(f"{'='*70}")

# Approach 1: Per-stock HAR OLS (current baseline, working well)
print(f"\n[Approach 1] Per-stock HAR OLS (30 separate models)")
models_per_stock = {}
for stock_id in np.unique(stocks_train):
    mask = (stocks_train == stock_id)
    if mask.sum() == 0:
        continue
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train[mask], y_train[mask])
    models_per_stock[stock_id] = model

# Predict with per-stock models
y_pred_per_stock = np.zeros_like(y_test)
for stock_id, model in models_per_stock.items():
    mask = (stocks_test == stock_id)
    if mask.any():
        y_pred_per_stock[mask] = model.predict(X_test[mask])

r2_per_stock = r2_score(y_test, y_pred_per_stock)
print(f"  Test R2: {r2_per_stock:+.4f}")

# Approach 2: Single shared model for all stocks (what GHAR is doing)
print(f"\n[Approach 2] Single shared HAR model (1 model for all stocks)")
model_shared = LinearRegression(fit_intercept=True)
model_shared.fit(X_train, y_train)
y_pred_shared = model_shared.predict(X_test)
r2_shared = r2_score(y_test, y_pred_shared)
print(f"  Test R2: {r2_shared:+.4f}")

print(f"\n[Comparison]")
print(f"  Per-stock R2: {r2_per_stock:+.4f}")
print(f"  Shared-model R2: {r2_shared:+.4f}")
print(f"  Difference: {r2_shared - r2_per_stock:+.4f}")

if abs(r2_shared - r2_per_stock) > 0.5:
    print(f"\n  [CRITICAL] Single shared model is MUCH WORSE than per-stock models")
    print(f"  >> This explains why GHAR (shared) != HAR OLS (per-stock)")

# =============================================================================
# TEST 3: Does identity transformation preserve features?
# =============================================================================

print(f"\n{'='*70}")
print("TEST 3: Verify identity transformation doesn't change features")
print(f"{'='*70}")

model_ghar = GHARSklearn(adj_method='iden', graph_end_date='2025-12-31')
returns = compute_log_returns(loader.close)
model_ghar._build_adjacency_matrices(returns)

# Transform training features
X_train_transformed = model_ghar._transform_features_by_date(X_train, stocks_train, dates_train)

# Check if transformed equals original (should for identity)
max_diff = np.max(np.abs(X_train_transformed - X_train))
print(f"\nMax difference between transformed and original: {max_diff:.10f}")

if max_diff < 1e-6:
    print(f"  [OK] Identity transformation preserves features (max_diff={max_diff:.10f})")
else:
    print(f"  [BUG] Identity transformation CHANGES features (max_diff={max_diff:.10f})!")

# Fit model on transformed features
model_shared_transformed = LinearRegression(fit_intercept=True)
model_shared_transformed.fit(X_train_transformed, y_train)

# Transform test features
X_test_transformed = model_ghar._transform_features_by_date(X_test, stocks_test, dates_test)
y_pred_transformed = model_shared_transformed.predict(X_test_transformed)

r2_transformed = r2_score(y_test, y_pred_transformed)
print(f"\nTest R2 with identity-transformed features: {r2_transformed:+.4f}")

print(f"\n[Conclusion]")
if abs(r2_transformed - r2_shared) < 0.01:
    print(f"  >> Identity transformation gives SAME result as shared model")
    print(f"  >> Transformation is correct, but shared model is the problem")
else:
    print(f"  >> BUG: Identity transformation changes model behavior!")

# =============================================================================
# TEST 4: Check transformation sample ordering
# =============================================================================

print(f"\n{'='*70}")
print("TEST 4: Does transformation preserve sample ordering?")
print(f"{'='*70}")

# Pick a specific sample and track it
sample_idx = 0
original_stock = stocks_train[sample_idx]
original_date = dates_train[sample_idx]
original_X = X_train[sample_idx].copy()
original_y = y_train[sample_idx]

print(f"\nTracking sample {sample_idx}:")
print(f"  Stock: {original_stock}, Date: {original_date}")
print(f"  Original X: {original_X}")
print(f"  Original y: {original_y}")

# After transformation
transformed_X = X_train_transformed[sample_idx]
print(f"  Transformed X: {transformed_X}")

# Find this sample in the transformed DataFrame
df = pd.DataFrame({
    'date': dates_train,
    'stock': stocks_train,
    'y': y_train,
})
for i in range(3):
    df[f'RV_{i}'] = X_train[:, i]

# Group by date and find the specific stock-date pair
grouped = df.groupby('date')
found_sample = None
for date, group in grouped:
    if date == original_date:
        sample_in_group = group[group['stock'] == original_stock]
        if len(sample_in_group) > 0:
            found_sample = sample_in_group.iloc[0]
            break

if found_sample is not None:
    print(f"\n  Found sample in grouped DataFrame:")
    print(f"    X from DF: {found_sample[['RV_0', 'RV_1', 'RV_2']].values}")
    print(f"    y from DF: {found_sample['y']}")
else:
    print(f"\n  [ERROR] Could not find sample in grouped DataFrame!")

print(f"\n{'='*70}\n")
