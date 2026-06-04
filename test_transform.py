"""Quick test to verify identity transformation bug."""
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

# Create identity adjacency model
model = GHARSklearn(adj_method='iden', graph_end_date='2025-12-31')
returns = compute_log_returns(loader.close)

# Build adjacency matrices
model._build_adjacency_matrices(returns)

# Test transformation on first 100 samples
n_test = 100
X_sample = X_train[:n_test]
stocks_sample = stocks_train[:n_test]
dates_sample = dates_train[:n_test]

# Transform features
X_transformed = model._transform_features_by_date(X_sample, stocks_sample, dates_sample)

# For identity adjacency, transformed should equal original
print("Testing identity transformation:")
print(f"  Original shape: {X_sample.shape}")
print(f"  Transformed shape: {X_transformed.shape}")

# Compare first 10 samples
diffs = []
for i in range(min(10, n_test)):
    orig = X_sample[i, :]
    trans = X_transformed[i, :3]  # First 3 features (identity adjacency)
    diff = np.max(np.abs(orig - trans))
    diffs.append(diff)
    print(f"  Sample {i}: max_diff = {diff:.6f}")

max_diff = np.max(diffs)
print(f"\n  Result: max difference = {max_diff:.6f}")

if max_diff < 1e-5:
    print("  [OK] Identity transformation is CORRECT")
else:
    print(f"  [BUG] Identity transformation is BROKEN (diff = {max_diff:.6f})")
    print(f"  >> This explains why GHAR(identity) != HAR OLS")
