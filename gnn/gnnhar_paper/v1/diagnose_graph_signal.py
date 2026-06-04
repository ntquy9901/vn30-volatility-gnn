"""
Diagnostic script to investigate why Pearson graph hurts performance.

Tests:
1. Graph construction quality (threshold, density, edge weights)
2. Feature transformation verification (non-identity graphs)
3. Different Pearson thresholds (0.1, 0.3, 0.5, 0.7)
4. Graph signal strength analysis
5. Comparison with random graph (control)

Goal: Determine if graph signal exists or if implementation has bugs.
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
from gnn.gnnhar_paper.graph_builder import GraphBuilder, build_identity_adjacency
from src.volatility_labels import compute_log_returns
from gnn.build_graph import VN30_TICKERS

print("\n" + "="*70)
print("  DIAGNOSTIC: WHY DOES PEARSON GRAPH HURT PERFORMANCE?")
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
# TEST 1: GRAPH CONSTRUCTION QUALITY
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 1: GRAPH CONSTRUCTION QUALITY (DIFFERENT THRESHOLDS)")
print(f"{'='*70}")

thresholds = [0.1, 0.3, 0.5, 0.7]
graph_stats = []

for thresh in thresholds:
    builder = GraphBuilder(method='pearson', threshold=thresh, corr_window=60)
    adj = builder.build_adjacency(returns, end_date='2025-12-31')

    n_edges = (adj > 0).sum() - np.trace(adj > 0)  # Exclude self-loops
    density = n_edges / (30 * 29)  # Max edges excluding self-loops
    avg_weight = adj[adj > 0].mean()
    max_weight = adj.max()

    graph_stats.append({
        'threshold': thresh,
        'density': density,
        'n_edges': n_edges,
        'avg_weight': avg_weight,
        'max_weight': max_weight,
    })

    print(f"\nThreshold {thresh}:")
    print(f"  Density: {density:.2%}")
    print(f"  Edges: {n_edges}")
    print(f"  Avg edge weight: {avg_weight:.4f}")
    print(f"  Max edge weight: {max_weight:.4f}")

# =============================================================================
# TEST 2: FEATURE TRANSFORMATION VERIFICATION
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 2: VERIFY FEATURE TRANSFORMATION (NON-IDENTITY GRAPH)")
print(f"{'='*70}")

# Build Pearson graph with threshold=0.3
builder = GraphBuilder(method='pearson', threshold=0.3, corr_window=60)
adj_pearson = builder.build_adjacency(returns, end_date='2025-12-31')

# Create model to access transformation
model = GHARSklearn(adj_method='pearson', threshold=0.3, graph_end_date='2025-12-31')
model._build_adjacency_matrices(returns)

# Transform a small sample
n_sample = 1000
X_sample = X_train[:n_sample]
stocks_sample = stocks_train[:n_sample]
dates_sample = dates_train[:n_sample]

X_transformed = model._transform_features_by_date(X_sample, stocks_sample, dates_sample)

print(f"\nFeature transformation check:")
print(f"  Original shape: {X_sample.shape}")
print(f"  Transformed shape: {X_transformed.shape}")
print(f"  Mean change per feature:")
for i in range(3):
    mean_orig = X_sample[:, i].mean()
    mean_trans = X_transformed[:, i].mean()
    change_pct = abs(mean_trans - mean_orig) / mean_orig * 100 if mean_orig > 0 else 0
    print(f"    RV_{i}: {mean_orig:.6f} -> {mean_trans:.6f} ({change_pct:+.1f}%)")

# Check if transformation changed features (should for Pearson)
max_diff = np.max(np.abs(X_transformed - X_sample))
print(f"\n  Max absolute difference: {max_diff:.6f}")

if max_diff > 0.01:
    print(f"  [OK] Pearson graph CHANGES features (as expected)")
else:
    print(f"  [WARN] Pearson graph does NOT change features significantly")

# =============================================================================
# TEST 3: PERFORMANCE WITH DIFFERENT THRESHOLDS
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 3: TEST PERFORMANCE WITH DIFFERENT PEARSON THRESHOLDS")
print(f"{'='*70}")

results = []

# Test identity baseline (reference)
model_iden = GHARSklearn(adj_method='iden', graph_end_date='2025-12-31')
model_iden._build_adjacency_matrices(returns)
X_train_iden = model_iden._transform_features_by_date(X_train, stocks_train, dates_train)
X_test_iden = model_iden._transform_features_by_date(X_test, stocks_test, dates_test)

model_lr = LinearRegression(fit_intercept=True, n_jobs=-1)
model_lr.fit(X_train_iden, y_train)
y_pred_iden = model_lr.predict(X_test_iden)

r2_iden = r2_score(y_test, y_pred_iden)
mae_iden = mean_absolute_error(y_test, y_pred_iden)

results.append({'method': 'identity', 'threshold': 'N/A', 'R2': r2_iden, 'MAE': mae_iden})
print(f"\nIdentity: R2 = {r2_iden:+.4f}, MAE = {mae_iden:.6f}")

# Test different Pearson thresholds
for thresh in thresholds:
    model = GHARSklearn(adj_method='pearson', threshold=thresh, graph_end_date='2025-12-31')
    model._build_adjacency_matrices(returns)

    X_train_trans = model._transform_features_by_date(X_train, stocks_train, dates_train)
    X_test_trans = model._transform_features_by_date(X_test, stocks_test, dates_test)

    model_lr = LinearRegression(fit_intercept=True, n_jobs=-1)
    model_lr.fit(X_train_trans, y_train)
    y_pred = model_lr.predict(X_test_trans)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({'method': 'pearson', 'threshold': thresh, 'R2': r2, 'MAE': mae})
    print(f"Pearson (thresh={thresh}): R2 = {r2:+.4f}, MAE = {mae:.6f}")

# =============================================================================
# TEST 4: RANDOM GRAPH CONTROL
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 4: RANDOM GRAPH CONTROL (SHOULD PERFORM LIKE BASELINE)")
print(f"{'='*70}")

# Create random adjacency matrix with same density as Pearson 0.3
np.random.seed(42)
adj_random = np.random.rand(30, 30).astype(np.float32)
adj_random = (adj_random + adj_random.T) / 2  # Make symmetric
np.fill_diagonal(adj_random, 1.0)  # Self-loops

# Normalize
adj_random = adj_random / adj_random.sum(axis=1, keepdims=True)

# Create custom model with random adjacency
model_random = GHARSklearn(adj_method='iden', graph_end_date='2025-12-31')
model_random._build_adjacency_matrices(returns)
model_random.adj_list[0] = adj_random  # Replace identity with random

X_train_rand = model_random._transform_features_by_date(X_train, stocks_train, dates_train)
X_test_rand = model_random._transform_features_by_date(X_test, stocks_test, dates_test)

model_lr = LinearRegression(fit_intercept=True, n_jobs=-1)
model_lr.fit(X_train_rand, y_train)
y_pred_rand = model_lr.predict(X_test_rand)

r2_rand = r2_score(y_test, y_pred_rand)
mae_rand = mean_absolute_error(y_test, y_pred_rand)

results.append({'method': 'random', 'threshold': 'N/A', 'R2': r2_rand, 'MAE': mae_rand})
print(f"\nRandom graph: R2 = {r2_rand:+.4f}, MAE = {mae_rand:.6f}")

# =============================================================================
# TEST 5: GRAPH SIGNAL STRENGTH ANALYSIS
# =============================================================================

print(f"\n{'='*70}")
print("  TEST 5: GRAPH SIGNAL STRENGTH ANALYSIS")
print(f"{'='*70}")

# Question: Does cross-stock correlation predict volatility correlation?

print(f"\n[Analysis 1] Stock RV correlation vs stock return correlation")

# Compute correlation matrix of returns (used for graph construction)
corr_returns = returns[returns.index <= pd.Timestamp('2025-12-31')].iloc[-60:].corr()

# Compute correlation matrix of RV (target variable)
rv_df = loader.rv[loader.rv.index <= pd.Timestamp('2025-12-31')].iloc[-60:]
corr_rv = rv_df.corr()

# Compare correlations
n_stocks = len(VN30_TICKERS)
corr_returns_vals = []
corr_rv_vals = []

for i in range(n_stocks):
    for j in range(i+1, n_stocks):
        t1, t2 = VN30_TICKERS[i], VN30_TICKERS[j]
        if t1 in corr_returns.columns and t2 in corr_returns.columns:
            c_ret = corr_returns.loc[t1, t2]
            c_rv = corr_rv.loc[t1, t2] if t1 in corr_rv.columns and t2 in corr_rv.columns else np.nan
            if not np.isnan(c_ret) and not np.isnan(c_rv):
                corr_returns_vals.append(abs(c_ret))
                corr_rv_vals.append(abs(c_rv))

corr_returns_vals = np.array(corr_returns_vals)
corr_rv_vals = np.array(corr_rv_vals)

# Compute correlation between return correlations and RV correlations
if len(corr_returns_vals) > 10:
    signal_corr = np.corrcoef(corr_returns_vals, corr_rv_vals)[0, 1]
    print(f"\nCorrelation between |return corr| and |RV corr|: {signal_corr:+.4f}")

    if signal_corr > 0.3:
        print(f"  [STRONG] Stocks with correlated returns ALSO have correlated volatility")
        print(f"  >> Graph signal EXISTS")
    elif signal_corr > 0.1:
        print(f"  [WEAK] Some relationship between return corr and RV corr")
        print(f"  >> Graph signal WEAK")
    else:
        print(f"  [NONE] No relationship between return corr and RV corr")
        print(f"  >> Graph signal ABSENT (return correlation does not predict volatility spillover)")

# =============================================================================
# SUMMARY AND CONCLUSIONS
# =============================================================================

print(f"\n{'='*70}")
print("  SUMMARY: DIAGNOSTIC RESULTS")
print(f"{'='*70}\n")

print(f"Performance comparison:")
print(f"{'Method':<15} {'Threshold':<12} {'R2':>10} {'MAE':>12} {'vs Identity':>15}")
print(f"{'-'*60}")

for res in results:
    method = res['method']
    thresh = str(res['threshold'])
    r2 = res['R2']
    mae = res['MAE']
    diff = r2 - r2_iden
    print(f"{method:<15} {thresh:<12} {r2:>+10.4f} {mae:>12.6f} {diff:>+15.4f}")

print(f"\n{'='*70}")

# Analysis
print(f"\n[Conclusions]")

# Find best threshold
pearson_results = [r for r in results if r['method'] == 'pearson']
if pearson_results:
    best_pearson = max(pearson_results, key=lambda x: x['R2'])
    if best_pearson['R2'] > r2_iden + 0.01:
        print(f"  [POSITIVE] Best Pearson graph (thresh={best_pearson['threshold']}) BEATS identity by {best_pearson['R2'] - r2_iden:.4f}")
        print(f"  >> Graph signal EXISTS with optimal threshold")
        print(f"  >> Recommendation: Use threshold={best_pearson['threshold']}")
    elif best_pearson['R2'] > r2_iden - 0.02:
        print(f"  [NEUTRAL] Best Pearson graph (thresh={best_pearson['threshold']}) similar to identity")
        print(f"  >> Graph signal WEAK or marginal")
        print(f"  >> Recommendation: Stick with identity (HAR OLS)")
    else:
        print(f"  [NEGATIVE] All Pearson graphs WORSE than identity")
        print(f"  >> Graph signal ABSENT or implementation still has bugs")
        print(f"  >> Recommendation: Do NOT use Pearson graph")

# Compare with random graph
if abs(r2_rand - r2_iden) < 0.05:
    print(f"\n  [INFO] Random graph performs similarly to identity (diff={abs(r2_rand - r2_iden):.4f})")
    print(f"  >> Expected: random graph should add noise and hurt performance")
    print(f"  >> Current behavior suggests transformation might not be using graph properly")
else:
    print(f"\n  [INFO] Random graph differs from identity by {abs(r2_rand - r2_iden):.4f}")

print(f"\n{'='*70}\n")
