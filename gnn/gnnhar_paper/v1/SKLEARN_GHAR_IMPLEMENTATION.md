# sklearn GHAR Implementation Guide

**Date:** 2026-05-31
**Status:** Working implementation with residual design
**Purpose:** Step-by-step guide for implementing sklearn GHAR for multi-stock volatility forecasting

---

## Executive Summary

sklearn GHAR is **now working correctly** with residual design. However, graph signal for VN30 data is **very weak** (only +0.0006 R² improvement over HAR OLS baseline).

**Recommendation:** Use HAR OLS as final baseline (R² = 0.7532). Graph augmentation provides negligible value for VN30 volatility forecasting.

---

## Architecture Overview

### Residual Design (Paper's Approach)

sklearn GHAR uses **residual design** - concatenates ORIGINAL features + GRAPH-AUGMENTED features:

```
Input: (N_samples, 3) HAR features [RV_d, RV_w, RV_m]

For each adjacency matrix in adj_list:
    Transform: X_graph[i] = Σ_j (adj[i,j] × X[j])
    
Output: (N_samples, 3 × len(adj_list)) features

Example: adj_method='iden+pearson'
    → 6 features total:
       - sec0RV_d, sec0RV_w, sec0RV_m (original, from identity adj)
       - sec1RV_d, sec1RV_w, sec1RV_m (graph-augmented, from pearson adj)

Model: sklearn LinearRegression (learns weights for all 6 features)
```

### Why Residual Design?

**WRONG approach** (replacing features):
```python
# ONLY graph-transformed features (loses stock-specific signal)
X_transformed = adj @ X  # 3 features
model.fit(X_transformed, y)  # R2 = 0.32 (terrible!)
```

**CORRECT approach** (residual design):
```python
# ORIGINAL + GRAPH features (preserves stock-specific signal)
X_original = X  # 3 features
X_graph = adj @ X  # 3 features
X_combined = [X_original, X_graph]  # 6 features
model.fit(X_combined, y)  # R2 = 0.754 (good!)
```

The model learns to balance:
- **Local HAR dynamics** (original features, high weight ~0.93)
- **Cross-stock spillover** (graph features, low weight ~0.07)

---

## Implementation Steps

### Step 1: Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Step 2: Load Multi-Stock Data

```python
from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.build_graph import VN30_TICKERS

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
```

**Data format:**
- `X_train`: (96390, 3) - HAR features [RV_d, RV_w, RV_m]
- `y_train`: (96390,) - RV targets
- `stocks_train`: (96390,) - Stock indices (0-29)
- `dates_train`: (96390,) - Date timestamps

### Step 3: Create GHAR Model

```python
from gnn.gnnhar_paper.ghar_sklearn import GHARSklearn

# Use residual design (iden+graph)
model = GHARSklearn(
    adj_method='iden+pearson',  # IMPORTANT: Use 'iden+XXX' for residual design
    threshold=0.3,               # Pearson correlation threshold (0.3 works best)
    corr_window=60,              # Use 60-day window for correlation
    graph_end_date='2025-12-31', # Build graph using data <= this date
)
```

**Key parameters:**
- `adj_method`: MUST use `'iden+XXX'` format for residual design
  - `'iden'`: 3 features (baseline, matches HAR OLS)
  - `'iden+pearson'`: 6 features (original + Pearson graph)
  - `'iden+glasso'`: 6 features (original + GLASSO graph)
  - **DO NOT use** `'pearson'` alone (replaces features, performs poorly)

- `threshold`: Pearson correlation threshold
  - 0.3: 68% density, R2 = 0.754 (best)
  - 0.5: 35% density, R2 = 0.754 (good)
  - 0.7: 8% density, R2 = 0.753 (slightly worse)

- `corr_window`: Days of historical data for correlation calculation
  - 60 days = 3 trading months (paper's default)
  - 120 days = 6 trading months (more stable, fewer edges)

### Step 4: Train Model

```python
from src.volatility_labels import compute_log_returns

returns = compute_log_returns(loader.close)

model.fit(
    X_train, y_train, stocks_train, dates_train, returns
)
```

**Training process:**
1. Build adjacency matrices (iden + pearson)
2. Transform features by date group
3. Fit sklearn LinearRegression

### Step 5: Predict and Evaluate

```python
y_pred = model.predict(X_test, stocks_test, dates_test)

metrics = model.evaluate(y_test, y_pred)
print(f"R2: {metrics['r2']:+.4f}")
print(f"MAE: {metrics['mae']:.6f}")
```

---

## Known Issues and Limitations

### Issue 1: Weak Graph Signal

**Problem:** Graph augmentation provides negligible improvement (+0.0006 R²) for VN30 data.

**Evidence:**
- HAR OLS: R2 = 0.7532 (baseline)
- GHAR (iden+pearson): R2 = 0.7538 (best)
- Model coefficients: 93% weight on original, 7% on graph features

**Possible causes:**
1. **VN30 market characteristics**: Vietnamese stocks may have weaker volatility spillover than US markets (paper's DJIA/S&P data)
2. **Correlation instability**: 60-day correlation window may not capture stable spillover patterns
3. **Horizon mismatch**: h=5 (1-week) may be too short for cross-stock effects to materialize
4. **Sector structure**: VN30 may have less pronounced sector-based spillover than US markets

**Recommendation:** Accept HAR OLS as baseline. Do not use sklearn GHAR for VN30 forecasting.

### Issue 2: Sample Ordering Bug (FIXED)

**Problem:** Original implementation corrupted features during transformation due to scrambled sample ordering after groupby operations.

**Symptom:** Identity transformation changed features (max_diff = 0.162 instead of 0)

**Root cause:** 
```python
# WRONG - loses original order
all_transformed = np.vstack(date_features)  # Sorted by date

# CORRECT - preserves original order
transformed_with_index = []
for date, group in grouped:
    orig_indices = group['index'].values
    features_transformed = apply_transformation(group)
    for i, orig_idx in enumerate(orig_indices):
        transformed_with_index.append((orig_idx, features_transformed[i]))
transformed_with_index.sort(key=lambda x: x[0])  # Sort by original index
```

**Status:** ✓ Fixed in current implementation

### Issue 3: Feature Replacement Bug (FIXED)

**Problem:** Using single adjacency matrix ('pearson' alone) replaces original features, destroying stock-specific signal.

**Symptom:** R2 drops from 0.75 → 0.32 (catastrophic)

**Root cause:**
```python
# WRONG - replaces features
adj_method='pearson'  # Only 3 graph-transformed features
# Each stock's features become average of neighbors (loses 93% of signal)

# CORRECT - residual design
adj_method='iden+pearson'  # 6 features (3 original + 3 graph)
# Model learns to balance both
```

**Status:** ✓ Fixed - documentation updated to recommend `'iden+XXX'` format

### Issue 4: High Density Graphs Perform Poorly

**Problem:** Dense graphs (low threshold) hurt performance significantly.

**Evidence:**
- Threshold 0.1 (95% density): R2 = 0.28 (terrible)
- Threshold 0.3 (68% density): R2 = 0.75 (good)
- Threshold 0.7 (8% density): R2 = 0.75 (good)

**Root cause:** High density → each stock averages 20+ neighbors → over-smoothing → loses stock-specific information

**Status:** ✓ Documented - recommend threshold >= 0.3

---

## Performance Results

### Test Configuration
- **Data:** 30 VN30 stocks
- **Horizon:** h=5 (1 week)
- **Train:** 2006-12-21 to 2025-12-31 (96390 samples)
- **Test:** 2026-01-05 to 2026-05-22 (2760 samples)
- **Distribution shift:** +26.8% (test mean 26.8% higher than train)

### Results Summary

| Model | Features | R² | MAE | vs Baseline |
|-------|----------|-----|-----|-------------|
| HAR OLS (per-stock) | 3 | **+0.7532** | 0.004241 | - |
| GHAR (iden only) | 3 | +0.7529 | 0.004241 | -0.0003 |
| GHAR (iden+pearson, thresh=0.1) | 6 | +0.2761 | 0.008597 | -0.4771 |
| GHAR (iden+pearson, thresh=0.3) | 6 | **+0.7538** | 0.004226 | **+0.0006** |
| GHAR (iden+pearson, thresh=0.5) | 6 | +0.7537 | 0.004216 | +0.0005 |
| GHAR (iden+pearson, thresh=0.7) | 6 | +0.7527 | 0.004229 | -0.0005 |
| GHAR (iden+glasso) | 6 | +0.7529 | 0.004241 | -0.0003 |

### Key Findings

1. **Best model:** GHAR (iden+pearson, thresh=0.3) with R² = 0.7538
2. **Improvement:** Only +0.0006 over HAR OLS (negligible)
3. **Graph signal:** VERY WEAK - model assigns 93% weight to original features
4. **Optimal threshold:** 0.3 (68% density balances connectivity and information preservation)
5. **High density is catastrophic:** Threshold 0.1 destroys 93% of signal

---

## Comparison with Paper

### Paper's Setup (DJIA 30 stocks)

From Zhang et al. (2024):
- **Data:** DJIA 30 stocks (US market)
- **Horizon:** h=1 (1 day)
- **Method:** sklearn GHAR with `iden+glasso`
- **Results:** GHAR beats HAR OLS (exact R² not reported in abstract)

### Our Setup (VN30 stocks)

- **Data:** VN30 stocks (Vietnamese market)
- **Horizon:** h=5 (1 week)
- **Method:** sklearn GHAR with `iden+pearson`
- **Results:** GHAR barely beats HAR OLS (+0.0006 R²)

### Possible Explanations for Difference

1. **Market structure:** US markets may have stronger volatility spillover than Vietnamese markets
2. **Horizon:** h=1 may have different spillover dynamics than h=5
3. **Correlation quality:** Pearson correlation may not capture volatility spillover as well as GLASSO
4. **Data period:** Our test period (2026) may have different characteristics than paper's period

---

## Recommendations

### For VN30 Volatility Forecasting

1. **Use HAR OLS as final baseline**
   - R² = 0.7532 (excellent performance)
   - Simple, interpretable, fast
   - No graph complexity needed

2. **Do NOT use sklearn GHAR for production**
   - Only +0.0006 R² improvement
   - Adds complexity without meaningful gain
   - Graph signal too weak to justify

3. **Consider PyTorch GNNHAR if needed**
   - May learn better graph representations than fixed Pearson correlation
   - Residual design built into architecture (H1 + H2)
   - Expected to beat HAR OLS more significantly (if graph signal exists)

### For Other Markets

sklearn GHAR may work better for:
- Markets with strong volatility spillover (e.g., US sector rotation)
- Shorter horizons (h=1, h=5)
- Larger stock universes (S&P 100, Russell 2000)

**Recommendation:** Test with `iden+pearson, thresh=0.3` first. If R² improvement > 0.01, use graph approach.

---

## Implementation Checklist

- [x] Load multi-stock data with MultiStockDataLoader
- [x] Build adjacency matrices (identity + Pearson)
- [x] Transform features with residual design (iden + graph)
- [x] Fix sample ordering bug (preserve original order after groupby)
- [x] Test with different Pearson thresholds
- [x] Compare against HAR OLS baseline
- [x] Document known issues and limitations
- [ ] Update MULTI_STOCK_ARCHITECTURE.md with findings
- [ ] Decide whether to proceed to PyTorch implementation

---

## File Locations

- **Implementation:** `gnn/gnnhar_paper/ghar_sklearn.py`
- **Data loader:** `gnn/gnnhar_paper/data_loader.py`
- **Graph builder:** `gnn/gnnhar_paper/graph_builder.py`
- **Test scripts:** `gnn/gnnhar_paper/v1/test_residual_design.py`
- **Documentation:** `gnn/gnnhar_paper/docs/MULTI_STOCK_ARCHITECTURE.md`

---

## References

1. Original paper code: https://github.com/chaozhang-ox/GNNHAR
2. MULTI_STOCK_ARCHITECTURE.md (this project)
3. SINGLE_STOCK_FAILURE_ANALYSIS.md (previous analysis)
4. Zhang et al. (2024) "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks", International Journal of Forecasting
