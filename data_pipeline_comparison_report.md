# Data Pipeline Comparison Report: HAR vs GNNHAR1L

**Date:** 2026-06-03
**Task:** Review data pipeline fairness between HAR baseline and GNNHAR1L
**Status:** CRITICAL MISCOMPARE FOUND

## Executive Summary

**VERDICT: UNFAIR COMPARISON**

The NumPy HAR baseline (`baselines/har_rv_baseline.py`) uses a **DIFFERENT training period** than both GNNHAR1L and the sklearn HAR baseline. This creates a biased comparison where models are evaluated on different data subsets.

---

## Critical Finding: Training Period Mismatch

### GNNHAR1L (train_multi_stock.py)
- **File:** `gnn/gnnhar_paper/train_multi_stock.py` (line 724)
- **Parameter:** `train_end='2025-12-31'` (default)
- **Training data:** All dates through 2025-12-31

### sklearn HAR (evaluate_sklearn_baseline.py)
- **File:** `gnn/gnnhar_paper/v1/evaluate_sklearn_baseline.py` (line 46)
- **Parameter:** `TRAIN_END = "2025-12-31"`
- **Training data:** All dates through 2025-12-31

### NumPy HAR (har_rv_baseline.py)
- **File:** `baselines/har_rv_baseline.py` (line 447)
- **Parameter:** `train_end="2024-12-31"`
- **Training data:** All dates through 2024-12-31

**Impact:** NumPy HAR is trained on **1 YEAR LESS DATA** than GNNHAR1L and sklearn HAR.

---

## Detailed Comparison

### 1. Data Loader Class

| Implementation | Data Loader | Source |
|----------------|-------------|--------|
| GNNHAR1L | `MultiStockDataLoader` | `gnn/gnnhar_paper/data_loader.py` |
| sklearn HAR | `MultiStockDataLoader` | `gnn/gnnhar_paper/data_loader.py` |
| NumPy HAR | Custom per-stock pipeline | `baselines/har_rv_baseline.py` |

**Status:** Different implementations (potential for feature computation differences)

---

### 2. Horizon Parameter

| Implementation | Horizon Value | Source |
|----------------|---------------|--------|
| GNNHAR1L | `h=5` (default) | `train_multi_stock.py:722` |
| sklearn HAR | `HORIZON = 5` | `evaluate_sklearn_baseline.py:45` |
| NumPy HAR | `h=5` (multi-horizon, includes 5) | `har_rv_baseline.py:449` |

**Status:** Consistent (all use h=5)

---

### 3. Training End Date

| Implementation | Train End | Source |
|----------------|----------|--------|
| GNNHAR1L | `2025-12-31` | `train_multi_stock.py:724` |
| sklearn HAR | `2025-12-31` | `evaluate_sklearn_baseline.py:46` |
| NumPy HAR | `2024-12-31` | `har_rv_baseline.py:447` |

**Status:** MISMATCH - NumPy HAR uses 2024-12-31, others use 2025-12-31

---

### 4. Test Start Date

| Implementation | Test Start | Source |
|----------------|------------|--------|
| GNNHAR1L | `2026-01-01` | `train_multi_stock.py:726` |
| sklearn HAR | `2026-01-01` | `evaluate_sklearn_baseline.py:47` |
| NumPy HAR | `2026-01-01` | `har_rv_baseline.py:448` |

**Status:** Consistent (all use 2026-01-01)

---

### 5. Validation Split

| Implementation | Val Split | Source |
|----------------|-----------|--------|
| GNNHAR1L | 80/20 from pre-2026 | `train_multi_stock.py:772` (val_split=0.2) |
| sklearn HAR | No val split (uses train set) | `evaluate_sklearn_baseline.py:65` |
| NumPy HAR | 80/20 from pre-test | `har_rv_baseline.py:451` (val_ratio=0.2) |

**Status:** GNNHAR1L and NumPy HAR both use 80/20 split, sklearn HAR uses train set only

---

### 6. RV Computation Function

| Implementation | RV Function | Source |
|----------------|-------------|--------|
| GNNHAR1L | `compute_rv(close, h)` | `src/volatility_labels.py:50` |
| sklearn HAR | `compute_rv(close, h)` | `src/volatility_labels.py:50` |
| NumPy HAR | `compute_rv(close, h)` | `src/volatility_labels.py:50` |

**Status:** Consistent (all use same function)

---

### 7. HAR Feature Building

| Implementation | Feature Formula | Source |
|----------------|-----------------|--------|
| GNNHAR1L | `[RV_d, RV_w, RV_m]` via `MultiStockDataLoader` | `data_loader.py:107-115` |
| sklearn HAR | `[RV_d, RV_w, RV_m]` via `MultiStockDataLoader` | `data_loader.py:107-115` |
| NumPy HAR | `[const, RV_d, RV_w, RV_m]` | `har_rv_baseline.py:54-63` |

**Status:** Consistent feature computation (NumPy HAR adds constant column)

**Feature formulas:**
- `RV_d = rv.shift(1)` (daily lag)
- `RV_w = rv.shift(1).rolling(5).mean()` (weekly avg)
- `RV_m = rv.shift(1).rolling(22).mean()` (monthly avg)

---

### 8. Stock Tickers

| Implementation | Tickers | Source |
|----------------|---------|--------|
| GNNHAR1L | `VN30_TICKERS` | `train_multi_stock.py:759` |
| sklearn HAR | `VN30_TICKERS` | `evaluate_sklearn_baseline.py:55` |
| NumPy HAR | `VN30_TICKERS` | `har_rv_baseline.py:217` |

**Status:** Consistent (all use VN30_TICKERS)

---

## Impact on Fairness

### Problem 1: Different Training Periods (CRITICAL)

**NumPy HAR:** Trains on data through 2024-12-31
- Training samples: ~2,200 days per stock (pre-2025)
- Misses entire 2025 data (252 trading days)

**GNNHAR1L + sklearn HAR:** Trains on data through 2025-12-31
- Training samples: ~2,452 days per stock (pre-2026)
- Includes 2025 data (252 additional trading days)

**Impact:** This is a **11.5% increase in training data** for GNNHAR1L/sklearn HAR. The comparison is NOT fair because:

1. **Data advantage:** GNNHAR1L sees more recent market patterns (2025 data including COVID-19 recovery patterns, policy changes, etc.)
2. **Model degradation:** NumPy HAR coefficients are fitted on older data and may not capture 2025 market dynamics
3. **Invalid comparison:** Performance differences may be due to training period, not model architecture

**Expected bias:** GNNHAR1L likely shows inflated performance simply because it has more recent training data.

---

### Problem 2: Different Data Pipelines (MODERATE)

**NumPy HAR:** Per-stock independent processing
- Loops through stocks independently
- Builds features per-stock
- Fits OLS per-stock

**GNNHAR1L + sklearn HAR:** Multi-stock batch processing
- Uses `MultiStockDataLoader` class
- Flattens across stocks and dates
- Batch processing for efficiency

**Potential issues:**
1. **Feature alignment:** Need to verify both produce identical feature matrices
2. **Missing data handling:** Different implementations may handle NaNs differently
3. **Edge cases:** Stock listing/delisting dates may differ between implementations

**Status:** Requires verification (feature matrix comparison needed)

---

### Problem 3: Validation Split Handling (MINOR)

**GNNHAR1L:** Uses validation split for early stopping
- Splits training data 80/20 (temporal)
- Uses val set for convergence monitoring

**NumPy HAR:** Uses validation split for reporting only
- Splits pre-test data 80/20 (temporal)
- Reports val metrics but doesn't use them

**sklearn HAR:** No validation split
- Uses full training set for fitting
- No early stopping (linear model)

**Impact:** Minimal for HAR (linear model), but affects GNNHAR1L training dynamics

---

## Required Fixes for Fair Comparison

### Fix 1: Standardize Training Period (MANDATORY)

**Option A:** Update NumPy HAR to use 2025-12-31
```python
# In baselines/har_rv_baseline.py line 447
train_end="2025-12-31",  # Changed from "2024-12-31"
```

**Option B:** Update GNNHAR1L to use 2024-12-31
```python
# In gnn/gnnhar_paper/train_multi_stock.py line 724
parser.add_argument('--train_end', type=str, default='2024-12-31',
                    help='Training end date (YYYY-MM-DD)')
```

**Recommendation:** **Option A** - Use 2025-12-31 for all models (more recent data, better for 2026 test set)

---

### Fix 2: Verify Feature Matrix Consistency (RECOMMENDED)

Create verification script to confirm both pipelines produce identical features:

```python
# diag_compare_har_features.py
# Load RV from both pipelines
# Compare feature matrices for same stock, same date
# Assert max difference < 1e-10
```

---

### Fix 3: Standardize Validation Split Reporting (OPTIONAL)

Ensure all models report:
- Train period (dates + sample count)
- Val period (dates + sample count) if used
- Test period (dates + sample count)
- ESS (Effective Sample Size)

---

## Verification Plan

1. **Update NumPy HAR training period** to 2025-12-31
2. **Re-run NumPy HAR baseline** with new training period
3. **Compare feature matrices** between NumPy HAR and MultiStockDataLoader
4. **Re-run GNNHAR1L** if training period changes
5. **Document results** with updated training periods

---

## Conclusion

**Current comparison is UNFAIR due to training period mismatch.**

The NumPy HAR baseline uses `train_end="2024-12-31"` while GNNHAR1L uses `train_end="2025-12-31"`. This gives GNNHAR1L an 11.5% data advantage (252 additional trading days of training data), which likely inflates its performance relative to the baseline.

**Required action:** Update `baselines/har_rv_baseline.py` line 447 to use `train_end="2025-12-31"` before reporting any model comparisons.

---

## References

- `gnn/gnnhar_paper/train_multi_stock.py:724` - GNNHAR1L train_end parameter
- `gnn/gnnhar_paper/v1/evaluate_sklearn_baseline.py:46` - sklearn HAR TRAIN_END constant
- `baselines/har_rv_baseline.py:447` - NumPy HAR train_end parameter
- `gnn/gnnhar_paper/data_loader.py` - MultiStockDataLoader implementation
- `src/volatility_labels.py:50` - compute_rv function
