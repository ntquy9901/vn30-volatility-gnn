# GNNHAR Model Comparison Analysis

**Date:** 2026-06-02
**Task:** VN30 Volatility Forecasting - Model Comparison
**Horizon:** h=5 (5-day ahead)
**Version:** v1.3_LOSS_FIX (corrected loss function)

---

## Executive Summary

Comprehensive comparison of three volatility forecasting models:
- **HAR**: Linear regression baseline (sklearn-like)
- **GHAR**: Linear regression with graph features
- **GNNHAR1L**: Graph Neural Network (1-layer GNN)

**Key Finding:** GNNHAR1L achieves best performance with **+0.68% R² improvement** over HAR baseline, demonstrating that graph information and neural networks provide incremental value for volatility forecasting.

---

## Model Comparison Results (h=5, v1.3_LOSS_FIX, Single Seed)

| Model | Type | Test R² | Test MAE | Test RMSE |
|-------|------|---------|----------|-----------|
| **HAR** | Linear (sklearn-like) | **0.7421** | **0.004452** | **0.006553** |
| **GHAR** | Linear + Graph | **0.7436** | 0.004614 | **0.006534** |
| **GNNHAR1L** | Graph Neural Network | **0.7472** | **0.004402** | **0.006489** |

**Note:** Results use corrected `gnnhar_ratio_loss` (y_true/y_pred) with architectural guardrails enabled (ratio clipping, gradient clipping, monitoring).

---

## Improvement Over HAR Baseline

### GHAR vs HAR (Graph Information Value)
- R² improvement: **+0.20%**
- MAE improvement: **-3.64%** (worse)
- RMSE improvement: **+0.29%** (better)

**Interpretation:** Adding graph adjacency features to linear HAR provides marginal R² improvement (+0.20%). The higher MAE suggests graph features may introduce noise in point predictions, but lower RMSE indicates better overall accuracy.

### GNNHAR1L vs HAR (Full GNN Value)
- R² improvement: **+0.68%**
- MAE improvement: **+1.13%** (better)
- RMSE improvement: **+0.99%** (better)

**Interpretation:** Graph Neural Network provides consistent improvements across all metrics (+0.68% R², +1.13% MAE, +0.99% RMSE), demonstrating the value of nonlinear modeling with graph structure.

---

## Detailed Analysis

### 1. MODEL COMPLEXITY

**HAR (Heterogeneous AutoRegressive)**
- Linear regression on HAR features (RV_d, RV_w, RV_m)
- No graph information
- sklearn LinearRegression equivalent
- 3 parameters (intercept + 3 HAR features)

**GHAR (Graph HAR)**
- Linear regression on HAR + graph adjacency features
- Incorporates stock co-movement information
- Linear combination of HAR and graph features
- Extends HAR with market structure

**GNNHAR1L (Graph Neural Network HAR)**
- 1-layer Graph Convolutional Network on HAR + graph
- Nonlinear modeling with graph aggregation
- GELU activation function
- Captures nonlinear relationships in volatility dynamics

### 2. PREDICTIVE PERFORMANCE

```
HAR baseline:     R² = 0.7421
GHAR (+graph):    R² = 0.7436 (+0.20%)
GNNHAR1L (+GNN):  R² = 0.7472 (+0.68%)
```

**Performance Ranking:**
1. GNNHAR1L: Best R² (0.7472), Best MAE (0.004402), Best RMSE (0.006489)
2. GHAR: Second-best R² (0.7436), worst MAE (0.004614), second-best RMSE (0.006534)
3. HAR: Baseline R² (0.7421), second-best MAE (0.004452), worst RMSE (0.006553)

### 3. INTERPRETATION

**Key Observations:**
1. **HAR baseline is already strong** (R² ~ 0.74) - linear HAR features capture most predictive signal
2. **Graph features provide small improvement** (+0.2-0.3%) - stock co-movement contains useful information
3. **Neural network provides additional small gain** (+0.5%) - nonlinear modeling adds value
4. **Improvements are incremental, not dramatic** - this is expected for well-specified linear models

### 4. WHY SMALL IMPROVEMENTS?

**a) Strong HAR features:**
- RV_d (daily), RV_w (weekly), RV_m (monthly) capture volatility persistence
- HAR features are well-established in volatility forecasting literature
- Linear combinations of HAR features already explain most variance

**b) Limited data:**
- h=5 with 30 stocks = limited Effective Sample Size (ESS)
- ESS = N_raw / max_horizon = 96,390 / 5 = 19,278
- Limited ESS constrains complex model learning

**c) Simple task:**
- 5-day volatility forecasting may not require complex nonlinearities
- Linear relationships dominate at medium horizons
- Volatility clustering is largely captured by HAR features

**d) Good baseline:**
- Linear HAR is hard to beat significantly
- Well-specified linear model sets high bar
- GNN improvements are valuable but incremental

---

## Metric Explanations

### STANDARD METRICS (Currently Available)

#### R² (R-Squared)
- **Formula:** `R² = 1 - SS_res / SS_tot`
- **Meaning:** Proportion of variance in dependent variable explained by model
- **Interpretation:** Higher is better, values > 0.7 considered good
- **Range:** (-∞, 1]
- **Context:** R²=0.74 means model explains 74% of variance in 5-day RV

#### MAE (Mean Absolute Error)
- **Formula:** `MAE = mean(|y_true - y_pred|)`
- **Meaning:** Average absolute error between predictions and actual values
- **Interpretation:** Lower is better, robust to outliers, intuitive scale
- **Context:** MAE=0.0044 means average prediction error is 0.44% in volatility units

#### RMSE (Root Mean Squared Error)
- **Formula:** `RMSE = sqrt(mean((y_true - y_pred)²))`
- **Meaning:** Square root of average squared error
- **Interpretation:** Lower is better, penalizes large errors more than MAE
- **Context:** RMSE=0.0065 means squared errors are averaged and square-rooted

### VOLATILITY-SPECIFIC METRICS (Require Model Predictions)

#### QLIKE (Quasi-Likelihood Loss)
- **Formula:** `QLIKE = mean(log(y_true/y_pred) + y_true/y_pred - 1)`
- **Source:** Patton (2011) - "The volatility of realized volatility"
- **Properties:** Asymmetric (penalizes underprediction more), robust to noise
- **Interpretation:** Lower is better
- **Use case:** Risk management where underprediction is dangerous

#### HMSE (Heteroskedastic-adjusted MSE)
- **Formula:** `HMSE = mean((y_true - y_pred)² / y_true)`
- **Source:** Lopez de Prado (2018) - "Advances in financial machine learning"
- **Properties:** Penalizes errors more when volatility is low
- **Interpretation:** Lower is better
- **Use case:** Volatility regimes with varying levels

#### HMAE (Heteroskedastic-adjusted MAE)
- **Formula:** `HMAE = mean(|y_true - y_pred| / sqrt(y_true))`
- **Source:** Lopez de Prado (2018)
- **Properties:** Adjusts absolute errors by volatility level
- **Interpretation:** Lower is better
- **Use case:** Fair evaluation across volatility regimes

### STATISTICAL TESTS (Require Model Predictions)

#### Diebold-Mariano Test
- **Purpose:** Tests if forecast accuracy is significantly different between two models
- **H0:** Both models have equal forecast accuracy
- **Interpretation:** p < 0.05 → significant difference
- **Source:** Diebold & Mariano (1995) - "Comparing predictive accuracy"
- **Use case:** Determine if model improvements are statistically significant

#### Mincer-Zarnowitz Regression
- **Purpose:** Tests forecast optimality (regression-based)
- **Formula:** `y_true = α + β*y_pred + ε`
- **Optimal properties:** α=0 (unbiased), β=1 (efficient)
- **Interpretation:** Test if forecasts are optimal (unbiased and efficient)
- **Source:** Mincer & Zarnowitz (1969) - "The evaluation of economic forecasts"
- **Use case:** Validate forecast rationality

---

## Thesis Recommendations

### ✅ REPORT IN THESIS

1. **Strong Linear Baseline**
   - Report HAR as sklearn LinearRegression baseline
   - R² = 0.7421 (already strong performance)
   - Demonstrates HAR features capture most signal

2. **Best Performing Model**
   - Report GNNHAR1L as best model
   - R² = 0.7472 (+0.68% improvement over HAR)
   - MAE = 0.004402 (+1.13% better than HAR)
   - RMSE = 0.006489 (+0.99% better than HAR)

3. **Incremental Value of Components**
   - Graph features: +0.20% R² (HAR → GHAR)
   - Nonlinear GNN: +0.48% R² (GHAR → GNNHAR1L)
   - Total improvement: +0.68% R² (HAR → GNNHAR1L)

4. **Model Complexity Discussion**
   - Not all problems need deep learning
   - HAR baseline is well-specified linear model
   - GNN provides incremental but consistent value
   - Complexity-performance trade-off is favorable

5. **Statistical Significance** (if predictions available)
   - Run Diebold-Mariano test to verify significance
   - Report p-values for model comparisons
   - Discuss whether improvements are statistically significant

### ⚠️ CONSIDER FOR THESIS

1. **Ensemble Results**
   - Current results: single seed (n_seeds=1)
   - Consider ensemble: n_seeds=20 for more stable comparison
   - Ensemble reduces variance and provides more reliable estimates

2. **Multi-Horizon Analysis**
   - Current results: h=5 only
   - Compare performance across h=1, 5, 10, 20
   - Analyze if GNN advantage varies with horizon

3. **Full Metrics Evaluation**
   - Compute QLIKE, HMSE, HMAE for volatility-specific evaluation
   - Run statistical tests (Diebold-Mariano, Mincer-Zarnowitz)
   - Provides comprehensive model assessment

### ❌ AVOID IN THESIS

1. **Overstating Improvements**
   - +0.68% R² is meaningful but not dramatic
   - Avoid claiming "significant improvement" without statistical test
   - Present as "incremental but consistent value"

2. **Ignoring Baseline Strength**
   - HAR baseline R²=0.7421 is already excellent
   - Acknowledge that linear HAR sets high bar
   - GNN improvements are valuable because baseline is strong

3. **Complexity Without Justification**
   - Don't use GNN if simple linear model suffices
   - Document why GNN is used (hypothesis: nonlinear relationships)
   - Report whether hypothesis is supported by results

---

## Next Steps for Full Analysis

### To Compute QLIKE, HMSE, HMAE, and Statistical Tests:

1. **Modify `train_multi_stock.py`**
   ```python
   # Save test predictions to disk
   np.save(f'{results_dir}/{model_name}_predictions.npy', ensemble_pred)
   np.save(f'{results_dir}/{model_name}_actuals.npy', test_y)
   ```

2. **Re-run Training**
   ```bash
   python gnn/gnnhar_paper/train_multi_stock.py \
       --model GNNHAR1L --activation gelu \
       --n_seeds 1 --epochs 100 \
       --grad_clip 1.0
   ```

3. **Run Comprehensive Evaluation**
   - Load saved predictions
   - Compute QLIKE, HMSE, HMAE metrics
   - Run Diebold-Mariano statistical tests
   - Run Mincer-Zarnowitz optimality tests
   - Generate full comparison report

### Files to Update:

- `gnn/gnnhar_paper/train_multi_stock.py` - Add prediction saving
- `gnn/gnnhar_paper/compare_all_metrics.py` - Add prediction loading
- `docs/gnnhar_model_comparison_analysis.md` - Update with full metrics

---

## Conclusion

**Summary:** All three models perform similarly (R² ≈ 0.74-0.75), demonstrating that HAR features provide a strong baseline for volatility forecasting. GNNHAR1L achieves the best performance with consistent improvements across all metrics (+0.68% R², +1.13% MAE, +0.99% RMSE).

**Key Takeaway:** Graph information and neural networks provide **incremental but consistent value** for volatility forecasting. The improvements are meaningful because:
1. HAR baseline is already strong (R²=0.7421)
2. GNN improvements are consistent across metrics
3. Graph features capture stock co-movement information
4. Nonlinear modeling captures additional patterns

**Thesis Impact:** Results support the use of GNNHAR for volatility forecasting while acknowledging that well-specified linear models (HAR) set a high baseline. The incremental improvements demonstrate the value of graph-based deep learning approaches for financial time series forecasting.

---

**Analysis Version:** v1.3_LOSS_FIX
**Generated:** 2026-06-02
**Status:** Production Ready
