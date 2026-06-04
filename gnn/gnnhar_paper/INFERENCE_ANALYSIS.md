# GNNHAR1L Inference Analysis - VIC & FPT Stocks
**Date**: 2026-06-03  
**Model**: GNNHAR1L (1-layer GCN with MLP)  
**Horizon**: h5 (5-day forward forecast)  
**Period**: 2026-01-05 to 2026-03-27 (55 trading days)

---

## 📊 Executive Summary

**Overall Performance**: **POOR** - Both stocks show negative R², indicating models perform worse than predicting the mean volatility.

| Stock | R² Score | MAE | RMSE | Assessment |
|-------|----------|-----|------|------------|
| **VIC** | **-2.54** | 0.0210 | 0.0233 | ❌ Very Poor |
| **FPT** | **-0.85** | 0.0091 | 0.0122 | ❌ Poor |

**Key Finding**: The model systematically underestimates volatility, especially during high-volatility periods.

---

## 🔍 Detailed Analysis by Stock

### VIC (Vingroup) - **Very Poor Performance**

**Metrics Analysis**:
- **R² = -2.54**: Extremely poor - 2.5× worse than predicting mean
- **MAE = 0.0210**: Average error of 2.1% absolute volatility
- **RMSE = 0.0233**: Root mean squared error 2.33%
- **RMSE > MAE**: Indicates some large prediction errors

**Volatility Range Analysis**:
```
Actual RV:    [0.006, 0.059]  → Mean: 0.033, Std: 0.015
Predicted RV: [0.013, 0.020]  → Mean: 0.017, Std: 0.002
```

**Critical Issues**:
1. **Severe underprediction**: Predicted mean (0.017) is 48% below actual mean (0.033)
2. **Narrow prediction range**: Model predicts 0.007 range vs actual 0.053 range (13× too narrow)
3. **No volatility clustering**: Model misses both high and low volatility periods
4. **Systematic bias**: Consistently predicts values that are too low

**Time Period Patterns**:
- **Jan 2026**: High volatility period (0.038-0.059) - Severely underpredicted
- **Feb 2026**: Moderate volatility (0.029-0.057) - Still underpredicted  
- **Mar 2026**: Declining volatility (0.006-0.056) - Mixed results

**Largest Errors**:
- 2026-01-29: Actual 0.059, Predicted 0.017, Error = +0.042 (underprediction)
- 2026-01-30: Actual 0.059, Predicted 0.017, Error = +0.042 (underprediction)
- 2026-03-12: Actual 0.006, Predicted 0.019, Error = -0.013 (overprediction)

---

### FPT (FPT Corporation) - **Poor Performance**

**Metrics Analysis**:
- **R² = -0.85**: Poor - 85% worse than predicting mean
- **MAE = 0.0091**: Average error of 0.91% absolute volatility
- **RMSE = 0.0122**: Root mean squared error 1.22%
- **Better than VIC**: Both MAE and R² are significantly better

**Volatility Range Analysis**:
```
Actual RV:    [0.011, 0.045]  → Mean: 0.022, Std: 0.010
Predicted RV: [0.012, 0.017]  → Mean: 0.015, Std: 0.001
```

**Issues**:
1. **Moderate underprediction**: Predicted mean (0.015) is 32% below actual mean (0.022)
2. **Narrow prediction range**: Model predicts 0.005 range vs actual 0.034 range (7× too narrow)
3. **Some overpredictions**: Unlike VIC, FPT shows both under and over-predictions
4. **Limited responsiveness**: Model fails to capture volatility dynamics

**Time Period Patterns**:
- **Jan 2026**: Mixed performance (0.012-0.034 actual)
- **Feb 2026**: Moderate underprediction
- **Mar 2026**: High volatility period (0.017-0.045) - Severely underpredicted

**Best Predictions**:
- 2026-02-06: Actual 0.016, Predicted 0.016, Error = 0.000 (nearly perfect)
- 2026-02-03: Actual 0.016, Predicted 0.017, Error = -0.001 (very close)

**Worst Predictions**:
- 2026-03-06: Actual 0.044, Predicted 0.014, Error = +0.030 (severe underprediction)
- 2026-03-05: Actual 0.045, Predicted 0.016, Error = +0.029 (severe underprediction)

---

## 🎯 Model-Specific Issues

### 1. **Prediction Range Compression**

**Problem**: Model predictions are compressed into a very narrow range.

```
VIC:  Actual range 0.053 → Predicted range 0.007 (13× compression)
FPT:  Actual range 0.034 → Predicted range 0.005 (7× compression)
```

**Cause**: 
- **Over-regularization**: Strong L2 penalty (weight_decay=1e-3) may be too aggressive
- **ReLU activation**: May cause output saturation if training on z-scored data
- **Ensemble averaging**: Multiple models with similar errors average to conservative values

### 2. **Systematic Underprediction Bias**

**Problem**: Both stocks show consistent underprediction.

```
VIC:  Mean actual 0.033 → Mean predicted 0.017 (48% bias)
FPT:  Mean actual 0.022 → Mean predicted 0.015 (32% bias)
```

**Cause**:
- **Training-test mismatch**: Training period may have different volatility characteristics
- **Conservative learning**: Model learned to predict "safe" low values
- **Asymmetric loss function**: May prefer underprediction over overprediction

### 3. **Lack of Volatility Clustering**

**Problem**: Model fails to capture heteroscedasticity (volatility clustering).

**Evidence**:
- High volatility periods (Jan 29-30, Mar 5-6) severely underpredicted
- Low volatility periods (Mar 10-12) overpredicted
- Model appears to predict a constant "baseline" volatility

**Cause**:
- **Insufficient graph signal**: Correlation-based adjacency may not capture spillover effects
- **Linear HAR dominance**: H1 pathway (linear HAR) may dominate H2 (graph pathway)
- **Training data period**: May not have included similar volatility regimes

---

## 🔬 Technical Analysis

### Model Architecture Issues

**GNNHAR1L Architecture**:
```python
H1 = Linear(3, 1)              # HAR pathway (local)
H2 = GCN(3, hidden) → MLP      # Graph pathway (spillover)
Output = H1 + H2               # Residual connection
```

**Potential Problems**:
1. **H1 dominance**: Linear HAR may overwhelm graph signal
2. **GCN expressiveness**: Single GCN layer may be insufficient
3. **Graph quality**: Static correlation matrix may not capture dynamic spillovers

### Training Data Issues

**Training Period**: Pre-2026 data  
**Test Period**: 2026-01-05 to 2026-03-27

**Possible mismatches**:
1. **Volatility regime shift**: 2026 may have different volatility characteristics
2. **Market structure changes**: COVID aftermath, policy changes
3. **Stock-specific events**: VIC/FPT may have company-specific factors

### Ensemble Method Issues

**Current approach**: Simple average of 5 models

**Problems**:
- No model screening (all 5 models used regardless of quality)
- No confidence weighting
- May amplify systematic biases if all models share similar errors

---

## 📈 Comparative Analysis

### VIC vs FPT Performance

**Why FPT performs better**:
1. **Lower volatility**: FPT actual range (0.034) vs VIC (0.053) - easier to predict
2. **Different sector**: Technology vs conglomerate - different volatility dynamics
3. **Market correlation**: FPT may be more predictable due to sector factors

**Why VIC performs worse**:
1. **Higher volatility**: Larger conglomerate with diverse business lines
2. **Market sensitivity**: More exposed to market-wide shocks
3. **Graph position**: May be in different network position affecting spillover capture

### Prediction Error Patterns

**VIC errors**:
- **Mean error**: +0.016 (systematic underprediction)
- **Error std**: 0.011 (high variability)
- **Worst case**: +0.042 (extreme underprediction)

**FPT errors**:
- **Mean error**: +0.007 (moderate underprediction)
- **Error std**: 0.009 (moderate variability)
- **Worst case**: +0.030 (severe underprediction)

---

## 💡 Recommendations

### Immediate Actions

1. **Inspect Training Curves**
   ```bash
   # Check if models converged properly
   ls results/gnnhar_paper/curve_h5_GNNHAR1L_*.png
   ```
   - Look for: underfitting (high train loss), overfitting (train-val gap)

2. **Compare with Baselines**
   ```bash
   # Test HAR (no graph) and GHAR (linear graph)
   python gnn/gnnhar_paper/infer_vic_fpt.py --model HAR --horizon 5
   python gnn/gnnhar_paper/infer_vic_fpt.py --model GHAR --horizon 5
   ```
   - If HAR performs similarly → Graph signal is weak
   - If GHAR performs better → Nonlinearity is unnecessary

3. **Test Different Periods**
   ```bash
   # Pre-COVID period
   python gnn/gnnhar_paper/infer_vic_fpt.py --start_date 2018-01-01 --end_date 2019-12-31
   
   # COVID period
   python gnn/gnnhar_paper/infer_vic_fpt.py --start_date 2020-01-01 --end_date 2021-12-31
   ```
   - Check if issue is period-specific or systematic

### Model Improvements

1. **Reduce Regularization**
   ```python
   # Current: weight_decay=1e-3 (strong)
   # Try: weight_decay=1e-4 (weaker)
   ```

2. **Add Model Screening**
   ```python
   # Screen models by validation loss (use top 50%)
   selected_models = screen_ensemble(models, val_losses, percentile=50)
   ```

3. **Try Different Architectures**
   - **GHAR**: Linear spillover only (may be more stable)
   - **GNNHAR2L**: 2-layer GCN (may capture more complex patterns)
   - **GNNHAR3L**: 3-layer GCN (may capture longer-range dependencies)

4. **Improve Graph Construction**
   ```python
   # Try dynamic adjacency (rolling window)
   # Try different correlation thresholds
   # Try sector-based graph construction
   ```

### Data Improvements

1. **Extend Training Period**
   - Include more recent data (2024-2025)
   - Ensure training data covers similar volatility regimes

2. **Add More Features**
   ```python
   # Current: [rv_d, rv_w, rv_m] (3 features)
   # Add: volume, VIX, macro indicators
   ```

3. **Stock-Specific Models**
   - Train separate models for high-volatility stocks (VIC)
   - Use ensemble of sector-specific models

---

## 🔍 Root Cause Hypothesis

### Primary Hypothesis: Training-Test Mismatch

**Evidence**:
1. **Severe underprediction**: Model predicts values 32-48% below actual
2. **Narrow prediction range**: Model shows no responsiveness to volatility changes
3. **Similar bias across stocks**: Both VIC and FPT show underprediction

**Conclusion**: Training data (pre-2026) may have different volatility characteristics than test period (2026).

### Secondary Hypothesis: Model Over-Regularization

**Evidence**:
1. **Compressed predictions**: All predictions clustered around mean
2. **No extreme values**: Model never predicts high or low volatility
3. **Conservative bias**: Systematic underprediction rather than random errors

**Conclusion**: Strong L2 regularization (weight_decay=1e-3) may prevent model from learning volatility dynamics.

### Tertiary Hypothesis: Weak Graph Signal

**Evidence**:
1. **Poor performance despite graph architecture**: GNNHAR1L should capture spillovers
2. **Similar issues across stocks**: Graph signal should help some stocks more than others
3. **Linear HAR may dominate**: H1 pathway may overwhelm H2 graph pathway

**Conclusion**: Correlation-based adjacency may not capture meaningful volatility spillovers for VN30 stocks.

---

## 📋 Next Steps (Priority Order)

### High Priority
1. **Compare with HAR baseline** - Determine if graph signal helps
2. **Inspect training curves** - Check for convergence issues
3. **Test different periods** - Check if issue is period-specific

### Medium Priority
4. **Reduce regularization** - Allow model to learn more volatility dynamics
5. **Try GHAR (linear)** - Test if nonlinearity is necessary
6. **Improve graph construction** - Use dynamic or sector-based adjacency

### Low Priority
7. **Train on extended period** - Include more recent data
8. **Add more features** - Volume, macro indicators
9. **Stock-specific tuning** - Different hyperparameters for high-volatility stocks

---

## 🎯 Success Criteria

**Good performance would be**:
- **R² > 0.5**: Model explains more than 50% of variance
- **MAE < 0.01**: Average error less than 1% volatility
- **Prediction range within 50% of actual range**: Responsive to volatility changes

**Current performance**:
- **R² = -2.54 (VIC), -0.85 (FPT)**: ❌ Failed
- **MAE = 0.021 (VIC), 0.009 (FPT)**: ❌ Failed (except FPT borderline)
- **Range compression 7-13×**: ❌ Failed

**Status**: **Model not ready for production use**. Requires significant improvements before deployment.

---

## 📊 Conclusion

The GNNHAR1L model shows **poor forecasting performance** on VIC and FPT stocks for the h5 horizon. The primary issues are:

1. **Systematic underprediction** (32-48% bias)
2. **Prediction range compression** (7-13× too narrow)
3. **Lack of volatility clustering capture**

The root causes are likely:
1. **Training-test mismatch** (different volatility regimes)
2. **Over-regularization** (preventing learning of dynamics)
3. **Weak graph signal** (spillovers not captured)

**Recommendation**: Investigate training data quality, compare with simpler baselines (HAR, GHAR), and reduce regularization before considering model deployment.

---

**Analysis Date**: 2026-06-03  
**Analyst**: Claude (AI Assistant)  
**Status**: ❌ Model Performance Poor - Requires Improvement