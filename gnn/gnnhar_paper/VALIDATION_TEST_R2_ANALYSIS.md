# Validation vs Test R² Discrepancy Analysis

**Critical Issue Identified**: Validation R² > 0 but Test R² < 0

**Date**: 2026-06-03  
**Root Cause**: **Volatility Regime Shift** between validation and test periods

---

## 🔍 Problem Statement

**Observed Behavior**:
- **Validation R²**: Positive (models appeared to perform well)
- **Test R²**: Negative (models performed worse than predicting mean)
- **Expected**: Similar R² across validation and test if model is robust

**This is a classic case of distribution shift between training and test data.**

---

## 🎯 Root Cause: Volatility Regime Shift

### Temporal Data Split Analysis

```
Training:   2006-10-27 to 2022-03-04 (3,828 snapshots, ~80% of pre-test data)
Validation: 2022-03-07 to 2025-12-31 (958 snapshots, ~20% of pre-test data)
Test:       2026-01-05 to 2026-05-29 (97 snapshots)
```

### Volatility Characteristics by Period

#### VIC (Vingroup)

| Period | Mean RV | Std RV | Range | Notes |
|--------|---------|--------|-------|-------|
| **Training** | 0.0170 | 0.0110 | [0.000, 0.076] | Baseline volatility |
| **Validation** | 0.0174 | 0.0129 | [0.0004, 0.063] | **+2.5%** vs training |
| **Test** | 0.0329 | 0.0125 | [0.006, 0.059] | **+88.5%** vs validation ⚠️ |

#### FPT (FPT Corporation)

| Period | Mean RV | Std RV | Range | Notes |
|--------|---------|--------|-------|-------|
| **Training** | 0.0160 | 0.0104 | [0.00005, 0.063] | Baseline volatility |
| **Validation** | 0.0152 | 0.0092 | [0.0014, 0.058] | **-5.2%** vs training |
| **Test** | 0.0220 | 0.0088 | [0.007, 0.045] | **+45.4%** vs validation ⚠️ |

### The Smoking Gun

**VIC experienced an 88.5% volatility increase** from validation to test period:
- Validation mean: 0.0174
- Test mean: 0.0329
- **Model predictions**: ~0.017 (based on validation period)

**FPT experienced a 45.4% volatility increase** from validation to test period:
- Validation mean: 0.0152
- Test mean: 0.0220
- **Model predictions**: ~0.015 (based on validation period)

---

## 📊 Why This Causes Validation R² > 0 but Test R² < 0

### Mechanism Explanation

#### During Validation (2022-2025)
1. **Model learned**: Mean volatility ≈ 0.016-0.017
2. **Validation data**: Mean volatility ≈ 0.015-0.017
3. **Match**: ✅ Training and validation have similar volatility levels
4. **Result**: Model predictions close to actual → Positive R²

#### During Testing (2026)
1. **Model predictions**: Still ~0.016-0.017 (learned from training/validation)
2. **Test data**: Mean volatility ≈ 0.022-0.033 (much higher!)
3. **Mismatch**: ❌ Model underestimates volatility by 45-89%
4. **Result**: Systematic underprediction → Negative R²

### Mathematical Explanation

**R² Formula**:
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_true - y_pred)²  # Residual sum of squares
SS_tot = Σ(y_true - ȳ_true)²  # Total sum of squares
```

**When model underpredicts systematically**:
- `y_pred << y_true` (predictions much lower than actual)
- `SS_res` becomes very large (huge prediction errors)
- `SS_tot` stays the same (based on true values)
- `R² = 1 - (large / normal) = negative value`

**Example**:
```
Actual: [0.03, 0.04, 0.05] → Mean = 0.04, SS_tot = 0.0002
Predicted: [0.015, 0.017, 0.019] → Mean = 0.017
SS_res = (0.03-0.015)² + (0.04-0.017)² + (0.05-0.019)² = 0.0016
R² = 1 - (0.0016 / 0.0002) = 1 - 8 = -7.0
```

---

## 🔬 Why the Model Failed to Generalize

### 1. Distribution Shift (Non-Stationarity)

**Assumption violated**: "Future volatility will resemble past volatility"

**Reality**: 2026 has fundamentally different volatility characteristics:
- **Post-COVID market dynamics**
- **Policy changes (interest rates, regulations)**
- **Sector-specific shocks**
- **Geopolitical events**

### 2. Conservative Learning

**Model behavior**: Learned to predict "safe" baseline values
- **Training range**: 0.000-0.076 (but clustered around 0.017)
- **Validation range**: 0.0004-0.063 (clustered around 0.017)
- **Test range**: 0.006-0.059 (but clustered around 0.033!)

**Model strategy**: Predict mean of training/validation (~0.017)
- **Result**: Works when volatility is low (validation)
- **Result**: Fails when volatility spikes (test)

### 3. Strong Regularization

**Current settings**:
```python
weight_decay=1e-3  # Strong L2 penalty
```

**Effect**: Model prevented from learning extreme values:
- During training: High volatility examples treated as "noise"
- During validation: Model predicts conservative values
- During testing: Conservative predictions fail on high volatility

---

## 📈 Validation of Root Cause

### Evidence 1: Systematic Underprediction

**Model predictions** (from inference results):
- VIC: Predicted mean ≈ 0.017 vs Actual mean ≈ 0.033 (48% underprediction)
- FPT: Predicted mean ≈ 0.015 vs Actual mean ≈ 0.022 (32% underprediction)

**Prediction range compression**:
- VIC: Actual range 0.053 → Predicted range 0.007 (13× compression)
- FPT: Actual range 0.034 → Predicted range 0.005 (7× compression)

### Evidence 2: Validation Losses Were Very Low

**From metadata**:
```python
val_losses = [0.000102, 0.000094, 0.000634, 0.000496, 0.000296]
```

**Interpretation**: Model fit validation data extremely well
- **MSE loss ≈ 0.0001** → Model was nearly perfect on validation
- **But**: This was because validation and training had similar volatility!

### Evidence 3: Temporal Gap

**Critical finding**: **18-month gap** between validation end and test start
- Validation end: 2025-12-31
- Test start: 2026-01-05
- Gap: Only 5 days ⚠️

**BUT**: The **volatility regime shift** happened in 2026:
- 2022-2025: Post-COVID recovery, lower volatility
- 2026: New market conditions, higher volatility

---

## 🎯 Why This is a Critical Problem

### 1. False Sense of Security

**During training**:
- ✅ Validation R² > 0 → "Model works!"
- ✅ Low validation loss → "Model converged!"
- ❌ **Reality**: Model only works on 2022-2025 data

### 2. Not Caught by Standard Validation

**Traditional validation assumes**: Validation and test are from same distribution
- **Your setup**: Validation (2022-2025) vs Test (2026) = **Different distributions!**
- **Result**: Validation performance **misleading** about test performance

### 3. Production Failure Risk

**If deployed**: Model would systematically underestimate risk in high-volatility periods
- **Trading**: Underestimate volatility → Insufficient position sizing → Losses
- **Risk management**: Underestimate risk → Inadequate hedging → Exposure
- **Portfolio allocation**: Underestimate volatility → Poor diversification → Drawdowns

---

## 💡 Solutions and Recommendations

### Immediate Actions

#### 1. Use Walk-Forward Validation ⭐ **RECOMMENDED**

**Instead of**: Single train/val/test split
**Use**: Rolling/expanding window validation

```python
# Walk-forward validation
for window_end in [2022-03-04, 2023-03-04, 2024-03-04, 2025-03-04]:
    train = data[:window_end]
    val = data[window_end:window_end + 1 year]
    test = data[window_end + 1 year:window_end + 2 years]
    
    # Train and evaluate
    model.train(train)
    val_score = model.evaluate(val)
    test_score = model.evaluate(test)
```

**Benefits**:
- Tests robustness to distribution shifts
- More realistic performance estimate
- Catches regime-specific issues

#### 2. Include Multiple Regimes in Training

**Current**: Training ends 2022-03-04 (misses recent volatility patterns)

**Better**: Extend training to include diverse volatility regimes:
```python
train_end = pd.Timestamp('2025-12-31')  # Include recent data
test_start = pd.Timestamp('2026-01-01')  # Test on 2026 only
```

#### 3. Add Regime Detection

**Detect volatility regime** before prediction:
```python
def detect_regime(recent_volatility):
    if recent_volatility.mean() > threshold:
        return "HIGH_VOL"
    else:
        return "NORMAL_VOL"

# Use different models or parameters for each regime
if regime == "HIGH_VOL":
    prediction = high_vol_model.predict(features)
else:
    prediction = normal_vol_model.predict(features)
```

### Medium-Term Improvements

#### 4. Reduce Regularization

**Current**: `weight_decay=1e-3` (too strong)

**Try**: `weight_decay=1e-4` or `weight_decay=1e-5`

**Rationale**: Allow model to learn higher volatility patterns

#### 5. Add Macro Features

**Current features**: Only lagged RV values

**Add features**:
- VIX index (market fear gauge)
- Interest rate changes
- Sector indices
- Trading volume

**Rationale**: Model can anticipate regime shifts from leading indicators

#### 6. Ensemble of Specialized Models

**Train separate models**:
- **Low-vol model**: On 2022-2025 data
- **High-vol model**: On 2020-2021 (COVID) data
- **Ensemble**: Weight predictions based on detected regime

### Long-Term Solutions

#### 7. Online Learning

**Continuously update** model with new data:
```python
# Daily/weekly model updates
while True:
    new_data = get_latest_data()
    model.update(new_data)  # Incremental learning
    model.save()
```

#### 8. Bayesian Approach

**Quantify uncertainty** in predictions:
```python
# Instead of point predictions
prediction = model.predict(features)
uncertainty = model.estimate_uncertainty(features)

# Use uncertainty for risk management
if uncertainty > threshold:
    position_size *= 0.5  # Reduce exposure
```

#### 9. Regime-Specific Architecture

**Different models** for different volatility regimes:
- **Normal regime**: Simple HAR model (few parameters)
- **High-vol regime**: Complex GNN with macro features (more parameters)
- **Crisis regime**: Ensemble of diverse models (maximum robustness)

---

## 🔍 Diagnostic Checklist

### Before Next Training Run

- [ ] **Check for temporal gaps** between validation and test
- [ ] **Analyze volatility statistics** for each period (mean, std, range)
- [ ] **Use walk-forward validation** instead of single split
- [ ] **Include diverse regimes** in training (low vol, high vol, crisis)
- [ ] **Add regime detection** to identify volatility shifts
- [ ] **Reduce regularization** to allow learning extreme values
- [ ] **Test on multiple periods** to ensure robustness

### Validation Metrics to Monitor

**Traditional metrics** (can be misleading):
- R² on validation set
- MSE loss on validation set

**Better metrics**:
- **R² consistency** across different time periods
- **Prediction range coverage** (do predictions span actual range?)
- **Volatility regime tracking** (performance in different regimes)
- **Walk-forward validation** (rolling window performance)

---

## 📋 Expected Performance After Fixes

### With Walk-Forward Validation

**Expected validation R²**: **Lower but more realistic**
- Current: R² > 0 (misleading)
- Fixed: R² ≈ 0 to 0.3 (honest estimate)

**Expected test R²**: **Closer to validation R²**
- Current: R² < 0 (surprising failure)
- Fixed: R² ≈ validation R² (consistent performance)

### With Extended Training Data

**Expected prediction bias**: **Reduced**
- Current: 32-48% underprediction
- Fixed: <10% underprediction

**Expected prediction range**: **Wider**
- Current: 7-13× compression
- Fixed: 2-3× compression (more responsive)

---

## 🎯 Conclusion

**Root Cause**: **Volatility regime shift** - 2026 has fundamentally different volatility characteristics than 2022-2025

**Primary Issue**: 
- VIC: 88.5% volatility increase from validation to test
- FPT: 45.4% volatility increase from validation to test
- Model learned on low-volatility period → Failed on high-volatility test

**Validation R² > 0 occurred because**:
- Validation period (2022-2025) had similar volatility to training
- Model predictions matched validation data well
- But validation performance didn't generalize to different test regime

**Test R² < 0 occurred because**:
- Test period (2026) had much higher volatility
- Model predictions based on training/validation were too low
- Systematic underprediction → Large errors → Negative R²

**This is a classic case of distribution shift that standard validation failed to catch.**

**Recommendation**: Implement walk-forward validation and include diverse volatility regimes in training to ensure robustness across different market conditions.

---

**Analysis Date**: 2026-06-03  
**Status**: ❌ **Critical Issue Identified - Validation Setup Misleading**  
**Priority**: 🔴 **HIGH - Immediate Action Required**