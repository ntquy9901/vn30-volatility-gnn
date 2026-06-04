# Critical Analysis: Published Paper vs Original Code vs Our Implementation

**Date:** 2026-05-31
**Paper:** "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)
**Source:** https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967

---

## 🚨 CRITICAL FINDING: Code Does NOT Match Published Paper!

### The Original Code (GNNHAR.py on GitHub)
- **Loss Function:** MSE (line 359: `criterion = nn.MSELoss()`)
- **This is NOT the paper's key innovation!**

### The Published Paper (ScienceDirect)
- **Loss Function:** Quasi-Likelihood (QL)
- **Results:** QL achieves ~13% lower MSE, ~4% lower QL error vs HAR
- **Quote:** "training with the quasi-likelihood loss leads to substantial improvements in model performance compared to the commonly used mean squared error"

**Conclusion:** The GitHub code is an **OLD VERSION** that does NOT implement the published paper's main contribution!

---

## 1. Loss Function Comparison

| Implementation | Training Loss | Published? | Results |
|----------------|---------------|------------|---------|
| **Published Paper** | Quasi-Likelihood (QL) | ✅ Yes | Best: ~13% MSE improvement |
| **GitHub Code** | MSE | ❌ No | Unknown (worse than QL) |
| **Our Code** | MSE | ❌ No | R² = -4.36 (catastrophic failure) |

### Why QL is Better (from paper Section 5)

**Heteroskedasticity Handling:**
- Volatility has heteroskedastic errors (variance changes over time)
- QL weights observations by inverse variance (gives less weight to high-volatility periods)
- MSE treats all errors equally (dominated by high-volatility periods)

**QL Formula (Section 3.2):**
```
L_QL(θ) = (1/N) Σ [exp(ŷ_t) - y_t * exp(ŷ_t)]
```

**Connection to Multiplicative Error Models (MEM):**
- QL-trained models ≈ Multiplicative Error Model (Engle 2002)
- MEM is standard for volatility forecasting
- Links to GARCH and econometric theory

---

## 2. Model Architecture Findings

### From Published Paper Abstract

**Multi-Hop Neighbors:**
> "incorporating spillover effects from multi-hop neighbors alone does **not** yield a clear advantage in terms of predictive accuracy."

**Conclusion:** GNNHAR2L and GNNHAR3L are **NOT better** than GNNHAR1L!

**Nonlinearity:**
> "modeling nonlinear spillover effects **enhances** the forecasting accuracy of realized volatilities, particularly for short-term horizons of up to one week."

**Conclusion:** GNNHAR1L (nonlinear) **IS better** than GHAR (linear)!

**Best Model:**
- **GNNHAR1L** (nonlinear, 1-hop only)
- Trained with **QL loss**
- Short horizons (1-day to 1-week)

### Original Code Architecture

The GitHub code implements the model architectures correctly:
- HAR: Linear baseline
- GHAR: Linear + 1 GCN layer (linear spillover)
- GNNHAR1L: Linear + 1 GCN + MLP (nonlinear spillover)
- GNNHAR2L: Linear + 2 GCN + MLP (2-hop nonlinear)
- GNNHAR3L: Linear + 3 GCN + MLP (3-hop nonlinear)

But uses **MSE loss** instead of QL!

---

## 3. Evaluation Metrics

### Published Paper (Table 2, Section 4)
- **Primary Metrics:** MSE, QLIKE (for both training and evaluation)
- **Secondary Metric:** R² (for model comparison)
- **Forecast Evaluation:** Out-of-sample R², comparing predicted vs actual RV

### Our Current Implementation
- **Training Loss:** MSE (wrong - should be QL)
- **Evaluation Metrics:** R², MAE, RMSE (missing QLIKE)
- **Test Set:** 15 samples only (way too small!)

---

## 4. Why Our Implementation Failed (R² = -4.36)

### Root Causes (in order of importance)

1. **Test Set Too Small** (CRITICAL)
   - Our: 15 samples (May 2026 only)
   - Should be: ~120 samples (full 2026) or rolling windows
   - R² unreliable with small samples

2. **Wrong Loss Function** (IMPORTANT)
   - Our: MSE
   - Should be: QL (paper's key innovation)
   - QL handles heteroskedasticity better

3. **Missing ReLU on Outputs** (MODERATE)
   - Paper equations show linear, but code has ReLU
   - ReLU ensures predictions ≥ 0 (appropriate for RV)
   - We removed ReLU → predictions can be negative

4. **Different RV Computation** (MINOR)
   - Paper: Sum of squared 5-min returns
   - Us: Rolling std of daily returns
   - Both measure volatility, different formulas

---

## 5. Our Implementation Errors

### Error 1: Using MSE Instead of QL

**Paper's Key Finding:**
> "compared to MSE-trained models, models employing QL as the EC generally achieve substantial improvements in predictive accuracy"

**Our Code (train_vic_ensemble.py line 225):**
```python
criterion = nn.MSELoss()  # ❌ WRONG - should use QL
```

**Impact:** Missing the paper's main innovation = worse performance

### Error 2: Small Test Set

**Paper Methodology (Section 3.3):**
- Rolling 22-day test windows throughout out-of-sample period
- Jan 2011 to July 2021 (~10 years of data)
- Retrain every 22 days

**Our Code (train_vic_ensemble.py lines 37-38):**
```python
TRAIN_END_DATE = "2026-04-30"  # Single period
TEST_START_DATE = "2026-05-01"   # Only May (15 samples!)
TEST_END_DATE = "2026-05-31"
```

**Impact:** R² = -4.36 meaningless with 15 samples

### Error 3: Removed ReLU from Outputs

**Paper Code (GNNHAR.py lines 23, 42, 60):**
```python
# All models have ReLU on output
res = self.relu(H1 + H2)
```

**Our Code (gnnhar_models.py line 77):**
```python
# HAR - No ReLU (we removed it)
return H1.squeeze(-1)  # ❌ Can predict negative values
```

**Impact:** Predictions can be negative (inappropriate for volatility)

### Error 4: Different RV Formula

**Paper (Section 3.1):**
```
RV_t = Σ_{τ=1}^{M} r_{t,τ}²  # Sum of squared returns
```

**Our Code (src/volatility_labels.py):**
```python
rv = log_ret.rolling(h, min_periods=h).std(ddof=1).shift(-h)
# Rolling standard deviation (different formula)
```

**Impact:** Minor - both measure volatility, but different scale

---

## 6. Action Plan to Match Published Paper

### Phase 1: Implement QL Loss Function (CRITICAL)

**Step 1.1: Implement QL loss**
```python
class QuasiLikelihoodLoss(nn.Module):
    """
    Quasi-likelihood loss for heteroskedastic volatility forecasting.
    From paper Section 3.2: L_QL(θ) = (1/N) Σ [exp(ŷ) - y*exp(ŷ)]

    Advantages over MSE:
    - Handles heteroskedasticity (variance changes over time)
    - Links to Multiplicative Error Models (MEM)
    - Weights observations by inverse variance
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: log(RV) predictions
        # target: actual RV values
        # QL = (1/N) Σ [exp(pred) - target * exp(pred)]
        exp_pred = torch.exp(pred)
        ql = torch.mean(exp_pred - target * exp_pred)
        return ql
```

**Step 1.2: Replace MSE with QL in training**
```python
# train_vic_ensemble.py line 225
criterion = QuasiLikelihoodLoss()  # Instead of nn.MSELoss()
```

### Phase 2: Restore ReLU to Model Outputs

**Step 2.1: Add ReLU to all models**
```python
# gnnhar_models.py - HAR model
class HAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.relu = nn.ReLU()  # ✅ Add back

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        return self.relu(H1.squeeze(-1))  # ✅ Apply ReLU
```

**Step 2.2: Apply to all GNNHAR models**
- GHAR: `return self.relu(H1 + H2).squeeze(-1)`
- GNNHAR1L: `return self.relu(H1 + H2).squeeze(-1)`
- GNNHAR2L: `return self.relu(H1 + H2).squeeze(-1)`
- GNNHAR3L: `return self.relu(H1 + H2).squeeze(-1)`

### Phase 3: Fix Test Set (Use Full 2026 Data)

**Step 3.1: Update test period**
```python
# train_vic_ensemble.py lines 37-38
TRAIN_END_DATE = "2025-12-31"  # All pre-2026 for training
TEST_START_DATE = "2026-01-01"   # Full 2026 for test
TEST_END_DATE = "2026-05-31"
```

**Expected test samples:** ~100-120 (vs current 15)

### Phase 4: Re-run Experiments

**With QL loss + ReLU + proper test set:**
- HAR (baseline): R² ≈ 0.3-0.5
- GHAR (linear spillover): R² ≈ 0.35-0.55
- GNNHAR1L (nonlinear): R² ≈ 0.5-0.7 **(best model per paper)**
- GNNHAR2L/GNNHAR3L: R² ≈ 0.4-0.6 (no better than 1L)

---

## 7. Summary Table: What to Fix

| Issue | Current | Should Be | Priority |
|-------|---------|-----------|----------|
| **Loss Function** | MSE | Quasi-Likelihood (QL) | 🔴 CRITICAL |
| **Test Set Size** | 15 samples | ~100-120 samples | 🔴 CRITICAL |
| **ReLU on Output** | Removed | Present | 🟡 IMPORTANT |
| **RV Computation** | Rolling std | Sum of squared returns | 🟢 OPTIONAL |
| **Evaluation** | R², MAE, RMSE | Add QLIKE | 🟢 NICE-TO-HAVE |

---

## 8. Key Takeaways

1. **The GitHub code is NOT the published paper** - it's an old version using MSE
2. **QL loss is the paper's main innovation** - ~13% MSE improvement
3. **Multi-hop models don't help** - GNNHAR1L is sufficient
4. **Nonlinearity DOES help** - GNNHAR1L beats GHAR
5. **Our test set is too small** - 15 samples can't produce reliable R²

**Next Step:** Implement QL loss and re-run experiments to see if we can match the paper's results.
