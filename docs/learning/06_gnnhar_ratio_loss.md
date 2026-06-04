# GNNHAR Ratio Loss — QLIKE Confusion Explained

**Date:** 2026-06-01  
**Source Question:** Why is the loss function called "quasi_likelihood_loss" when it's not standard QLIKE?  
**Status:** Critical bug fix and naming clarification in v1.3_LOSS_FIX

---

## Context: Naming Confusion

The GNNHAR paper's code contains a misleading function name that causes confusion:

```python
# In GNNHAR.py line 322 (paper's GitHub code)
def quasi_likelihood_loss(outputs, forecast_y):
    true_fore = outputs / (forecast_y + 1e-4)
    l_v = torch.mean(true_fore - torch.log(true_fore + 1e-4))
    return l_v
```

**Problem:** The function is named `quasi_likelihood_loss`, suggesting it implements the well-known QLIKE loss from volatility forecasting literature. But it **does not**.

---

## Standard QLIKE vs. GNNHAR Ratio Loss

### Standard QLIKE (from volatility literature)

**Formula:**
```
QLIKE = (1/T) * Σ [σ_t²/σ̂_t² + log(σ̂_t²) - log(σ_t²)]
     = (1/T) * Σ [σ̂_t/σ_t - log(σ̂_t/σ_t) - 1]
```

**Key characteristic:** The `-1` term

### GNNHAR "Quasi-Likelihood" (paper's implementation)

**Formula:**
```
L_ratio = (1/T) * Σ [σ_t²/(σ̂_t² + eps) - log(σ_t²/(σ̂_t² + eps) + eps)]
```

**Key difference:** Missing the `-1` term

---

## Why This Matters

### 1. Different Optimization Landscape

The `-1` in QLIKE is not a minor detail — it changes the loss function:

| Loss Function | Perfect Prediction | Under-prediction | Over-prediction |
|--------------|-------------------|------------------|-----------------|
| **Standard QLIKE** | 0 (minimum) | Grows slowly | Grows fast |
| **GNNHAR Ratio** | 1 (minimum) | Grows AGGRESSIVELY | Grows slowly |

**Critical difference:** The penalty asymmetry is INVERTED.

### 2. Risk Management Implications

- **Standard QLIKE:** Penalizes over-prediction more (treats over-estimation as worse)
- **GNNHAR Ratio:** Penalizes under-prediction more (treats under-estimation as worse)

For volatility risk management, GNNHAR's behavior is **CORRECT**:
- Under-predicting volatility = DANGEROUS (insufficient capital reserves)
- Over-predicting volatility = SAFE (conservative, wasteful but not catastrophic)

### 3. Confusion in Code Review

When you see `quasi_likelihood_loss` in code, you expect:
- Standard QLIKE formula with `-1` term
- Well-documented behavior from volatility literature
- Consistency with other implementations

But you get:
- Custom ratio loss without `-1`
- Paper-specific behavior not in literature
- Inconsistent naming across codebases

---

## The Fix in v1.3_LOSS_FIX

### What Was Done

1. **Renamed function** to `gnnhar_ratio_loss` — truth in advertising
2. **Kept deprecated alias** `quasi_likelihood_loss` with deprecation warning
3. **Added comprehensive documentation** explaining the difference
4. **Created this learning doc** for future reference

### Before v1.3_LOSS_FIX

```python
def quasi_likelihood_loss(y_pred, y_true, eps=1e-4):
    ratio = y_pred / (y_true + eps)  # WRONG: inverted ratio
    loss = ratio - torch.log(ratio)   # WRONG: missing eps in log
    return loss.mean()
```

**Bugs:**
- Inverted ratio (`y_pred/y_true` instead of `y_true/y_pred`)
- Missing `eps` in log term
- Misleading name suggesting standard QLIKE

### After v1.3_LOSS_FIX

```python
def gnnhar_ratio_loss(y_pred, y_true, eps=1e-4):
    """Custom GNNHAR ratio loss (NOT standard QLIKE)."""
    ratio = y_true / (y_pred + eps)  # CORRECT ratio
    loss = ratio - torch.log(ratio + eps)  # CORRECT eps placement
    return loss.mean()

# Deprecated alias with warning
def quasi_likelihood_loss(y_pred, y_true, eps=1e-4):
    warnings.warn("Deprecated: Use gnnhar_ratio_loss instead")
    return gnnhar_ratio_loss(y_pred, y_true, eps)
```

**Fixes:**
- Corrected ratio direction
- Added `eps` to log term
- Clear documentation that this is NOT QLIKE
- Deprecation path for old name

---

## Migration Guide

### For New Code

```python
from gnn.gnnhar_paper.gnnhar_models import gnnhar_ratio_loss

loss = gnnhar_ratio_loss(y_pred, y_true, eps=1e-4)
```

### For Existing Code

```python
# Old (will show deprecation warning)
from gnn.gnnhar_paper.gnnhar_models import quasi_likelihood_loss
loss = quasi_likelihood_loss(y_pred, y_true)  # Still works, but warns

# Updated (recommended)
from gnn.gnnhar_paper.gnnhar_models import gnnhar_ratio_loss
loss = gnnhar_ratio_loss(y_pred, y_true)  # Clear name, no warning
```

---

## Common Pitfalls

### Pitfall 1: Assuming QLIKE Behavior

**Wrong:**
```python
# "This is QLIKE, so perfect prediction should give loss = 0"
assert loss.item() == 0  # FAILS — minimum is 1.0
```

**Right:**
```python
# "This is GNNHAR ratio loss, minimum is 1.0"
assert abs(loss.item() - 1.0) < 0.01  # PASSES
```

### Pitfall 2: Forgetting Numerical Guards

**Wrong:**
```python
ratio = y_true / y_pred  # No eps guard
loss = ratio - torch.log(ratio)  # No log guard
```

**Right:**
```python
ratio = y_true / (y_pred + eps)  # Guard division
loss = ratio - torch.log(ratio + eps)  # Guard log domain
```

### Pitfall 3: Ignoring Asymmetric Penalty

**Wrong:**
```python
# "Over-prediction should be penalized more" (QLIKE intuition)
assert loss_over > loss_under  # WRONG for GNNHAR ratio loss
```

**Right:**
```python
# "Under-prediction should be penalized more" (risk management)
assert loss_under > loss_over  # CORRECT for GNNHAR ratio loss
```

---

## Why Use MSE Instead?

From `CONSTRAINTS.md C3`:

> QLIKE has been removed from this project. Current approach:
> - Train loss: MSE on z-scored HAR residuals
> - Evaluation metrics: R², MAE, RMSE only
> - Z-scoring is mandatory for equal learning signal per stock

**Reasons:**
1. **Stability:** MSE is numerically stable, no edge cases near zero
2. **Interpretability:** R², MAE, RMSE map directly to business metrics
3. **Simplicity:** Well-understood curvature, easier to debug
4. **Standardization:** Works with z-scored residuals for fair comparison

**When to use GNNHAR ratio loss:**
- Reproducing paper results exactly
- Risk management focus (asymmetric penalty needed)
- Scale-invariance required (percentage errors matter)

---

## Takeaways

1. **Naming matters** — `quasi_likelihood_loss` sounds like QLIKE but isn't
2. **The `-1` term matters** — changes the entire optimization landscape
3. **Asymmetric penalty matters** — under-prediction vs over-prediction risk
4. **Documentation matters** — future you will thank current you for clear docs
5. **Tests matter** — the test suite caught both the ratio bug and the eps bug

**Lesson:** When a function name doesn't match its behavior, rename it. Truth in advertising beats backward compatibility every time.

---

## References

- GNNHAR Paper: "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)
- Standard QLIKE: Patton & Sheppard (2015), "Good Volatility, Bad Volatility, and the Cross-Section of Stock Returns"
- v1.3_LOSS_FIX: Bug fix commit correcting ratio inversion and eps placement
- Test suite: `gnn/gnnhar_paper/tests/test_quasi_likelihood_loss.py`
