# Bug Report: Critical Loss Function Ratio Inversion (v1.3_LOSS_FIX)

**Date Discovered:** 2026-06-01  
**Date Fixed:** 2026-06-01  
**Severity:** CRITICAL — Invalidates all previous GNNHAR model results  
**Affected Versions:** v1.0, v1.1, v1.2 (all GNNHAR models before v1.3)  
**Fix Version:** v1.3_LOSS_FIX

---

## Executive Summary

A critical bug was discovered in the `quasi_likelihood_loss` function implementation that fundamentally inverted the optimization objective. The ratio was computed as `y_pred/y_true` instead of `y_true/y_pred`, causing models to optimize for the inverse of the intended objective function.

**Impact:** All GNNHAR model results (v1.0-v1.2) are INVALID and cannot be used for thesis conclusions or comparison.

**Status:** Bug fixed, all results re-Tagged, training restarted with corrected loss function.

---

## Root Cause Analysis

### The Bug

**Location:** `gnn/gnnhar_paper/gnnhar_models.py` (lines 87-95 in v1.2)

**Incorrect Implementation (v1.0-v1.2):**
```python
# WRONG: Inverted ratio
ratio = y_pred / (y_true + eps)
loss = ratio - torch.log(ratio + eps)
return loss.mean()
```

**Correct Implementation (v1.3_LOSS_FIX):**
```python
# CORRECT: Proper ratio direction
ratio = y_true / (y_pred + eps)
loss = ratio - torch.log(ratio + eps)
return loss.mean()
```

### Why This Happened

The bug was introduced during initial implementation of the GNNHAR paper's loss function. The paper code shows:

```python
# From GNNHAR.py line 322 (paper's GitHub)
true_fore = outputs / (forecast_y + 1e-4)
```

Where:
- `outputs` = target (y_true)
- `forecast_y` = prediction (y_pred)

The implementation mistakenly reversed the order to `y_pred / y_true`, inverting the optimization landscape.

### Why It Wasn't Caught Earlier

1. **Loss still decreased:** The inverted ratio still created a valid loss landscape, so training appeared to converge
2. **Reasonable metrics:** Models still achieved R² values, masking the underlying objective mismatch
3. **No validation against paper:** Initial implementation wasn't verified against paper's exact formula
4. **Misleading function name:** Called "quasi_likelihood_loss" but behavior didn't match standard QLIKE

---

## Impact Assessment

### Technical Impact

| Aspect | Impact |
|--------|--------|
| **Optimization objective** | Completely inverted - models minimized wrong function |
| **Asymmetric penalty** | Reversed - penalized over-prediction instead of under-prediction |
| **Model behavior** | Trained to optimize inverse of intended objective |
| **Convergence** | Models converged, but to wrong optimum |

### Results Impact

**INVALID Results (cannot be used for thesis):**
- All GNNHAR models: v1.0, v1.1, v1.2
- All Optuna hyperparameter studies using loss function
- All metrics and evaluation results from these models
- All learning curves and training artifacts

**VALID Results (unaffected):**
- HAR-RV baseline models (don't use this loss function)
- Other baseline models (LSTM, MLP, etc.)
- v1.3_LOSS_FIX and later versions

### Timeline Impact

- **Additional time needed:** 10-14 days for re-running experiments
- **Thesis timeline:** Extends final draft completion by ~2 weeks
- **GPU time:** ~5 days for Optuna, ~3 days for final model training
- **Buffer available:** Still within submission timeline

### Academic Integrity Impact

**Risk Level:** LOW IF HANDLED CORRECTLY

**Mitigation:**
- Full transparency in documentation
- Clear version labeling (v1.3_LOSS_FIX)
- Thesis section documenting bug discovery and correction
- Archive (don't delete) invalid results for reference

**Positive framing:** This demonstrates research rigor and systematic code review

---

## The Fix

### Changes Made in v1.3_LOSS_FIX

**1. Corrected Ratio Direction**
```python
# Before (WRONG)
ratio = y_pred / (y_true + eps)

# After (CORRECT)
ratio = y_true / (y_pred + eps)
```

**2. Added Eps to Log Term**
```python
# Before (WRONG)
loss = ratio - torch.log(ratio)

# After (CORRECT)
loss = ratio - torch.log(ratio + eps)
```

**3. Enhanced Documentation**
```python
def gnnhar_ratio_loss(y_pred, y_true, eps=1e-4):
    """
    GNNHAR Ratio Loss — custom loss from GNNHAR paper (NOT standard QLIKE).

    Formula: L = mean(target / (pred + eps) - log(target / (pred + eps) + eps))

    This penalizes under-prediction MORE heavily (correct for risk management).
    """
```

**4. Renamed Function**
- Old name: `quasi_likelihood_loss` (misleading - suggests standard QLIKE)
- New name: `gnnhar_ratio_loss` (accurate - describes actual behavior)
- Backward compatibility: Deprecated alias with warning

**5. Added Comprehensive Tests**
- 7 critical tests validating correctness
- All edge cases covered (zero volatility, near-zero predictions, gradient flow)
- Test suite: `gnn/gnnhar_paper/tests/test_quasi_likelihood_loss.py`

### Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `gnnhar_models.py` | Fixed ratio, added eps to log, renamed function | Core fix |
| `train_multi_stock.py` | Updated imports, changed to gnnhar_ratio_loss | Use new function |
| `optuna_gnnhar_optimization.py` | Updated imports | Use new function |
| `tests/test_quasi_likelihood_loss.py` | Created comprehensive test suite | Validate fix |
| `tests/test_function_rename.py` | Created rename validation test | Ensure compatibility |
| `docs/learning/06_gnnhar_ratio_loss.md` | Created learning doc | Explain confusion |

---

## Recovery Plan

### Immediate Actions (Completed 2026-06-01)

- [x] Fix loss function implementation
- [x] Add comprehensive test suite
- [x] Update documentation and function names
- [x] Version tag: v1.3_LOSS_FIX
- [x] Add warnings to training scripts

### Short-term Actions (Week of 2026-06-01)

- [ ] Re-run Optuna hyperparameter optimization (5 days GPU time)
- [ ] Re-train GNNHAR models with corrected loss (3 days GPU time)
- [ ] Archive invalid results to separate folder
- [ ] Update thesis results section
- [ ] Draft advisor communication

### Medium-term Actions (Week of 2026-06-08)

- [ ] Compare old vs. new results (appendix material)
- [ ] Document "lessons learned" in thesis
- [ ] Finalize all GNNHAR experiments
- [ ] Update CONSTRAINTS.md with loss function verification checklist

### Long-term Actions (Thesis completion)

- [ ] Include bug discovery in methodology section
- [ ] Frame as quality assurance case study
- [ ] Ensure all cited results use v1.3_LOSS_FIX or later

---

## Artifact Segregation

### Directory Structure

```
results/
├── invalid_loss_bug/           # v1.0-v1.2 results (DO NOT USE)
│   ├── GNNHAR1L_relu_h5_20260528_143022.json
│   ├── GNNHAR1L_gelu_h5_20260530_153145.json
│   └── [all other pre-v1.3 results]
├── v1.3_LOSS_FIX/              # Valid results (USE THESE)
│   ├── GNNHAR1L_gelu_h5_20260601_221345.json
│   └── [all v1.3+ results]
└── optuna/
    ├── optuna_studies_invalid/  # Pre-v1.3 Optuna results
    └── optuna_studies_v1.3/    # Post-v1.3 Optuna results
```

### Tagging Convention

Invalid results should be clearly marked:
```json
{
  "version": "v1.2_DROPOUT",
  "valid": false,
  "invalid_reason": "loss_function_bug_v1.3_LOSS_FIX",
  "note": "DO NOT USE - results based on inverted loss ratio"
}
```

Valid results:
```json
{
  "version": "v1.3_LOSS_FIX",
  "valid": true,
  "loss_function": "gnnhar_ratio_loss",
  "note": "Valid - uses corrected loss function"
}
```

---

## Communication Strategy

### Advisor Communication (Draft)

**Subject:** Critical Bug Discovery in GNNHAR Loss Function — Impact Mitigation Plan

Dear [Advisor],

I discovered a critical bug in the GNNHAR model loss function that invalidates previous optimization results.

**Key Points:**
- **Bug:** Inverted ratio in loss computation (y_pred/y_true vs y_true/y_pred)
- **Impact:** All GNNHAR models v1.0-v1.2 optimized for wrong objective
- **Fix:** Version v1.3_LOSS_FIX deployed 2026-06-01 with corrected loss
- **Risk:** No impact on HAR-RV baseline or other models; GNNHAR requires re-run

**Recovery Plan:**
1. Re-run Optuna hyperparameter optimization (estimated 5 days GPU time)
2. Re-train final GNNHAR models with corrected loss (estimated 3 days)
3. Update thesis results section with v1.3 findings only
4. Archive invalid results in separate folder for transparency

**Timeline Impact:** +10 days to original schedule (still within submission buffer)

**Questions for Guidance:**
1. Should I include bug discovery as a "lessons learned" section in thesis?
2. Prioritize key horizons (h=1, h=20) to save time, or run full study?
3. Include old vs. new comparison in appendix, or focus solely on valid results?

Full technical documentation: `docs/bug_fix_v1.3_loss_ratio.md`

[Your name]

---

## Lessons Learned

### What Went Wrong

1. **Insufficient validation against paper source**
2. **Misleading function name masked behavior**
3. **No comprehensive test suite during initial implementation**
4. **Assumption that "training converges" = "correct implementation"**

### What Went Right

1. **Systematic code review caught the bug** (pre-submission, not post-publication)
2. **Version control allows clear invalid/valid separation**
3. **Transparent documentation and communication**
4. **Comprehensive fix with tests prevents recurrence**

### Process Improvements Implemented

1. **Test-driven development** for critical loss functions
2. **Paper-source verification** for all mathematical implementations
3. **Clear naming conventions** (no misleading function names)
4. **Version tagging** for all experimental results
5. **Artifact segregation** for invalid vs. valid results

---

## Verification Checklist

### For Future Loss Function Implementation

- [ ] Compare implementation line-by-line with paper source code
- [ ] Verify parameter order (which is target, which is prediction?)
- [ ] Check epsilon placement (guards both division AND log?)
- [ ] Test edge cases (zero values, near-zero values, perfect prediction)
- [ ] Validate gradient flow (no NaN/inf)
- [ ] Verify against paper's test values if available
- [ ] Document minimum loss value (not all losses minimize to 0)
- [ ] Add comprehensive test suite before use
- [ ] Use clear, descriptive function names
- [ ] Document why this loss vs. alternatives (MSE, MAE, etc.)

---

## References

- **GNNHAR Paper:** "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)
- **Paper Code:** GitHub.com/chaozhang-ox/GNNHAR (GNNHAR.py line 322)
- **Fix Date:** 2026-06-01
- **Fix Version:** v1.3_LOSS_FIX
- **Test Suite:** `gnn/gnnhar_paper/tests/test_quasi_likelihood_loss.py`
- **Learning Doc:** `docs/learning/06_gnnhar_ratio_loss.md`

---

## Status: RESOLVED ✅

**Bug:** Fixed and validated  
**Tests:** All passing  
**Documentation:** Complete  
**Training:** Restarted with corrected loss  
**Communication:** Advisor notification pending

**Next Steps:** Proceed with re-running experiments using v1.3_LOSS_FIX

---

*Last Updated: 2026-06-01*  
*Version: 1.0*  
*Status: Final*
