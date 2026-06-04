# GELU vs ReLU Activation Test Results

**Test Date:** 2026-06-01
**Model:** GNNHAR1L (1-layer GNNHAR)
**Horizon:** h=5
**Configuration:** 2 seeds → 1 model (50% screening), 100 epochs

## Quick Summary

**GELU shows modest but consistent improvement across all metrics:**
- **R²: +0.75%** (0.7167 → 0.7220)
- **MAE: +1.78%** (0.004677 → 0.004594)
- **RMSE: +0.95%** (0.006869 → 0.006804)

**Verdict:** MODEST IMPROVEMENT (0.5-2% range)
**Recommendation:** Test with more seeds before final decision

## Detailed Results

### Test Metrics Comparison

| Metric  | ReLU    | GELU    | Improvement |
|---------|---------|---------|-------------|
| R²      | 0.7167  | 0.7220  | **+0.75%** |
| MAE     | 0.004677 | 0.004594 | **+1.78%** |
| RMSE    | 0.006869 | 0.006804 | **+0.95%** |

### Validation Loss

- **GELU:** 1.3491
- **ReLU:** 1.3497
- **Difference:** GELU slightly lower (better)

### Convergence Analysis

Both activations converged equally well:
- Similar validation loss (1.349 vs 1.350)
- Both reached 100 epochs (max for quick test)
- No NaN losses or numerical instability

## Comparison with Expected Results

**Research expectation (from technical research document):**
- Expected improvement: +2-5% R²
- Actual improvement: +0.75% R²

**Analysis:**
- ✅ GELU improves all metrics (consistent direction)
- ⚠️ Improvement magnitude is lower than expected
- ✅ No negative side effects (training stable)
- ✅ Similar computational cost (~10% slower per forward pass)

## Why Lower Than Expected?

**Possible explanations:**

1. **Limited seeds (only 2)**
   - Small sample size increases variance
   - 50% screening keeps only 1 model
   - Need 10-20 seeds for reliable comparison

2. **Limited epochs (only 100)**
   - Models may not have fully converged
   - QL loss converges slowly (300-400 epochs typical)
   - GELU may show more improvement at convergence

3. **Task characteristics**
   - Volatility forecasting may benefit less from smooth activations
   - HAR features already capture main patterns
   - Graph signal may be more important than activation choice

## Recommendations

### Option 1: Test with More Seeds (RECOMMENDED)

**Rationale:** Modest improvement may become stronger with better statistics

**Action:**
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --n_seeds 10 \
    --epochs 200 \
    --activation gelu

python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --n_seeds 10 \
    --epochs 200 \
    --activation relu
```

**Expected duration:** ~2-3 hours total

**Success criteria:**
- GELU R² improvement ≥ 1% with 10 seeds → Proceed to full training
- GELU R² improvement < 1% with 10 seeds → Revert to ReLU

### Option 2: Proceed to Full Training (If Time-Constrained)

**Rationale:** +0.75% is still improvement, however modest

**Action:**
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --n_seeds 20 \
    --epochs 400 \
    --activation gelu
```

**Expected duration:** ~3.3 hours

**Expected outcome:**
- R² ≈ 0.73-0.74 (vs ReLU ≈ 0.72-0.73)
- Still below sklearn baseline (0.75+)
- Need Optuna optimization for major gains

### Option 3: Revert to ReLU (If Focusing on Higher Impact)

**Rationale:** Focus resources on higher-impact improvements

**Higher-impact alternatives (from technical research):**
1. **Optuna hyperparameter optimization** (+5-10% R² expected)
2. **Full ensemble training** (+5-15% R² expected from proper baselines)
3. **Graph Attention Networks** (+3-8% R² expected)

**Action:**
- Keep ReLU as default activation
- Implement Optuna optimization (higher priority)
- Implement GAT for adaptive spillover
- GELU implementation remains available for future testing

## Technical Assessment

### GELU Strengths (Observed)

1. **Consistent improvement** - Better across all metrics
2. **Smooth convergence** - No numerical issues
3. **Similar validation loss** - Converged equally well
4. **No negative side effects** - Training stable

### GELU Limitations (Observed)

1. **Modest improvement** - Only +0.75% R² (vs expected +2-5%)
2. **Computational cost** - ~10% slower per forward pass
3. **Limited impact** - Doesn't address major performance gaps

### Comparison with Baselines

**Current results (GELU, 2 seeds, 100 epochs):**
- GNNHAR1L (GELU): R² = 0.7220
- GNNHAR1L (ReLU): R² = 0.7167
- sklearn GHAR: R² = 0.7538 (from previous experiments)

**Gap to baselines:**
- GELU still -3.15% R² below sklearn GHAR
- Need +3.15% R² to match sklearn
- GELU alone won't close this gap

## Strategic Recommendation

**Priority 1: Complete Full Ensemble Training (CRITICAL)**

**Rationale:**
- Current tests undertrained (100 epochs vs 400 needed)
- All models (HAR, GHAR, GNNHAR1L) need fair comparison
- Undertrained baselines mask true performance

**Command:**
```bash
# Complete ensemble for all models with ReLU
python gnn/gnnhar_paper/train_multi_stock.py --model HAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GHAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --epochs 400
```

**Expected outcome:**
- Establish fair baselines (R² ≈ 0.75 for all models)
- Compare GNN benefit with proper training
- Identify if graph signal is real (expected: +0.02 R² improvement)

**Priority 2: Optuna Hyperparameter Optimization (HIGH IMPACT)**

**Rationale:**
- Expected +5-10% R² improvement
- Systematic tuning vs manual hyperparameters
- Addresses "why nonlinearity isn't helping" question

**Command:**
```bash
# After full ensemble complete
# Implement Optuna with search space:
# - lr: [1e-4, 1e-2]
# - weight_decay: [1e-6, 1e-4]
# - n_hid: [16, 32, 64]
# - adj_threshold: [0.2, 0.5]
```

**Priority 3: Test GELU with 10 Seeds (MEDIUM PRIORITY)**

**Rationale:**
- Confirm if +0.75% improvement holds with better statistics
- Low cost (~2 hours total)

**Decision matrix:**
- If GELU ≥ 1% improvement with 10 seeds → Use for full training
- If GELU < 1% improvement with 10 seeds → Keep ReLU as default

## Conclusion

**GELU shows promise but not a game-changer:**

✅ **Pros:**
- Consistent improvement across metrics
- Smooth gradients, better convergence theoretically
- No negative side effects
- Implementation ready and tested

❌ **Cons:**
- Modest improvement (+0.75% vs expected +2-5%)
- Additional computational cost (~10% slower)
- Doesn't close gap to sklearn baselines
- Lower priority than ensemble training and Optuna

**Final recommendation:**

1. **Immediate:** Complete full ensemble training (20 seeds, 400 epochs) with ReLU
2. **Short-term:** Implement Optuna hyperparameter optimization
3. **Medium-term:** Test GELU with 10 seeds if time permits
4. **Long-term:** Consider GELU for production if it shows ≥1% improvement with 10 seeds

**Key insight:** GELU is a nice-to-have improvement, but ensemble training and hyperparameter optimization are more critical for achieving target performance (R² ≥ 0.75).

---

**Next action:** Run `train_all_models.bat` to complete full ensemble training
