# GELU vs ReLU Test Results - Quick Summary

## Test Configuration
- **Model:** GNNHAR1L
- **Seeds:** 2 → 1 model (50% screening)
- **Epochs:** 100
- **Date:** 2026-06-01

## Results Comparison

| Metric | ReLU | GELU | Improvement |
|--------|------|------|-------------|
| **R²** | 0.7167 | 0.7220 | **+0.75%** ✅ |
| **MAE** | 0.004677 | 0.004594 | **+1.78%** ✅ |
| **RMSE** | 0.006869 | 0.006804 | **+0.95%** ✅ |

## Verdict

**MODEST IMPROVEMENT** (0.5-2% range)

✅ GELU improves all metrics consistently
⚠️ Improvement magnitude lower than expected (+0.75% vs +2-5% expected)
✅ No negative side effects (training stable)

## Recommendations

### Option 1: Test with More Seeds (RECOMMENDED)
```bash
# Test with 10 seeds, 200 epochs
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --n_seeds 10 --epochs 200 --activation gelu

python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --n_seeds 10 --epochs 200 --activation relu
```

**Decision:**
- If GELU ≥ 1% improvement → Use for full training
- If GELU < 1% improvement → Keep ReLU as default

### Option 2: Focus on Higher Impact (ALTERNATIVE)

**Higher priorities (from technical research):**
1. **Full ensemble training** (+5-15% R² expected)
2. **Optuna optimization** (+5-10% R² expected)
3. **Graph Attention Networks** (+3-8% R² expected)

**Rationale:** GELU's +0.75% is modest compared to these improvements

## Key Insights

**Why GELU showed lower improvement than expected:**
1. Limited seeds (only 2) - high variance
2. Limited epochs (100 vs 400 needed) - undertrained
3. Volatility task may benefit less from smooth activations

**Comparison with baselines:**
- GELU: R² = 0.7220
- sklearn GHAR: R² = 0.7538
- **Gap:** -3.15% (GELU still below sklearn)

**Conclusion:**
- GELU is a nice-to-have improvement
- Ensemble training + Optuna are more critical
- Keep GELU implementation for future testing

## Next Steps

**Immediate priority:**
```bash
# Complete full ensemble training (CRITICAL)
python gnn/gnnhar_paper/train_multi_stock.py --model HAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GHAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --epochs 400
```

**Expected outcome:**
- Establish fair baselines (R² ≈ 0.75)
- Identify true graph signal
- Close gap to sklearn baselines

---

**Status:** GELU implementation complete and tested ✅
**Recommendation:** Focus on ensemble training first, test GELU with 10 seeds later
