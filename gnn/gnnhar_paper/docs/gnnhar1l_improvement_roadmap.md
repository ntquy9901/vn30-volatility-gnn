# GNNHAR1L Improvement Roadmap

**Current Status:** 2026-06-01
**Current Performance:** R² = 0.722 (GELU, 2 seeds, 100 epochs)
**Target Performance:** R² ≥ 0.75 (match sklearn baselines)
**Gap to Target:** -3.15% R²

---

## Priority 1: Complete Full Ensemble Training (CRITICAL)

**Why first?**
- Current tests undertrained (100 epochs vs 400 needed for QL loss convergence)
- Need fair comparison: all models (HAR, GHAR, GNNHAR1L) with same training
- Undertrained baselines mask true performance
- This is blocking all other improvements

**Expected impact:** +5-15% R² improvement
- From previous experiments: proper training adds +0.02-0.05 R²
- Ensemble averaging reduces variance
- Early stopping prevents overfitting

**Command:**
```bash
# Complete ensemble for all models (20 seeds, 400 epochs)
python gnn/gnnhar_paper/train_multi_stock.py --model HAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GHAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --epochs 400
```

**Expected duration:** ~9 hours total
- HAR: ~2.2 hours
- GHAR: ~3.3 hours
- GNNHAR1L: ~3.3 hours

**Success criteria:**
- R² ≥ 0.75 for all models (match sklearn baselines)
- Low variance across 20 seeds (std < 0.02 R²)
- GHAR beats HAR by +0.02 R² (graph signal validation)

**Decision:**
- ✅ If GNNHAR1L ≥ 0.75 → Proceed to Priority 2
- ⚠️ If GNNHAR1L < 0.75 → Investigate convergence issues

---

## Priority 2: Optuna Hyperparameter Optimization (HIGH IMPACT)

**Why second?**
- Highest impact improvement after ensemble training
- Systematic tuning vs manual hyperparameter selection
- Addresses "why nonlinearity isn't helping" question
- Expected +5-10% R² improvement

**Search space:**
```python
{
    'lr': [1e-4, 1e-3, 1e-2],           # Learning rate
    'weight_decay': [1e-6, 1e-5, 1e-4],  # L2 regularization
    'n_hid': [16, 32, 64],               # Hidden dimension
    'adj_threshold': [0.2, 0.3, 0.4],    # Graph density
    'dropout': [0.0, 0.1, 0.2],          # If added
}
```

**Implementation:**
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    n_hid = trial.suggest_categorical('n_hid', [16, 32, 64])
    adj_threshold = trial.suggest_float('adj_threshold', 0.2, 0.5)

    # Train model with these hyperparameters
    val_r2 = train_and_evaluate(lr, weight_decay, n_hid, adj_threshold)
    return val_r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Expected duration:** ~8 hours (100 trials × ~5 min/trial)

**Success criteria:**
- Find hyperparameters with R² ≥ 0.77
- Improvement over baseline ≥ +5% R²

**Decision:**
- ✅ If Optuna achieves ≥ 0.77 → Proceed to production
- ⚠️ If Optuna < 0.77 → Consider architectural changes

---

## Priority 3: Graph Attention Networks (MEDIUM IMPACT)

**Why third?**
- Addresses over-smoothing problem in deep GNNs
- Adaptive neighbor selection (vs fixed adjacency)
- Expected +3-8% R² improvement
- Solves architectural limitation

**Architecture change:**
```python
# Current: GCN (fixed adjacency)
H2 = GCN(3, n_hid)(node_feat, adj)  # Fixed weights

# Proposed: GAT (learned attention)
H2 = GAT(3, n_hid, heads=4)(node_feat, adj)  # Learned attention
```

**Benefits:**
1. Adaptive neighbor selection (learns which stocks matter)
2. Attention weights show spillover importance (interpretability)
3. Better generalization to unseen correlation patterns
4. Prevents over-smoothing in deeper networks

**Expected duration:** ~14 hours (8 hours implementation + 6 hours training)

**Success criteria:**
- GAT beats GCN by ≥ +3% R²
- Attention weights reveal meaningful spillover patterns
- No over-smoothing with 2+ layers

**Decision:**
- ✅ If GAT beats GCN by ≥ 3% → Use for production
- ⚠️ If GAT < 3% improvement → Keep GCN (simpler)

---

## Priority 4: Test GELU with 10 Seeds (LOW PRIORITY)

**Why fourth?**
- GELU showed modest improvement (+0.75% R²)
- Lower impact than ensemble or Optuna
- Worth confirming with better statistics

**Command:**
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

**Expected duration:** ~2 hours total

**Decision:**
- ✅ If GELU ≥ 1% improvement with 10 seeds → Use for full training
- ❌ If GELU < 1% improvement → Keep ReLU as default

---

## Priority 5: Encoder-Decoder Architecture (RESEARCH FRONTIER)

**Why last?**
- Complex architecture (20-30 hours implementation)
- Multi-horizon capability (nice-to-have, not critical)
- Higher risk, longer development time

**Architecture:**
```python
# Current: Single horizon per model
model = GNNHAR1L(h=5)  # Predicts h=5 only

# Proposed: Multi-horizon model
encoder = GCN_Encoder(node_feat, adj)  # Encodes context
decoder = Attention_Decoder(encoder, h)  # Predicts h=1,5,10,20
```

**Benefits:**
1. Single model for all horizons (vs training separate models)
2. Cross-attention learns which neighbors matter for each horizon
3. Adaptive spillover effects across time horizons

**Expected duration:** ~30 hours (20 hours implementation + 10 hours testing)

**Success criteria:**
- Single model achieves R² ≥ 0.73 for all horizons [1,5,10,20]
- Better than training separate models per horizon

**Decision:**
- ✅ If time permits and thesis requires multi-horizon → Implement
- ⚠️ Otherwise → Focus on Priorities 1-3

---

## Summary and Recommendation

### Immediate Action (This Week)

**Priority 1: Full Ensemble Training**
```bash
python gnn/gnnhar_paper/train_multi_stock.py --model HAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GHAR --n_seeds 20 --epochs 400
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --epochs 400
```

**Expected outcome:** R² ≥ 0.75 for all models

### Short-Term (Next 2 Weeks)

**Priority 2: Optuna Optimization**
- Implement Optuna with 100 trials
- Search space: lr, weight_decay, n_hid, adj_threshold
- Expected: R² ≥ 0.77

### Medium-Term (1-2 Months)

**Priority 3: Graph Attention Networks**
- Replace GCN with GAT
- Test adaptive neighbor selection
- Expected: R² ≥ 0.78

### Expected Performance Trajectory

| Stage | R² Target | Improvement |
|-------|-----------|-------------|
| Current (undertrained) | 0.722 | Baseline |
| After Priority 1 (ensemble) | 0.750 | +4.0% |
| After Priority 2 (Optuna) | 0.770 | +2.7% |
| After Priority 3 (GAT) | 0.780 | +1.3% |
| **Total improvement** | **0.780** | **+8.0%** |

### Risk Assessment

**Low risk:**
- Priority 1: Well-understood, proven approach
- Priority 2: Optuna is mature, extensively tested

**Medium risk:**
- Priority 3: GAT requires architectural changes
- Priority 4: GELU impact uncertain

**High risk:**
- Priority 5: Complex architecture, longer development

---

## Next Step

**Start Priority 1 immediately:**
```bash
cd D:\bmad-projects\luanvan_exp\moirai
python gnn/gnnhar_paper/train_multi_stock.py --model HAR --n_seeds 20 --epochs 400
```

**Expected completion:** Tomorrow morning (~9 hours overnight)

**Decision point after Priority 1:**
- If R² ≥ 0.75 → Proceed to Priority 2 (Optuna)
- If R² < 0.75 → Investigate convergence issues first

---

**Status:** Ready to execute
**First action:** Run full ensemble training tonight
