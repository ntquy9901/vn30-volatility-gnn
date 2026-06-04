# Single-Stock Training Failure Analysis

**Date:** 2026-05-31
**Status:** Neural networks fail, HAR OLS succeeds (R² = 0.63)
**Root Cause:** Architecture mismatch - paper uses multi-stock (N=30-100), we use single-stock (N=1)

---

## Experimental Results Summary

### Configuration
- **Model:** VIC single-stock forecasting
- **Horizon:** 5 days
- **Train period:** 2020-01-01 to 2025-12-31 (1260 samples)
- **Val period:** 2026-01-01 to 2026-04-30 (316 samples)
- **Test period:** 2026-01-01 to 2026-05-31 (109 samples)
- **Loss:** MSE (QL loss incompatible with NO ReLU)
- **Activation:** NO ReLU (QL loss + ReLU causes 75% seed failures)
- **Weight decay:** 1e-5 (paper's value, not 1e-3)

### Results (20 seeds, ensemble screening)

| Model | Ensemble R² | MAE | Status |
|-------|-------------|-----|--------|
| **HAR OLS** | **+0.6275** | **0.001117** | **Baseline** |
| HAR (NN) | -0.1412 | 0.002189 | Worse than baseline |
| GHAR | -12.9208 | 0.007720 | Catastrophic failure |
| GNNHAR1L | -1.3109 | 0.003078 | Bad |
| GNNHAR2L | -5.6841 | 0.005366 | Bad |
| GNNHAR3L | -7.6768 | 0.006083 | Catastrophic |

**Key findings:**
1. **HAR OLS works well:** R² = 0.63 matches paper's reported performance
2. **All neural networks fail:** 0/6 models beat HAR baseline
3. **High seed instability:** 18/20 seeds catastrophically overfit (R² = -100 to -8000)
4. **GHAR worst performer:** Projection layer adds instability without graph information

---

## Root Cause Analysis

### 1. Architecture Mismatch (Critical)

**Paper design (Multi-Stock):**
```python
# Data: 30-100 stocks × 1000+ dates
# Shape: (30000, 3) flattened stock-date pairs
node_feat: (batch_size, N=30, 3)  # N stocks
adj: (N, N) = (30, 30)  # Real correlation graph from GLASSO
```

**Our implementation (Single-Stock):**
```python
# Data: 1 stock × 1260 dates
# Shape: (1260, 3) single stock samples
node_feat: (batch_size, N=1, 3)  # Only 1 stock!
adj: (N, N) = (1, 1)  # Identity matrix, no graph structure!
```

**Impact:**
- GCN layers receive no neighborhood information (identity adjacency)
- Graph branch (H2) adds no value over linear HAR (H1)
- Extra parameters in GCN+MLP cause overfitting with 1260 samples
- Effective Sample Size (ESS) = 1260/5 = 253 for h=5, insufficient for deep models

### 2. QL Loss Incompatibility (Critical)

**Issue:** QL loss requires **positive predictions** (uses log of prediction ratio).

**With ReLU:**
- Predictions → 0 (ReLU forces non-negative)
- QL loss → -log(eps) ≈ 9.21 (constant, no gradients)
- 75% of seeds collapse to val_loss=9.2103

**Without ReLU:**
- Predictions → negative (unconstrained)
- QL loss → log(negative) = NaN
- Training diverges

**Conclusion:** QL loss fundamentally incompatible with single-stock training where ReLU causes collapse.

### 3. Target Scaling Bug (Fixed)

**Bug:** HAR baseline predictions were raw RV (not divided by horizon), but neural network targets were scaled (divided by horizon=5).

**Fix:** Scale HAR predictions by horizon to match neural network targets.
```python
har_pred_aligned = har_pred_aligned / HORIZON
```

**Result:** HAR OLS R² improved from -698 to +0.63 (correct baseline).

---

## Why Does HAR OLS Succeed Where Neural Networks Fail?

### HAR OLS: Stable and Effective

**Model:** `RV_t = α + β_d·RV_{t-1} + β_w·RV^(5)_t + β_m·RV^(20)_t`

**Advantages:**
1. **3 parameters only** → ESS=253, obs/param = 84 (well-identified)
2. **Closed-form OLS solution** → no optimization instability
3. **Linear model** → no activation function issues
4. **Proven architecture** → Corsi (2009), standard in volatility literature

**Performance:** R² = 0.63 on 2026 test data (matches paper's benchmarks)

### Neural Networks: Overparameterized and Unstable

**Models:** HAR (4 params), GHAR (69 params), GNNHAR1L (70 params), GNNHAR2L (118 params), GNNHAR3L (167 params)

**Disadvantages:**
1. **67-167 parameters** → ESS=253, obs/param = 1.5-3.8 (severe overfitting)
2. **Nonlinear optimization** → random seed sensitivity, local minima
3. **Graph structure unused** → GCN layers add parameters without information
4. **ReLU activation issues** → dying neurons without proper initialization

**Performance:** All models worse than HAR OLS, most catastrophically worse (R² = -1 to -13)

---

## Key Lessons Learned

### 1. GNNHAR Models Require Multi-Stock Data

The paper's architecture is fundamentally designed for cross-stock volatility spillover:

**Graph construction:**
- Adjacency from GLASSO (Graphical Lasso) on stock returns
- Edge weights = partial correlations
- Spillover effects: correlated stocks affect each other's volatility

**Single-stock limitation:**
- Identity adjacency matrix (no edges)
- No spillover information
- GCN layers reduce to linear transforms with extra parameters

**Conclusion:** Cannot replicate paper's results with single-stock data.

### 2. QL Loss Requires ReLU (Incompatible with Single-Stock)

**Paper's setup:**
- Multi-stock training stabilizes QL loss + ReLU
- Batch contains diverse stocks → predictions don't all collapse to 0
- Ensemble screening removes failed seeds

**Single-stock reality:**
- All samples from same stock → predictions either all 0 or all negative
- QL loss singularity at pred=0 (loss = -log(eps) ≈ 9.21)
- No ReLU → negative predictions → log(negative) = NaN

**Conclusion:** QL loss unusable for single-stock training.

### 3. HAR OLS is the Right Baseline

For single-stock volatility forecasting, **HAR-RV (Corsi 2009)** is the appropriate baseline:

- Simple, stable, well-established
- Requires minimal data (ESS=253 sufficient for 3 parameters)
- Performs well on test data (R² = 0.63)
- Matches paper's reported baseline performance

**Neural networks only add value when:**
1. Multi-stock data with real graph structure (N ≥ 30)
2. Sufficient ESS for model complexity (obs/param ≥ 10)
3. Stable loss function (MSE, not QL with ReLU)
4. Proper cross-stock batching and training

---

## Recommended Next Steps

### Option A: Accept HAR OLS as Final Result

**Rationale:**
- HAR OLS achieves paper's reported performance (R² = 0.63)
- Single-stock setup cannot leverage paper's graph architecture
- Neural networks add no value without multi-stock data

**Action:**
- Report HAR OLS as VIC volatility forecasting baseline
- Document why GNNHAR models fail (this analysis)
- Proceed to multi-stock experiments if graph effects needed

### Option B: Multi-Stock Training (Full Paper Replication)

**Rationale:**
- Only way to properly evaluate GNNHAR architecture
- Leverages cross-stock spillover effects
- Matches paper's experimental setup

**Requirements:**
1. Load all 30 VN30 stocks (not just VIC)
2. Construct correlation-based adjacency matrix (GLASSO or Pearson threshold)
3. Flatten data across stocks: (30 stocks × 1200 dates, 3 features)
4. Train with multi-stock batching (random stock-date pairs per batch)
5. Use QL loss + ReLU (paper's setup, stable with multi-stock)

**Expected outcome:** GNNHAR models should beat HAR baseline (R² > 0.63)

### Option C: Simplified Single-Stock Models

**Rationale:**
- If goal is single-stock forecasting only, skip graph architecture
- Use simpler models appropriate for ESS=253

**Models to try:**
1. **Ridge/Lasso HAR:** Linear HAR with regularization
2. **Small MLP:** 1-2 hidden layers, 8-16 neurons max
3. **LSTM-SISO:** Single-input single-output LSTM (proven in project)

**Avoid:**
- Graph architectures (no benefit without graph data)
- Deep models (>100 parameters for ESS=253)
- QL loss (unstable for single-stock)

---

## Conclusion

**Current status:** HAR OLS baseline working (R² = 0.63), neural networks failing.

**Root cause:** Architecture mismatch - paper designed for multi-stock with graph structure, we're using single-stock with identity adjacency.

**Recommendation:** Either accept HAR OLS as final result (single-stock setup) or implement full multi-stock training (paper replication). Hybrid approaches (single-stock + graph architecture) will continue to fail.

**Files modified:**
- `gnn/gnnhar_paper/gnnhar_models.py`: Removed ReLU from all models (line 98, 139, 202, 261, 326)
- `gnn/gnnhar_paper/vic/train_vic_ensemble.py`: Switched to MSE loss, fixed HAR baseline scaling, extended test period
- `gnn/gcn_layer.py`: Already has proper Xavier initialization (matching paper)

**Next decision point:** Pursue multi-stock training (Option B) or accept HAR OLS baseline (Option A)?
