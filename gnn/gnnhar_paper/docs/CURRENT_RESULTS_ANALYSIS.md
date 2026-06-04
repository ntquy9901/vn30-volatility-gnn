# Multi-Stock PyTorch GNNHAR Results Analysis (2026-05-31)

## Executive Summary

**Status:** PRELIMINARY RESULTS (1 seed only, not final ensemble)

Multi-stock PyTorch GNNHAR training has been tested with 3 models (HAR, GHAR, GNNHAR1L) using 1 random seed each. All models were trained with multi-stock data (30 stocks pooled, ~96,000 training samples) and early stopping.

**Key Finding:** Graph models (GHAR, GNNHAR1L) perform WORSE than baseline HAR model in preliminary testing, suggesting weak graph signal for VN30 volatility forecasting.

## Results Comparison

### PyTorch Models (Multi-Stock, 1 Seed)

| Model | Epochs Trained | R2 | MAE | RMSE | vs HAR |
|-------|----------------|-----|-----|------|--------|
| **HAR** | 20 (fixed) | **+0.7456** | **0.004254** | **0.006508** | baseline |
| **GHAR** | 189 (early stop) | +0.7331 | 0.004427 | 0.006667 | **-0.0125** |
| **GNNHAR1L** | 162 (early stop) | +0.7200 | 0.004616 | 0.006828 | **-0.0256** |

### sklearn Baselines (Multi-Stock, Full Training)

| Model | Features | R2 | MAE | vs PyTorch HAR |
|-------|----------|-----|-----|----------------|
| **HAR OLS** | 3 | +0.7532 | 0.004241 | +0.0076 |
| **GHAR (iden+pearson)** | 6 | +0.7538 | 0.004226 | +0.0082 |

## Detailed Analysis

### Finding 1: Graph Models Underperform Baseline

**Observation:** Both GHAR and GNNHAR1L perform worse than HAR baseline
- GHAR: R² = 0.7331 (Δ = -0.0125 vs HAR)
- GNNHAR1L: R² = 0.7200 (Δ = -0.0256 vs HAR)

**Possible Causes:**

1. **Insufficient Training:** HAR trained only 20 epochs (fixed limit), while GHAR/GNNHAR1L used early stopping (189/162 epochs). Graph models may need more epochs to converge.

2. **Overfitting:** Graph models have more parameters (GHAR: 69, GNNHAR1L: 70) vs HAR (4). With only 1 seed, no ensemble averaging, may overfit to training noise.

3. **Weak Graph Signal:** Consistent with sklearn GHAR results (+0.0006 improvement only). Pearson correlation graph may not capture meaningful volatility spillover for VN30 market.

4. **Regularization Mismatch:** PyTorch uses weight_decay=1e-5 (very light regularization). sklearn uses OLS (no regularization). Different optimization objectives may affect performance.

### Finding 2: sklearn Still Beats PyTorch

**Observation:** sklearn HAR OLS (R² = 0.7532) beats PyTorch HAR (R² = 0.7456) by Δ = +0.0076

**Possible Causes:**

1. **Loss Function Difference:**
   - sklearn: MSE loss (minimizes squared error directly)
   - PyTorch: QL loss (minimizes ratio-based error)

   QL loss is optimized for heteroskedasticity but may not minimize R² directly.

2. **Training Duration:**
   - sklearn: Closed-form OLS (instant, optimal solution)
   - PyTorch HAR: 20 epochs only (may not have converged)

3. **Random Initialization:**
   - PyTorch HAR uses random weight initialization
   - sklearn uses analytical solution (no randomness)
   - With 1 seed only, PyTorch may have poor initialization

### Finding 3: Early Stopping Patterns

**Observation:** Graph models trigger early stopping
- GHAR: 189 epochs (patience=150, stopped after 150 epochs without improvement)
- GNNHAR1L: 162 epochs

**Validation Loss Trends:**
- All models: val_loss ≈ 1.35 (similar convergence)
- No clear gap between train/val (suggests underfitting, not overfitting)

**Implication:** Graph models are not overfitting - they're struggling to find useful signal from the graph structure.

## Comparison with sklearn GHAR

### sklearn GHAR Results (Residual Design)

| Configuration | R2 | Improvement over HAR OLS |
|--------------|-----|--------------------------|
| HAR OLS (per-stock) | +0.7532 | baseline |
| GHAR (iden only) | +0.7529 | -0.0003 |
| GHAR (iden+pearson, thresh=0.1) | +0.2761 | -0.4771 (catastrophic) |
| GHAR (iden+pearson, thresh=0.3) | **+0.7538** | **+0.0006** |
| GHAR (iden+pearson, thresh=0.5) | +0.7537 | +0.0005 |
| GHAR (iden+pearson, thresh=0.7) | +0.7527 | -0.0005 |
| GHAR (iden+glasso) | +0.7529 | -0.0003 |

### sklearn vs PyTorch GHAR

| Aspect | sklearn GHAR | PyTorch GHAR |
|--------|-------------|--------------|
| **Architecture** | Linear on graph-transformed features | GCN layer in neural network |
| **Training** | OLS (closed-form) | QL loss (gradient descent) |
| **Graph usage** | Static feature transformation | Learned weighting |
| **R² (thresh=0.3)** | +0.7538 | +0.7331 |
| **Difference** | **baseline** | **-0.0207 worse** |

**Why PyTorch GHAR Worse:**

1. **Learned vs Fixed Weights:**
   - sklearn: Uses fixed Pearson correlation weights (interpretable, stable)
   - PyTorch: Learns weights through gradient descent (may converge to suboptimal solution)

2. **Loss Function Mismatch:**
   - sklearn OLS directly optimizes MSE (maximizes R²)
   - PyTorch QL loss optimizes ratio-based error (different objective)

3. **Regularization:**
   - sklearn: No regularization (unconstrained OLS)
   - PyTorch: weight_decay=1e-5 (very light, but may still constrain)

## Graph Signal Analysis

### Correlation Analysis

**Return Correlation → RV Correlation:**
- Correlation: +0.63 (strong relationship)
- This suggests stock return correlations predict volatility correlations

**But Forecasting Improvement:**
- sklearn GHAR: Only +0.0006 R² improvement
- PyTorch GHAR: -0.0207 R² degradation

**Interpretation:** While correlated stocks exist, the correlation structure doesn't provide useful forecasting signal for future volatility. Possible reasons:

1. **Lag Effect:** Correlations may exist contemporaneously, but not predictively
2. **Instability:** 60-day correlation window may not capture stable spillover patterns
3. **Horizon Mismatch:** h=5 (1 week) may be too short for spillover effects to manifest
4. **Market Structure:** VN30 may have weaker sector-based volatility transmission than US markets

## Limitations of Current Results

### Major Limitations

1. **Single Seed Only:**
   - Results are from 1 random seed each
   - High variance due to random initialization
   - No ensemble averaging (paper uses 20 seeds, screens top 50%)

2. **Insufficient Training:**
   - HAR: 20 epochs (may not have converged)
   - GHAR/GNNHAR1L: Early stopping at 162-189 epochs
   - Paper trains up to 1500 epochs (8-10x more)

3. **No Hyperparameter Tuning:**
   - Using paper's default hyperparameters (lr=1e-3, weight_decay=1e-5, n_hid=16)
   - These may not be optimal for VN30 data

4. **Single Graph Configuration:**
   - Only tested Pearson correlation with threshold=0.3 (68% density)
   - sklearn found optimal threshold=0.3, but PyTorch may prefer different density

### What Results Tell Us

**Reliable Conclusions:**
- Multi-stock training pipeline works correctly (no bugs)
- Graph signal is WEAK for VN30 (consistent with sklearn)
- PyTorch models converge without instability

**Unreliable Conclusions:**
- Exact R² values (high variance with 1 seed)
- Whether GHAR/GNNHAR1L truly worse than HAR (need ensemble)
- Optimal graph configuration for PyTorch

## Next Steps

### Immediate Actions

1. **Train Full HAR Ensemble (20 seeds, 1500 epochs)**
   - Establish solid PyTorch baseline
   - Compare with sklearn HAR OLS (R² = 0.7532)
   - Expected: PyTorch HAR should approach sklearn performance

2. **Train Full GHAR Ensemble (20 seeds, 1500 epochs)**
   - Test if learned graph weights beat fixed Pearson
   - Compare with sklearn GHAR (R² = 0.7538)
   - Expected: If graph signal exists, PyTorch should match or beat sklearn

3. **Train Full GNNHAR1L Ensemble (20 seeds, 1500 epochs)**
   - Test if nonlinear spillover helps
   - Expected: Should beat GHAR if nonlinear patterns exist

### Conditional Actions (based on results)

**If PyTorch GHAR beats sklearn GHAR (R² > 0.754):**
- Conclusion: Learned graph weights add value
- Proceed to GNNHAR2L, GNNHAR3L
- Test different graph thresholds (0.1, 0.3, 0.5, 0.7)

**If PyTorch GHAR matches sklearn GHAR (R² ≈ 0.753-0.754):**
- Conclusion: Learned weights ≈ fixed weights (no advantage)
- GNNHAR1L may still show improvement from nonlinearity
- Consider both approaches valid

**If PyTorch GHAR worse than sklearn GHAR (R² < 0.753):**
- Conclusion: QL loss or PyTorch optimization unsuitable for this task
- Fall back to sklearn GHAR as final graph model
- Consider using MSE loss in PyTorch instead of QL loss

## Expected Timeline

**Full Ensemble Training (20 seeds × 1500 epochs):**

| Model | Estimated Time | Status |
|-------|---------------|--------|
| HAR | ~2 hours | Ready to start |
| GHAR | ~3 hours | Ready to start |
| GNNHAR1L | ~3 hours | Ready to start |
| **Total** | **~8 hours** | 1 day of compute |

**Command to Run:**
```bash
# Run sequentially (recommended for stability)
python gnn/gnnhar_paper/train_multi_stock.py --model HAR --n_seeds 20 --epochs 1500
python gnn/gnnhar_paper/train_multi_stock.py --model GHAR --n_seeds 20 --epochs 1500
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --epochs 1500
```

## Conclusion

Preliminary results show weak graph signal for VN30 volatility forecasting, consistent with sklearn GHAR findings (+0.0006 improvement only). However, these results are unreliable due to:

1. Single seed (high variance)
2. Insufficient training epochs
3. No ensemble averaging

**Verdict:** Results are inconclusive. Full ensemble training (20 seeds, 1500 epochs) is required to draw reliable conclusions about PyTorch GNNHAR performance relative to sklearn baselines.

**Hypothesis:** Given sklearn GHAR's minimal improvement (+0.0006), PyTorch models are expected to show similar weak graph signal. The best case is PyTorch GHAR matching sklearn GHAR (R² ≈ 0.754), with GNNHAR1L potentially showing marginal improvement from nonlinearity (R² ≈ 0.755-0.760).
