# GNNHAR1L Improvement Plan — Integrated BMAD Agent Recommendations

**Date:** 2026-06-03  
**Model:** GNNHAR1L (Graph Neural Network HAR, 1-layer)  
**Current Performance:** R²=0.7472 (+0.68% over HAR baseline)  
**Status:** v1.3_LOSS_FIX (critical bug resolved)  
**Planning Method:** BMAD Party Mode — Multi-agent collaborative analysis

---

## Executive Summary

**Key Finding:** Current GNNHAR1L implementation has both **code quality issues** (CONTRAINTS.md violations, bugs) and **architectural improvement opportunities**, but the highest-value path focuses on **robustness and analysis** rather than chasing marginal R² gains.

**Agent Consensus:**
- 🏗️ **Winston (Architect):** Systematic, incremental improvements with ESS constraints
- 💻 **Amelia (Engineer):** Fix code bugs and CONSTRAINTS.md violations immediately  
- 📋 **John (PM):** Focus on thesis value, not just metric improvements
- 📊 **Mary (Analyst):** Deepen theoretical analysis rather than chase small R² gains

**Recommended Approach:** 3-phase plan balancing code quality, architectural improvements, and thesis value within timeline constraints.

---


## 📝 MANDATORY: File Naming & Timestamp Convention

**Status:** ✅ **ENFORCED** — All new code, results, and documents MUST follow this convention  
**Rationale:** Prevent file overwrites, ensure reproducibility, maintain experiment traceability  
**Scope:** Model checkpoints, learning curves, result JSONs, analysis scripts, documentation

### Timestamp Format

**Standard format:** `YYYYMMDD_HHMMSS` (e.g., `20260603_213000`)

```python
# Python implementation
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### File Naming Template

**Pattern:** `{prefix}_{description}_{timestamp}.{ext}`

**Examples:**
- **Learning curves:** `GNNHAR1L_seed42_learning_curve_20260603_213000.png`
- **Results JSON:** `GNNHAR1L_gelu_h5_20260603_213000.json`
- **Model checkpoints:** `GATHAR1L_best_model_20260603_213000.pt`
- **Analysis scripts:** `analyze_attention_weights_20260603_213000.py`
- **Documentation:** `experiment_report_v4_20260603_213000.md`

### Mandatory Files (Must Include Timestamps)

1. **Training Outputs:**
   - ✅ Learning curve PNGs: `{model}_seed{seed}_learning_curve_{timestamp}.png`
   - ✅ Results JSONs: `{model}_{activation}_h{horizon}_{timestamp}.json`
   - ✅ Model checkpoints: `{model}_seed{seed}_checkpoint_{timestamp}.pt`

2. **Analysis Outputs:**
   - ✅ Per-stock analysis: `per_stock_r2_{timestamp}.csv`
   - ✅ Attention weights: `attention_weights_{timestamp}.npz`
   - ✅ Comparison charts: `{model1}_vs_{model2}_comparison_{timestamp}.png`

3. **Documentation:**
   - ✅ Experiment reports: `experiment_report_{version}_{timestamp}.md`
   - ✅ Analysis notebooks: `{analysis_topic}_{timestamp}.ipynb`

### Verification Checklist

Before committing any experiment code, verify:

- [ ] All output files include timestamp in filename
- [ ] Timestamp format matches `YYYYMMDD_HHMMSS` standard
- [ ] Files from same training run share same timestamp
- [ ] No hardcoded paths without timestamps
- [ ] Old files are never overwritten by new runs

### Implementation Example (train_multi_stock.py)

```python
# CORRECT: Learning curve with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plot_file = save_path / f'{model_name}_seed{seed}_learning_curve_{timestamp}.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"  Saved learning curve: {plot_file}")

# CORRECT: Results JSON with timestamp
results_file = save_path / f'{model_name}_{activation}_h{horizon}_{timestamp}.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to: {results_file}")
```

### Current Compliance Status

**Already Compliant:**
- ✅ `train_multi_stock.py` — Lines 89-93, timestamps in learning curves
- ✅ Result JSONs — Use timestamps in all outputs

**Needs Enforcement:**
- 🔄 New GATHAR1L analysis scripts (must follow convention)
- 🔄 Manual analysis plots (must include timestamps)
- 🔄 Model checkpoints (if saving intermediate models)

**Non-Compliant Examples (DO NOT USE):**
- ❌ `learning_curve.png` (no timestamp)
- ❌ `results.json` (no timestamp)
- ❌ `model.pt` (no timestamp)
- ❌ `experiment_report.md` (no timestamp)

---



## Phase 1: Code Quality & CONSTRAINTS.md Compliance (Week 1)

**Priority:** HIGH — Must fix before any architectural experiments

### 1.1 Critical Bug Fixes (Amelia's Findings)

**C1 — Multi-horizon mask handling bug:**
```python
# File: gnngnnhar_paper/train_multi_stock.py, line ~120
# BUG: Current mask expansion fails for HORIZONS=[1,5,10,20]
mask_expanded = mask.unsqueeze(-1).expand_as(predictions)  # Wrong

# FIX:
mask_expanded = mask.unsqueeze(-1).expand(-1, len(HORIZONS))  # Explicit (n_stocks, 4)
assert mask_expanded.shape == predictions.shape  # Validate correctness
```

**C2 — Memory leak in training loop:**
```python
# File: gnngnnhar_paper/train_multi_stock.py, line ~280
# BUG: Unbounded tensor retention causes memory growth
all_predictions.append(predictions.detach().cpu())  # Don't do this

# FIX: Pre-allocate numpy arrays
val_preds = np.zeros((n_val_samples, n_stocks, len(HORIZONS)))
val_preds[i_batch] = predictions.detach().cpu().numpy()
```

### 1.2 CONSTRAINTS.md Violations (Amelia's Findings)

**C3 — Per-horizon loss printing (R2 violation):**
```python
# Add to train_single_model() after line ~260
for h_idx, h in enumerate(HORIZONS):
    h_loss = F.mse_loss(predictions[:, h_idx], targets[:, h_idx])
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | h={h:2d} loss={h_loss:.6f}")
```

**C4 — Data split console output (R3 violation):**
```python
# Add to train_ensemble() after line ~400
print(f"[DATA SPLIT] Train: {train_start} to {train_end} ({n_train} samples)")
print(f"[DATA SPLIT] Val:   {val_start} to {val_end} ({n_val} samples)")
print(f"[DATA SPLIT] ESS:   {n_train // max(HORIZONS)} per stock")
```

**C5 — Learning curve saving (R2 violation):**
```python
# Fix in plot_learning_curves() line ~350
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"GNNHAR1L_learning_curve_{timestamp}.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"[OK] Saved learning curve to {filename}")
```

### 1.3 Testing Infrastructure (Amelia's Recommendation)

**C7 — Add unit tests for mask logic:**
```python
# File: gnngnnhar_paper/tests/test_train_multi_stock.py
def test_mask_expansion():
    n_stocks, n_horizons = 5, 4
    mask = torch.ones(n_stocks, dtype=bool)
    mask_expanded = mask.unsqueeze(-1).expand(-1, n_horizons)
    assert mask_expanded.shape == (n_stocks, n_horizons)
```

**Timeline:** 3-4 days  
**Risk:** LOW — Well-defined fixes with clear validation  
**Thesis Value:** Essential for reproducibility and correctness

---

## Phase 2: Baseline Sanity Check Baseline Sanity Check & Diagnostic Analysis Diagnostic Analysis (Enhanced with GATHAR1L Comparison) (Week 1-2)

**Priority:** HIGH — Understanding what we're optimizing before changing it

### 2.1 Ablation Study (Winston's Phase 1)

**Test each component's contribution:**
1. **GCN-only:** Remove MLP, keep GCN + residual
2. **MLP-only:** Remove GCN, keep HAR + MLP  
3. **No residual:** Remove H1+H2 sum, use only H2
4. **No dropout:** Set dropout=0.0 vs 0.1, 0.2, 0.3
5. **Activation comparison:** ReLU vs GELU vs SiLU

**Expected outcomes:**
- Quantify graph contribution (+0.20% R² already observed)
- Quantify nonlinearity contribution (+0.48% R² already observed)
- Identify if any component is redundant (negative contribution)

### 2.2 Per-Stock Performance Analysis (Mary's Recommendation)

**Research question:** Do some stocks benefit more from graph features than others?

**Analysis:**
```python
# Compute per-stock R² scores
for ticker in VN30_TICKERS:
    stock_mask = (test_stocks == ticker)
    stock_r2 = r2_score(test_y[stock_mask], test_pred[stock_mask])
    print(f"{ticker}: R² = {stock_r2:.4f}")
```

**Expected patterns:**
- Sector clustering (banking stocks might benefit more from other banking stocks)
- High-volatility stocks (GAS, NVL) might benefit more from graph smoothing
- Market leaders (VCB, VIC) might benefit less (already capture market signal)

### 2.3 Regime-Dependent Analysis (Mary's Phase 1)

**Research question:** Does graph value increase during volatility crises?

**Analysis:**
```python
# Identify high-volatility periods
rv_threshold = test_y.quantile(0.9)
crisis_mask = test_y > rv_threshold

# Compute R² during crisis vs normal times
crisis_r2 = r2_score(test_y[crisis_mask], test_pred[crisis_mask])
normal_r2 = r2_score(test_y[~crisis_mask], test_pred[~crisis_mask])
```

**Expected patterns:**
- Graph features might help more during crises (contagion effects)
- HAR baseline might struggle with regime shifts
- This could strengthen thesis contribution significantly

### 2.4 Temporal Validation (Winston's Concern)

**Research question:** Does GNNHAR1L generalize to truly held-out time periods?

**Analysis:**
```python
# Train on pre-COVID, test on COVID and post-COVID
train_end = "2019-12-31"
val_start = "2020-01-01"
test_start = "2022-01-01"
```

**Risk assessment:**
- If +0.68% vanishes on strict temporal split, current results may be optimistic
- If improvement holds, much stronger thesis case
- Essential for academic defensibility

**Timeline:** 4-5 days  
**Risk:** LOW — No code changes, just analysis  
**Thesis Value:** HIGH — Strengthens methodology section significantly

---


### 2.5 Graph Architecture Comparison: GNNHAR1L vs GATHAR1L (NEW)

**Status:** 🆕 **NEW ADDITION** - Comprehensive graph mechanism analysis  
**Priority:** HIGH — Understanding which graph mechanism works best for volatility

**Research Questions:**
1. **Comparison Baseline:** Does attention (GAT) outperform averaging (GCN)?
2. **Architectural Improvement:** Can GATHAR1L beat GNNHAR1L performance?
3. **Ablation Study:** Does attention provide interpretable stock relationships?

**GATHAR1L Architecture:**
```python
# Graph Attention HAR - 1-layer with attention mechanism
H1 = Linear(3, 1)(node_feat)              # Local HAR
H2 = GAT(3, n_hid, heads=1)(node_feat, adj) # Attention-weighted neighbors
H2 = activation(H2)                        # Nonlinearity
H2 = MLP(n_hid, 1)(H2)                    # Projection
output = H1 + H2                          # Residual connection
```

**Key Differences from GNNHAR1L:**
- **GCN (GNNHAR1L):** Averages all neighbors equally → `(batch, N, n_hid)`
- **GAT (GATHAR1L):** Learns attention weights per neighbor → `(batch, N, heads, n_hid)`

**Implementation Strategy (Lightweight):**
```python
# Use PyTorch Geometric GATConv layer
from torch_geometric.nn import GATConv

class GATHAR1L(nn.Module):
    def __init__(self, n_hid=16, heads=1, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gat1 = GATConv(3, n_hid, heads=heads, dropout=dropout, concat=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.activation = nn.GELU()
        
    def forward(self, node_feat, adj, edge_index):
        H1 = self.linear1(node_feat)
        H2 = self.gat1(node_feat, edge_index)
        H2 = self.activation(H2)
        H2 = self.dropout(H2)
        H2 = self.mlp1(H2)
        return (H1 + H2).squeeze(-1)
```

**Training Configuration (Same as GNNHAR1L):**
- Horizon: h=5 (same as baseline comparison)
- Ensemble: 20 seeds, screen top 50%
- Loss: gnnhar_ratio_loss (identical)
- Activation: GELU (same)
- Dropout: 0.1 (same)

**Success Criteria (Comprehensive):**

**A. Performance Improvement (Beat GNNHAR1L):**
- Target: R² > 0.7472 (current GNNHAR1L baseline)
- Threshold: R² > 0.75 (1% improvement) to justify attention mechanism
- Metrics: R², MAE, RMSE comparison with GNNHAR1L

**B. Interpretability Gain (Attention Weights):**
```python
# Extract attention weights for interpretability
def extract_attention_insights(gat_model, edge_index, stock_names):
    """Identify which stocks drive volatility for each target."""
    attention_weights = gat_model.get_attention_weights(edge_index)
    
    # For each stock, rank top-5 influencers
    for i, stock in enumerate(stock_names):
        top_influencers = attention_weights[i].topk(5)
        print(f"{stock} most influenced by: {top_influencers}")
```

**Expected Insights:**
- Sector leaders (VCB, VIC) likely have high attention weights
- High-volatility stocks (GAS, NVL) may be attention hubs
- Cross-sector dependencies (banking → real estate)

**C. Robustness Check (Graph Noise Tolerance):**
```python
# Test robustness to graph noise
robustness_test = {
    "original_adj": test_performance,
    "noisy_adj_10%": test_with_perturbed_adj(0.1),
    "noisy_adj_20%": test_with_perturbed_adj(0.2),
}
```

**Expected Results:**
- If GAT > GCN robustness: Attention adapts to graph noise better
- If GAT < GCN robustness: Simple averaging is more stable
- Quantify robustness gap: % performance drop under noise

**Comparison Matrix:**
| Model | Mechanism | Expected R² | Interpretability | Robustness | Params |
|-------|-----------|-------------|------------------|-------------|--------|
| HAR | Linear only | 0.7421 | ✅ High (3 weights) | ✅ High | 3 |
| GHAR | GCN averaging | 0.7436 | ❌ Low (black box) | ✅ High | ~50 |
| GNNHAR1L | GCN + MLP | 0.7472 | ❌ Low (black box) | ✅ Medium | ~400 |
| GATHAR1L | GAT attention | **?** | ✅ **High (weights)** | **?** | ~450 |

**Thesis Value:**
- **If GAT wins:** "Attention mechanisms capture volatility spillovers better than averaging"
- **If GCN wins:** "Simple neighbor averaging suffices for volatility forecasting"
- **Either outcome:** Strong theoretical contribution about graph mechanisms

**Timeline:** 3-4 days (implementation + training + analysis)  
**Risk:** MEDIUM — New architecture, but uses proven GAT layers  
**Thesis Value:** VERY HIGH — Resolves graph mechanism question definitively
## Phase 3: Architectural Improvements (Week 2-3)

**Priority:** MEDIUM — Only if Phase 2 shows clear improvement opportunities

### 3.1 Graph-Specific Regularization (Winston's Option C)

**DropEdge Implementation:**
```python
# In forward_pass_with_mask(), add edge dropout
def apply_dropedge(adj, drop_rate=0.2):
    """Randomly remove edges to prevent overfitting to graph structure."""
    edge_mask = torch.rand(adj.shape, device=adj.device) > drop_rate
    return adj * edge_mask
```

**Benefits:**
- Forces robustness to graph noise (non-stationary correlations)
- Regularization without adding parameters
- Proven technique in GNN literature

**Temporal Smoothness Penalty:**
```python
# Add to loss function in train_single_model()
def temporal_smoothness_loss(predictions, targets, lambda_smooth=0.01):
    """Penalize large jumps in consecutive predictions."""
    time_diff = predictions[1:] - predictions[:-1]
    return lambda_smooth * torch.mean(time_diff**2)

total_loss = gnnhar_ratio_loss(pred, target) + temporal_smoothness_loss(pred, target)
```

**Benefits:**
- Volatility is temporally autocorrelated — predictions shouldn't jump
- Regularizes without changing model architecture
- Well-motivated for financial time series

### 3.2 Cautious Graph Deepening (Winston's Phase 3)

**2-layer GCN Test:**
```python
# Create GNNHAR2L variant and test
# File: gnngnnhar_paper/gnnhar_models.py (already exists)
# Test if 2nd layer helps or causes over-smoothing
```

**Over-smoothing check:**
```python
# Measure pairwise similarity of node embeddings
def check_over_smoothing(embeddings):
    """If all embeddings become similar, over-smoothing occurred."""
    similarity = F.cosine_similarity(embeddings.unsqueeze(1), 
                                   embeddings.unsqueeze(0), dim=2)
    return similarity.mean().item()  # Should stay < 0.8
```

**Stop criterion:** If 2-layer doesn't improve R² by >1%, don't proceed to 3-layer.

### 3.3 Gradient Clipping Adjustment (Amelia's C6)

**Current issue:** `clip_value=1.0` too aggressive for GNN gradients

**Fix:**
```python
# In train_single_model() line ~240
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # GNN-friendly
```

**Timeline:** 5-7 days  
**Risk:** MEDIUM — May not improve performance if graph is already well-specified  
**Thesis Value:** MEDIUM — Shows systematic experimentation, but diminishing returns

---

## Success Criteria & Stop Conditions

### When to Stop Improving (Winston's Guidance)

**Metric-based stopping:**
- If R² > 0.78 (5% over baseline): **SHIP IT** — don't over-optimize
- If 2-week architectural search yields <1% improvement: **STOP** — diminishing returns
- If validation variance increases: **STOP** — overfitting risk

### Thesis Success Metrics (John's Questions)

**Defendable thesis requires:**
1. **Clear contribution statement:** "Graph structure + neural modeling improves volatility forecasting"
2. **Statistical significance:** Diebold-Mariano test showing GNNHAR ≠ HAR (p < 0.05)
3. **Theoretical grounding:** WHY does graph help? (regime analysis, sector effects)
4. **Reproducibility:** All code fixes, tests, and documented artifacts

**NOT required for thesis success:**
- Beating HAR by large margins (baseline is already strong)
- Complex architectures if simple ones work
- State-of-the-art performance (volatility is inherently stochastic)

---

## Updated Implementation Priority Matrix (Enhanced with GATHAR1L)
## Updated Implementation Priority Matrix (Enhanced with GATHAR1L)

| Priority | Improvement | Impact | Effort | Risk | Thesis Value | Status |
|----------|-------------|--------|--------|------|--------------|---------|
| **P0** | Fix C1-C7 (all bugs) | HIGH | 3 days | LOW | Essential | ✅ DONE |
| **P1** | Phase 2: Core diagnostics | HIGH | 4 days | LOW | HIGH | 🔄 READY |
| **P1** | Phase 2.5: GATHAR1L comparison | **HIGH** | **3-4 days** | **MEDIUM** | **VERY HIGH** | **🔄 READY** |
| **P1** | Phase 2: Temporal validation | HIGH | 1 day | LOW | HIGH | 🔄 READY |
| **P2** | Phase 3: DropEdge | MEDIUM | 2 days | LOW | MEDIUM | ⏳ COND. |
| **P2** | Phase 3: Temporal smoothness | MEDIUM | 1 day | LOW | MEDIUM | ⏳ COND. |
| **P3** | Phase 3: 2-layer GCN | LOW | 2 days | MEDIUM | LOW | ⏳ COND. |

**Timeline Summary:** Phase 1 ✅ COMPLETE → Phase 2 🔄 READY (7-9 days with GATHAR1L) → Phase 3 ⏳ CONDITIONAL  
**Critical Path:** Phase 2 analysis + GATHAR1L comparison → decide on Phase 3 based on findings

---

## Risk Assessment

### Technical Risks (Winston's Concerns)

**R1 — ESS Starvation:**
- Current: ESS=123 per stock, parameters~400
- Adding layers: ESS/params ratio degrades further
- **Mitigation:** Monitor validation performance, stop if overfitting detected

**R2 — Graph Data Leakage:**
- Risk: Adjacency matrix uses future correlations
- **Mitigation:** Verify graph built on training-window stats only

**R3 — Temporal Generalization:**
- Risk: +0.68% vanishes on strict temporal split
- **Mitigation:** Phase 2 temporal validation will reveal this early

### Timeline Risks (John's Concerns)

**R4 — Architectural Over-Engineering:**
- Risk: Spend 3 weeks on complex improvements for 0.2% R² gain
- **Mitigation:** Success criteria defined, stop conditions enforced

**R5 — Lost Thesis Focus:**
- Risk: Optimization focus replaces thesis contribution focus
- **Mitigation:** Weekly check: "Does this strengthen the thesis story?"

---

## Decision Framework for Proceeding

### After Phase 1 (Code Quality):
- **If bugs found:** Fix immediately, no architectural work until stable
- **If no bugs:** Proceed to Phase 2 with confidence in baseline

### After Phase 2 (Diagnostics):
- **If per-stock analysis shows strong patterns:** Focus on regime/sector analysis
- **If temporal validation fails:** Reconsider entire approach (graph may be overfitting)
- **If ablation shows redundant components:** Simplify before adding complexity

### After Phase 3 (Architecture):
- **If DropEdge adds >1% R²:** Ship with graph regularization
- **If 2-layer GCN fails:** Don't try 3-layer (accept 1-layer as optimal depth)
- **If total improvement <2%:** STOP — focus on thesis writing instead

---

## Open Questions for User (John's Detective Work)

1. **What's the thesis deadline?** (Determines how aggressive to be with improvements)
2. **What's the defendable contribution?** (Better R² vs. better understanding vs. both?)
3. **What's the committee's expectation?** (State-of-the-art vs. solid contribution vs. novel approach?)
4. **How much time is left for writing?** (Optimization shouldn't crowd out analysis)

---

## Agent Consensus Statement

**All BMAD agents agree on the following:**

1. **Code quality fixes (Phase 1) are non-negotiable** — Must fix CONSTRAINTS.md violations before any optimization
2. **Diagnostic analysis (Phase 2) is essential** — Don't optimize what you don't understand
3. **Architectural improvements (Phase 3) are optional** — Only proceed if Phase 2 shows clear opportunity
4. **Thesis value > metric optimization** — Focus on defendable contribution, not max R²

**The Path Forward:**
Start with Phase 1 (code fixes) → Phase 2 (diagnostics) → Decide on Phase 3 based on evidence.

**The Boring, Stable Answer:** Fix the code, understand the baseline, make incremental improvements with clear stopping criteria. Don't chase exotic architectures for marginal gains when thesis success depends on clear contribution and reproducible science.

---
## Updated Next Steps

**Current Status:**
1. ✅ Phase 1 complete — All bugs C1-C7 fixed and verified
2. 🔄 Phase 2 ready — Enhanced with GATHAR1L comparison (7-9 days total)
3. ⏳ Phase 3 conditional — Dependent on Phase 2 findings

**Recommended Actions:**
1. ✅ **START Phase 2.1-2.4** — Core diagnostics (ablation, per-stock, regime, temporal)
2. 🆕 **IMPLEMENT GATHAR1L** — Lightweight GAT variant using PyTorch Geometric (3-4 days)
3. 🔄 **COMPARE GNNHAR1L vs GATHAR1L** — Performance + interpretability + robustness analysis
4. 🔄 **DECIDE on Phase 3** — Based on comprehensive Phase 2 evidence

**Updated Timeline:**
- Phase 1: ✅ COMPLETE (3-4 days)
- Phase 2.1-2.4: Core diagnostics (4-5 days)
- Phase 2.5: GATHAR1L comparison (3-4 days)
- Phase 3: ⏳ CONDITIONAL (5-7 days, IF justified by Phase 2 findings)
- **Total remaining:** 1.5-2 weeks for comprehensive analysis + conditional improvements

**GATHAR1L Success Impact:**
- **If GAT wins (R² > 0.75):** Use GATHAR1L as primary model, GNNHAR1L as baseline comparison
- **If GCN wins (R² > GAT):** Use GNNHAR1L as primary model, explain why averaging beats attention
- **Either outcome:** Strong thesis contribution on graph mechanism selection

---

**Generated:** 2026-06-03  
**Updated:** 2026-06-03 (Phase 1 completed, Phase 2 enhanced with GATHAR1L)  
**Method:** BMAD Party Mode — Multi-agent collaborative planning  
**Status:** Phase 1 ✅ COMPLETE — Phase 2 🔄 READY (enhanced with GATHAR1L comparison)
