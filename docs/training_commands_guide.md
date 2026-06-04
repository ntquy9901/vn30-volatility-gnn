# Self-Training Commands Guide (v1.3_LOSS_FIX + Optuna Best Configs)

**Purpose:** Comprehensive command reference for training GNNHAR models with corrected loss function, architectural guardrails, and Optuna-optimized hyperparameters.

**Version:** v1.3_LOSS_FIX + Optuna Best  
**Last Updated:** 2026-06-04  
**Status:** Production Ready with Best Configurations

---

## 🏆 **BEST CONFIGURATIONS FROM OPTUNA STUDIES**

### **Configuration #1: ReLU + Graphical Lasso (OVERALL BEST)** ⭐

**Validation R²:** 0.8001  
**Study:** 2026-06-03 (Trial #77)

```bash
python gnn/gnnhar_paper/train_multi_stock.py \
  --model GNNHAR1L \
  --n_seeds 20 \
  --n_hid 32 \
  --epochs 400 \
  --lr 0.00012934 \
  --weight_decay 0.0000607 \
  --batch_size 256 \
  --horizon 5 \
  --activation relu \
  --adj_method glasso \
  --dropout 0.1757 \
  --grad_clip 1.0
```

**Why this is best:**
- Graphical Lasso adjacency captures complex correlations better than Pearson
- Higher dropout (0.176) provides better regularization  
- Lower learning rate prevents overfitting
- **+1.8% better than GELU config**

**All horizons with Config #1:**
```bash
# H=1
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --n_hid 32 --lr 0.00012934 --weight_decay 0.0000607 --batch_size 256 --horizon 1 --activation relu --adj_method glasso --dropout 0.1757

# H=5
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --n_hid 32 --lr 0.00012934 --weight_decay 0.0000607 --batch_size 256 --horizon 5 --activation relu --adj_method glasso --dropout 0.1757

# H=10
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --n_hid 32 --lr 0.00012934 --weight_decay 0.0000607 --batch_size 256 --horizon 10 --activation relu --adj_method glasso --dropout 0.1757

# H=20
python gnn/gnnhar_paper/train_multi_stock.py --model GNNHAR1L --n_seeds 20 --n_hid 32 --lr 0.00012934 --weight_decay 0.0000607 --batch_size 256 --horizon 20 --activation relu --adj_method glasso --dropout 0.1757
```

---

### **Configuration #2: GELU + Pearson**

**Validation R²:** 0.7856  
**Study:** 2026-06-01 (Trial #47)

```bash
python gnn/gnnhar_paper/train_multi_stock.py \
  --model GNNHAR1L \
  --n_seeds 20 \
  --n_hid 32 \
  --epochs 400 \
  --lr 0.002357 \
  --weight_decay 0.000001389 \
  --batch_size 512 \
  --horizon 5 \
  --activation gelu \
  --adj_method pearson \
  --adj_threshold 0.367 \
  --dropout 0.1301 \
  --grad_clip 1.0
```

**Why use this:**
- GELU activation may generalize better to unseen data
- Faster training (higher learning rate, larger batch size)
- Simpler adjacency construction (Pearson is faster than GLasso)

---

### **Quick Comparison:**

| Config | Val R² | Adjacency | Activation | Speed | Recommendation |
|--------|--------|-----------|------------|-------|----------------|
| **#1** | **0.8001** | **GLasso** | **ReLU** | Slower | **⭐ Use for thesis** |
| #2 | 0.7856 | Pearson | GELU | Faster | Use if GLasso too slow |

**Recommendation:** Use **Config #1** for thesis results - it's the absolute best performer.

---

## 🚀 Quick Test Commands (5-10 minutes)

### Quick Verification Test
```bash
# Quick test: Verify guardrails work (1 seed, 50 epochs)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_seeds 1 \
    --epochs 50 \
    --n_hid 32 \
    --grad_clip 1.0
```

### Test All Models
```bash
# Quick test: All models with guardrails (1 seed each, 20 epochs)
for model in HAR GHAR GNNHAR1L GNNHAR2L GNNHAR3L; do
    python gnn/gnnhar_paper/train_multi_stock.py \
        --model $model \
        --n_seeds 1 \
        --epochs 20 \
        --grad_clip 1.0
done
```

**Purpose:** Verify that corrected loss function and guardrails work correctly before committing to long training runs.

---

## 🎯 Production Training Commands (2-4 hours each)

### ⭐ BEST: GNNHAR1L + ReLU + Graphical Lasso (Optuna Winner)

**Validation R²:** 0.8001 (Optuna Study 2026-06-03, Trial #77)

```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation relu \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.00012934 \
    --weight_decay 0.0000607 \
    --n_hid 32 \
    --batch_size 256 \
    --horizon 5 \
    --adj_method glasso \
    --dropout 0.1757 \
    --grad_clip 1.0
```

**Use when:** You want the BEST performance - this config achieved Val R²=0.8001, beating GELU by +1.8%.

**Key advantages:**
- Graphical Lasso (glasso) adjacency outperforms Pearson correlation
- Higher dropout (0.176) prevents overfitting
- Lower learning rate (0.000129) for stable training

---

### GNNHAR1L + GELU + Optuna Parameters

**Validation R²:** 0.7856 (Optuna Study 2026-06-01, Trial #47)

```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.0024 \
    --weight_decay 1.4e-06 \
    --n_hid 32 \
    --batch_size 512 \
    --horizon 5 \
    --adj_method pearson \
    --adj_threshold 0.367 \
    --dropout 0.13 \
    --grad_clip 1.0
```

**Use when:** You want GELU activation (expected +2-5% improvement over baseline ReLU).

### GNNHAR1L + RELU (Baseline Comparison)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation relu \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --n_hid 16 \
    --adj_threshold 0.3 \
    --dropout 0.1 \
    --grad_clip 1.0 \
    --horizon 5
```

**Use when:** You want to compare ReLU vs GELU activation functions.

### GNNHAR2L + GELU (2-Layer Architecture)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR2L \
    --activation gelu \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.002 \
    --weight_decay 1e-5 \
    --n_hid 32 \
    --adj_threshold 0.35 \
    --dropout 0.15 \
    --grad_clip 1.0 \
    --horizon 5
```

**Use when:** You want to test if 2-hop graph aggregation improves performance.

### GNNHAR3L + GELU (3-Layer Architecture)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR3L \
    --activation gelu \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.002 \
    --weight_decay 1e-5 \
    --n_hid 32 \
    --adj_threshold 0.35 \
    --dropout 0.2 \
    --grad_clip 1.0 \
    --horizon 5
```

**Use when:** You want to test if 3-hop aggregation provides additional benefits (watch for over-smoothing).

### GHAR (Linear Graph Baseline)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GHAR \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --n_hid 16 \
    --adj_threshold 0.3 \
    --dropout 0.1 \
    --grad_clip 1.0 \
    --horizon 5
```

**Use when:** You want to test if graph information helps without nonlinearity.

### HAR (Linear Baseline)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model HAR \
    --n_seeds 20 \
    --epochs 200 \
    --horizon 5
```

**Use when:** You need a simple baseline for comparison (no graph, no guardrails needed).

---

## 🧪 Optuna Hyperparameter Optimization (8-10 hours)

### GNNHAR1L + GELU (100 Trials, Full Study)
```bash
python gnn/gnnhar_paper/optuna_gnnhar_optimization.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_trials 100 \
    --epochs 200 \
    --batch_size 512 \
    --device cpu \
    --horizon 5 \
    --train_end 2025-12-31 \
    --test_start 2026-01-01
```

**Use when:** You want to find optimal hyperparameters with the corrected loss function.

### GNNHAR1L + RELU (100 Trials)
```bash
python gnn/gnnhar_paper/optuna_gnnhar_optimization.py \
    --model GNNHAR1L \
    --activation relu \
    --n_trials 100 \
    --epochs 200 \
    --batch_size 512 \
    --device cpu \
    --horizon 5 \
    --train_end 2025-12-31 \
    --test_start 2026-01-01
```

**Use when:** You want to compare optimal hyperparameters for ReLU vs GELU.

### Quick Optuna Test (20 Trials, ~30 Minutes)
```bash
python gnn/gnnhar_paper/optuna_gnnhar_optimization.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_trials 20 \
    --epochs 100 \
    --batch_size 512 \
    --device cpu \
    --horizon 5
```

**Use when:** You want to test the Optuna pipeline or get preliminary results.

---

## 📊 Multi-Horizon Training

### Train All Horizons
```bash
# h=1 (short-term)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 \
    --horizon 1 --grad_clip 1.0

# h=5 (medium-term)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 \
    --horizon 5 --grad_clip 1.0

# h=10
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 \
    --horizon 10 --grad_clip 1.0

# h=20 (long-term)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 \
    --horizon 20 --grad_clip 1.0
```

---

## 🧬 Test Guardrails Behavior

### Test Ratio Clipping Effect
```bash
# With ratio clipping (default, recommended)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 3 --epochs 100 \
    --grad_clip 1.0
```

### Test Gradient Clipping
```bash
# With gradient clipping (recommended)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 3 --epochs 100 \
    --grad_clip 1.0

# Without gradient clipping (for comparison/testing)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 3 --epochs 100 \
    --grad_clip 0
```

---

## 📝 Training Output Explanation

### What You'll See During Training

**Normal training with guardrails:**
```
======================================================
  Training Ensemble: GNNHAR1L
======================================================
  Models: 20, n_hid: 32, lr: 0.002400, weight_decay: 0.000001
  Activation: GELU
  Dropout: 0.130
======================================================
  [INFO] Version: v1.3_LOSS_FIX
  [INFO] Using CORRECTED gnnhar_ratio_loss (y_true/y_pred)
  [INFO] Loss function: GNNHAR Ratio Loss (NOT standard QLIKE)
  [INFO] Guardrails: ratio clipping=YES, gradient clipping=max_norm=1.0
  [INFO] Monitoring: ratio stats every 10 epochs

  Epoch 10: Train Loss=1.023456, Val Loss=1.045678
           Ratio: mean=0.9876, std=0.1234, range=[0.8234, 1.1234]

  Epoch 20: Train Loss=1.018234, Val Loss=1.034567
           Ratio: mean=0.9654, std=0.0987, range=[0.7891, 1.0678]

  ...

  SUMMARY
======================================================
  Best Val R²: 0.7856
  Test R²: 0.7623
  Test MAE: 0.0045
  Test RMSE: 0.0067
  N models: 18 (2 screened)
======================================================
```

### Warning Example (If Issues Detected)
```
  Epoch 50: Train Loss=1.234567, Val Loss=1.456789
           Ratio: mean=1.2345, std=2.3456, range=[0.7891, 145.2345]
           [WARN] Extreme ratio detected (max=145.23)
                  Model may be predicting near-zero volatility
```

---

## 🎛️ Parameter Reference

### Complete Parameter List

| Parameter | Description | Default | Recommended Range | Example |
|------------|-------------|---------|-------------------|---------|
| `--model` | Model architecture | GNNHAR1L | HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L | `--model GNNHAR2L` |
| `--activation` | Activation function | relu | relu, gelu | `--activation gelu` |
| `--n_seeds` | Ensemble models | 20 | 10-20 (more = better but slower) | `--n_seeds 15` |
| `--epochs` | Max epochs | 400 | 200-500 | `--epochs 300` |
| `--lr` | Learning rate | 0.001 | 0.001-0.003 | `--lr 0.002` |
| `--weight_decay` | L2 regularization | 1e-5 | 1e-6 to 1e-4 | `--weight_decay 5e-6` |
| `--n_hid` | Hidden dimension | 16 | 16, 32, 64 | `--n_hid 32` |
| `--adj_threshold` | Graph density | 0.3 | 0.2-0.5 | `--adj_threshold 0.35` |
| `--dropout` | Dropout rate | 0.0 | 0.0-0.3 | `--dropout 0.15` |
| `--grad_clip` | Gradient clipping max norm | 1.0 | 0.5-2.0 (0 to disable) | `--grad_clip 1.5` |
| `--horizon` | Forecast horizon | 5 | 1, 5, 10, 20 | `--horizon 1` |

### Architectural Guardrail Parameters

| Parameter | Description | Default | When to Change |
|------------|-------------|---------|---------------|
| `--grad_clip` | Gradient clipping max norm | 1.0 | Set to 0 to disable, increase if gradients exploding |

**Internal guardrail parameters (in gnnhar_ratio_loss):**
- `clip_ratio`: Whether to clip ratio (default: True)
- `clip_min`: Minimum ratio value (default: 1e-4)
- `clip_max`: Maximum ratio value (default: 1e4)

---

## 📁 Results Location

### Output Directory
```
results/gnnhar_paper/multi_stock/
```

### File Naming Convention
**Format:** `{model}_{activation}_h{horizon}_{timestamp}.json`

**Examples:**
- `GNNHAR1L_gelu_h5_20260602_143022.json`
- `GNNHAR2L_relu_h1_20260602_151045.json`
- `GHAR_gelu_h20_20260602_161233.json`

### Results JSON Structure
```json
{
  "model": "GNNHAR1L",
  "activation": "gelu",
  "version": "v1.3_LOSS_FIX",
  "dropout": 0.13,
  "grad_clip": 1.0,
  "horizon": 5,
  "adj_method": "pearson",
  "adj_threshold": 0.367,
  "n_seeds": 20,
  "n_hid": 32,
  "test_r2": 0.7623,
  "test_mae": 0.0045,
  "test_rmse": 0.0067,
  "n_models": 18,
  "model_val_losses": [1.023, 1.045, ...],
  "model_epochs": [380, 400, 385, ...]
}
```

---

## ⏱️ Expected Training Times

### Time Estimates by Configuration

| Configuration | Seeds | Epochs | Approx. Time | Purpose |
|----------------|-------|--------|-------------|---------|
| Quick test | 1-3 | 50 | 5-10 min | Verify setup works |
| Quick test | 1-3 | 100 | 10-15 min | Quick validation |
| Single model | 20 | 400 | 2-3 hours | Production model |
| Optuna (20 trials) | - | 100 | 30 min | Quick optimization |
| Optuna (100 trials) | - | 200 | 8 hours | Full optimization |

### Hardware Assumptions
- **CPU:** Single core, modern processor
- **RAM:** 8GB+ sufficient
- **GPU:** Not required (runs on CPU)
- **Storage:** <1GB for all results

---

## 🚦 Recommended Workflow

### Step 1: Quick Verification (5-10 min)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 3 --epochs 50 \
    --grad_clip 1.0
```
**Why:** Verify that corrected loss and guardrails work before long training.

### Step 2: Choose Your Approach

**Option A: Use Optuna's Best Parameters** (Recommended)
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 \
    --lr 0.0024 --weight_decay 1.4e-06 --n_hid 32 \
    --adj_threshold 0.367 --dropout 0.13 --grad_clip 1.0
```
**Why:** Fastest path to good results with corrected loss.

**Option B: Run Fresh Optuna Study**
```bash
python gnn/gnnhar_paper/optuna_gnnhar_optimization.py \
    --model GNNHAR1L --activation gelu --n_trials 100 \
    --epochs 200 --horizon 5
```
**Why:** Find optimal parameters specifically for corrected loss.

**Option C: Quick Comparison Study**
```bash
# Train GNNHAR1L-RELU
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation relu --n_seeds 10 --epochs 200

# Train GNNHAR1L-GELU
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 10 --epochs 200
```
**Why:** Compare activation functions quickly.

---

## 🔍 Troubleshooting

### If Training Fails

**Problem:** `ModuleNotFoundError` or `ImportError`
```bash
# Solution: Check you're in the right directory
cd D:\bmad-projects\luanvan_exp\moirai

# Verify path
pwd
```

**Problem:** Training very slow
```bash
# Solution: Reduce n_seeds or epochs
--n_seeds 10 --epochs 200

# Or reduce model complexity
--n_hid 16
```

**Problem:** "Extreme ratio detected" warnings
```bash
# Solution 1: Check if learning rate is too high
--lr 0.001  # Reduce from 0.002

# Solution 2: Increase dropout
--dropout 0.2  # Increase regularization

# Solution 3: Monitor if it persists (may be transient)
```

**Problem:** Loss = NaN or Inf
```bash
# Solution: Check data quality and predictions
# This should not happen with guardrails enabled
# If it does, run: python gnn/gnnhar_paper/tests/test_architectural_guardrails.py
```

---

## 📊 Comparing Results

### Compare Old vs New (Invalid vs Valid)

**Invalid (v1.0-v1.2):** Located in `results/invalid_loss_bug/`  
**Valid (v1.3+):** Located in `results/v1.3_LOSS_FIX/`

**DO NOT:** Use results from `invalid_loss_bug/` for any conclusions  
**DO:** Only use results from v1.3_LOSS_FIX or later

---

## 🎯 Success Criteria

### Training is Successful When:

1. ✅ **No NaN or Inf losses** - All epochs show finite loss values
2. ✅ **Ratio stats healthy** - Mean ~1.0, max < 100 (no warnings)
3. ✅ **Convergence achieved** - Val loss plateaus or early stopping triggers
4. ✅ **Test R² reasonable** - > 0.7 for this problem domain
5. ✅ **No gradient explosions** - Gradient clipping working
6. ✅ **Results file created** - JSON file with all metrics

### Expected Performance Ranges:

- **Val R²:** 0.75 - 0.82 (good)
- **Test R²:** 0.72 - 0.80 (good)
- **MAE:** 0.003 - 0.008 (lower is better)
- **RMSE:** 0.005 - 0.010 (lower is better)

---

## 📚 Related Documentation

### Technical Details
- **Bug fix details:** `docs/bug_fix_v1.3_loss_ratio.md`
- **Loss function explanation:** `docs/learning/06_gnnhar_ratio_loss.md`
- **Stakeholder communication:** `docs/advisor_communication_draft.md`
- **Recovery plan:** `docs/recovery_priority_tracker.md`

### Test Suites
- **Loss function tests:** `gnn/gnnhar_paper/tests/test_quasi_likelihood_loss.py`
- **Guardrail tests:** `gnn/gnnhar_paper/tests/test_architectural_guardrails.py`
- **Function rename test:** `gnn/gnnhar_paper/tests/test_function_rename.py`

### Configuration
- **Project constraints:** `CONSTRAINTS.md`
- **Project context:** `project-context.md`

---

## 🎓 Quick Reference Card

### Most Common Commands

```bash
# Quick test (5 min)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 1 --epochs 50 --grad_clip 1.0

# Production training (3 hours)
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 \
    --lr 0.0024 --weight_decay 1.4e-06 --n_hid 32 \
    --adj_threshold 0.367 --dropout 0.13 --grad_clip 1.0

# Optuna optimization (8 hours)
python gnn/gnnhar_paper/optuna_gnnhar_optimization.py \
    --model GNNHAR1L --activation gelu --n_trials 100 --epochs 200 --horizon 5
```

---

## ✅ Pre-Training Checklist

Before running production training, verify:

- [ ] Bug fix applied: Version v1.3_LOSS_FIX or later
- [ ] All tests pass: `python gnn/gnnhar_paper/tests/test_architectural_guardrails.py`
- [ ] Guardrails enabled: `--grad_clip 1.0` (or keep default)
- [ ] Results directory ready: `results/v1.3_LOSS_FIX/` exists
- [ ] Invalid results archived: `results/invalid_loss_bug/` separated
- [ ] Sufficient time available: 2-4 hours per model
- [ ] Clear understanding of which horizon you're training

---

**Remember:** All commands now use the **corrected v1.3_LOSS_FIX** implementation with production-grade architectural guardrails. This is the version that matches the GNNHAR paper exactly, with enhanced stability and monitoring.

---

*Last Updated: 2026-06-02*  
*Version: 1.0*  
*Status: Production Ready*
