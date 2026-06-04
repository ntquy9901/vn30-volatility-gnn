# Optuna Hyperparameter Optimization Implementation

**Date:** 2026-06-01
**Version:** v1.1_GELU
**Status:** Ready to run

## Overview

Implemented Optuna hyperparameter optimization for GNNHAR1L volatility forecasting. Optuna uses Bayesian optimization (TPE sampler) to efficiently search hyperparameter space and find optimal configuration.

## What is Optuna?

**Optuna:** Automatic hyperparameter optimization framework
- **Algorithm:** Tree-structured Parzen Estimator (TPE) - Bayesian optimization
- **Efficiency:** Prunes unpromising trials early (MedianPruner)
- **Reproducibility:** SQLite storage for study persistence
- **Visualization:** Optuna Dashboard for real-time monitoring

## Expected Impact

**From technical research:**
- **Expected improvement:** +5-10% R² over baseline
- **Risk level:** LOW (Optuna is mature, extensively tested)
- **Duration:** ~8 hours (100 trials × ~5 min/trial)

## Hyperparameter Search Space

```python
{
    'lr': [1e-4, 1e-2],              # Learning rate (log scale)
    'weight_decay': [1e-6, 1e-4],    # L2 regularization (log scale)
    'n_hid': [16, 32, 64],           # Hidden dimension
    'adj_threshold': [0.2, 0.5],     # Graph density (correlation threshold)
    'dropout': [0.0, 0.3],            # Dropout rate (if implemented)
}
```

## Implementation Details

### 1. Main Script: `optuna_gnnhar_optimization.py`

**Key components:**

**Objective function:**
```python
def objective(trial, ...):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    n_hid = trial.suggest_categorical('n_hid', [16, 32, 64])
    adj_threshold = trial.suggest_float('adj_threshold', 0.2, 0.5)

    # Train model with these hyperparameters
    result = train_single_model(...)

    # Return validation R² (Optuna maximizes this)
    return result['val_r2']
```

**TPE Sampler (Tree-structured Parzen Estimator):**
- Models hyperparameter distribution from previous trials
- Suggests promising hyperparameters based on history
- More efficient than grid search or random search

**Median Pruner:**
- Stops unpromising trials early
- Saves time by aborting poorly performing configurations
- Allows focusing resources on promising regions

**Study Persistence:**
```python
study = optuna.create_study(
    storage='sqlite:///optuna_studies.db',  # SQLite database
    load_if_exists=True,  # Resume if study exists
)
```

### 2. Updated `train_single_model()`

**Added validation R² calculation:**
```python
# Calculate R² at best epoch
val_r2 = 1 - ss_res / ss_tot

return {
    'best_val_loss': best_val_loss,
    'val_r2': val_r2,  # NEW: for Optuna optimization
    'n_epochs': epoch + 1,
    ...
}
```

## Usage

### Quick Start (Windows)

```batch
cd D:\bmad-projects\luanvan_exp\moirai
gnn\gnnhar_paper\run_optuna_gnnhar1l.bat
```

### Quick Start (Linux/Mac)

```bash
cd D:\bmad-projects/luanvan_exp/moirai
bash gnn/gnnhar_paper/run_optuna_gnnhar1l.sh
```

### Manual Execution

**Step 1: Install Optuna**
```bash
pip install optuna optuna-dashboard
```

**Step 2: Run optimization**
```bash
python gnn/gnnhar_paper/optuna_gnnhar_optimization.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_trials 100 \
    --epochs 200 \
    --horizon 5
```

### Monitoring Progress (Optional)

**Launch Optuna Dashboard:**
```bash
optuna-dashboard sqlite:///optuna_studies.db
```

**Open browser:** http://localhost:8080

**Dashboard shows:**
- Real-time trial progress
- Best hyperparameters so far
- Importance analysis (which params matter most)
- Hyperparameter relationships

## Expected Results

### Output Files

**JSON results:**
```
results/gnnhar_paper/optuna/GNNHAR1L_gelu_optuna_20260601_220000.json
```

**Contents:**
```json
{
  "model": "GNNHAR1L",
  "activation": "gelu",
  "study_name": "GNNHAR1L_gelu_h5_20260601_220000",
  "n_trials": 100,
  "best_trial_number": 42,
  "best_val_r2": 0.7684,
  "best_params": {
    "lr": 0.0032,
    "weight_decay": 0.000015,
    "n_hid": 32,
    "adj_threshold": 0.28,
    "dropout": 0.12
  }
}
```

### Expected Performance

**Baseline (before Optuna):**
- R² = 0.722 (GELU, 2 seeds, 100 epochs)

**After Optuna (expected):**
- R² ≥ 0.77 (+6.7% improvement)
- Best hyperparameters identified
- Reduced variance across seeds

## Next Steps After Optimization

### Step 1: Train Final Model

**Use best hyperparameters from JSON:**
```bash
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_seeds 20 \
    --epochs 400 \
    --lr 0.0032 \
    --weight_decay 0.000015 \
    --n_hid 32 \
    --adj_threshold 0.28
```

**Expected duration:** ~3.3 hours
**Expected R²:** ≥ 0.77

### Step 2: Compare with Baselines

**Expected hierarchy:**
1. GNNHAR1L (Optuna): R² ≥ 0.77 ⭐
2. sklearn GHAR: R² = 0.754
3. GNNHAR1L (baseline): R² = 0.722
4. HAR OLS: R² = 0.753

### Step 3: Analyze Hyperparameter Importance

**Optuna provides importance analysis:**
```python
study = optuna.load_study(study_name, storage)
importance = optuna.importance.get_param_importances(study)

# Expected importance ranking:
# 1. adj_threshold (graph density) - MOST IMPORTANT
# 2. lr (learning rate)
# 3. n_hid (hidden dimension)
# 4. weight_decay (regularization)
# 5. dropout (if used)
```

## Technical Details

### Why TPE Sampler?

**TPE (Tree-structured Parzen Estimator):**
- Models P(best_hyperparams | past_trials)
- More efficient than grid/random search
- Handles continuous, categorical, conditional parameters
- Standard choice for hyperparameter optimization in 2026

### Why Median Pruner?

**Median Pruner:**
- Compares trial to median of previous trials
- Prunes if trial performs worse than median
- Conservative pruning (doesn't prune too aggressively)
- Saves ~30-50% time on unpromising configurations

### Why SQLite Storage?

**Benefits:**
- Persistent across script restarts
- Can resume interrupted studies
- Easy to analyze with Optuna Dashboard
- Database file: `optuna_studies.db`

## Optimization Strategy

### Search Space Design

**Learning rate (log scale):**
- Range: [1e-4, 1e-2]
- Log scale: Because LR effects are multiplicative
- Current default: 1e-3 (middle of range)

**Weight decay (log scale):**
- Range: [1e-6, 1e-4]
- Log scale: Because regularization effects are multiplicative
- Current default: 1e-5 (middle of range)

**Hidden dimension (categorical):**
- Options: [16, 32, 64]
- Categorical: Model architecture parameter
- Current default: 16 (smallest option)

**Adjacency threshold (continuous):**
- Range: [0.2, 0.5]
- Continuous: Graph density parameter
- Current default: 0.3 (middle of range)
- 0.2 = denser graph (more connections)
- 0.5 = sparser graph (fewer connections)

**Dropout (continuous):**
- Range: [0.0, 0.3]
- Conservative: Don't exceed 0.3 for small models
- Current: Not implemented yet

### Trial Configuration

**100 trials with median pruning:**
- 10 startup trials (no pruning)
- 30 warmup steps per trial (allow convergence)
- Expected pruning: ~40% of trials aborted early
- Expected time savings: ~30-40%

**Max epochs: 200 (vs 400 for full training)**
- Faster convergence for hyperparameter search
- Sufficient to identify good configurations
- Final model uses full 400 epochs

## Troubleshooting

### Issue: Optuna not installed

**Solution:**
```bash
pip install optuna optuna-dashboard
```

### Issue: Study takes too long

**Solution:**
```bash
# Reduce trials for quick test
--n_trials 20  # Instead of 100

# Or reduce max epochs
--epochs 100  # Instead of 200
```

### Issue: Best trial has poor R²

**Possible causes:**
1. Training data insufficient (need full ensemble first)
2. Search space too narrow (expand ranges)
3. Need more trials (increase n_trials)

**Solution:**
- Complete full ensemble training first (Priority 1)
- Then re-run Optuna with trained baselines

### Issue: Database locked

**Solution:**
```bash
# Delete old database and start fresh
rm optuna_studies.db
# Or use different storage URL
--storage sqlite:///optuna_studies_v2.db
```

## Files Created

1. **`optuna_gnnhar_optimization.py`** - Main optimization script
2. **`run_optuna_gnnhar1l.bat`** - Windows batch script
3. **`run_optuna_gnnhar1l.sh`** - Linux/Mac shell script
4. **`train_multi_stock.py`** - Updated to return validation R²

## Files Modified

1. **`train_multi_stock.py`** - Added `val_r2` to `train_single_model()` return value

## Success Criteria

**Optimization successful if:**
- ✅ Best trial R² ≥ 0.75 (match sklearn baselines)
- ✅ Improvement over baseline ≥ +5% R²
- ✅ Identify meaningful hyperparameter patterns

**Final model successful if:**
- ✅ R² ≥ 0.77 with Optuna hyperparameters
- ✅ Low variance across 20 seeds (std < 0.02 R²)
- ✅ Beats sklearn GHAR (R² > 0.754)

## Research Contribution

**Academic value:**
- First systematic hyperparameter optimization for GNNHAR1L
- Identifies critical hyperparameters (graph density expected #1)
- Provides reproducible optimization methodology
- Enables fair comparison with sklearn baselines

## References

1. Optuna Documentation: https://optuna.readthedocs.io/
2. TPE Paper: "Algorithms for Hyper-Parameter Optimization" (Bergstra et al.)
3. Technical Research: `technical-gnnhar1l-improvements-volatility-forecasting-research-2026-06-01.md`

---

**Implementation Status:** Complete ✅
**Testing Status:** Ready to run
**Expected Duration:** ~8 hours
**Expected Outcome:** R² ≥ 0.77 (+5-10% improvement)
