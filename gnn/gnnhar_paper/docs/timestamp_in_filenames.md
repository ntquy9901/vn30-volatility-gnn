# Timestamp in Filenames Implementation

**Date:** 2026-06-01
**Version:** v1.1_GELU (updated)
**Status:** Complete

## Overview

Added timestamp to all output filenames (learning curves PNG images and JSON result files) to prevent file overwrites and enable proper tracking of multiple training runs.

## Changes Made

### 1. Updated Function Signatures

**`plot_learning_curves()`**
```python
# Before
def plot_learning_curves(..., save_path: Path)

# After
def plot_learning_curves(..., save_path: Path, timestamp: str = None)
```

**`plot_ensemble_learning_curves()`**
```python
# Before
def plot_ensemble_learning_curves(..., save_path: Path)

# After
def plot_ensemble_learning_curves(..., save_path: Path, timestamp: str = None)
```

**`train_ensemble()`**
```python
# Before
def train_ensemble(..., activation: str = 'relu') -> dict

# After
def train_ensemble(..., activation: str = 'relu', timestamp: str = None) -> dict
```

### 2. Updated Filename Generation

**Individual seed learning curves:**
```python
# Before
GNNHAR1L_seed42_learning_curve.png

# After (with timestamp)
GNNHAR1L_seed42_learning_curve_20260601_213000.png
```

**Ensemble learning curves:**
```python
# Before
GNNHAR1L_ensemble_learning_curve.png

# After (with timestamp)
GNNHAR1L_ensemble_learning_curve_20260601_213000.png
```

**JSON result files:**
```python
# Before (already had timestamp)
GNNHAR1L_gelu_h5_20260601_213000.json

# After (same timestamp used for all files)
GNNHAR1L_gelu_h5_20260601_213000.json
```

### 3. Timestamp Generation

**In `main()`:**
```python
# Generate timestamp once for the entire run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Pass to train_ensemble
result = train_ensemble(..., timestamp=timestamp)

# Use same timestamp for JSON file
result_file = results_dir / f'{args.model}_{args.activation}_h{args.horizon}_{timestamp}.json'
```

**In `train_ensemble()`:**
```python
# Generate timestamp if not provided (for backward compatibility)
if timestamp is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Pass to plotting functions
plot_learning_curves(..., timestamp=timestamp)
plot_ensemble_learning_curves(..., timestamp=timestamp)
```

## Benefits

1. **No file overwrites** - Multiple training runs create separate files
2. **Consistent timestamps** - All files from same run share identical timestamp
3. **Better organization** - Easy to identify when results were generated
4. **Tracking experiments** - Timestamp links learning curves to JSON results

## File Naming Convention

**Format:** `{model_name}_{description}_{timestamp}.{ext}`

**Examples:**
- `GNNHAR1L_seed42_learning_curve_20260601_213000.png`
- `GNNHAR1L_ensemble_learning_curve_20260601_213000.png`
- `GNNHAR1L_gelu_h5_20260601_213000.json`

**Timestamp format:** `YYYYMMDD_HHMMSS` (e.g., `20260601_213000` = June 1, 2026, 21:30:00)

## Backward Compatibility

**Fully backward compatible:**
- `timestamp` parameter defaults to `None` in all functions
- If `timestamp=None`, functions behave as before (no timestamp in filename)
- Existing code that doesn't pass timestamp continues to work

**Migration:**
```python
# Old code (still works)
plot_learning_curves(..., save_path=results_dir)

# New code (recommended)
plot_learning_curves(..., save_path=results_dir, timestamp=timestamp)
```

## Testing

**Tested:**
- ✅ Function signatures include timestamp parameter
- ✅ Filename generation with timestamp works correctly
- ✅ Filename generation without timestamp (backward compatibility)
- ✅ All files from same run use identical timestamp

## Example Output

**After training GNNHAR1L with 2 seeds:**
```
results/gnnhar_paper/multi_stock/
├── GNNHAR1L_seed42_learning_curve_20260601_213000.png
├── GNNHAR1L_seed123_learning_curve_20260601_213000.png
├── GNNHAR1L_ensemble_learning_curve_20260601_213000.png
└── GNNHAR1L_gelu_h5_20260601_213000.json
```

All files share timestamp `20260601_213000`, indicating they were generated in the same training run.

## Next Steps

**Recommended:**
1. Test training with multiple runs to verify timestamps work correctly
2. Clean up old files without timestamps (if any exist)
3. Update documentation to reflect new naming convention

---

**Implementation Status:** Complete ✅
**Files Modified:** `train_multi_stock.py`
**Backward Compatible:** Yes
