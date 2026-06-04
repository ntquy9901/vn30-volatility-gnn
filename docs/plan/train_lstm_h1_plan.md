# Plan: Train LSTM for Stocks with ESS_h > 3500 (h=1)

## Context

You want to train per-stock LSTM models targeting 1-day-ahead volatility (h=1) for stocks where the effective sample size (ESS) is **more than 3500** raw trading days.

**Formula:** ESS_h=1 = N_raw / 1 = N_raw (since h=1, no overlapping window penalty)

This filters for data-rich stocks: identify which VN30 stocks have sufficient training history and test LSTM performance with abundant data using h=1 (most aggressive RV target, shortest horizon).

## Key Requirements (from CONSTRAINTS.md + New R6)

Must follow ALL mandatory rules:
- **R1 (MIMO):** Single model with output heads, configurable horizons. Here: HORIZONS=[1]
- **R2 (Monitoring):** Print per-horizon loss each epoch + save training curve PNG
- **R3 (Data Split):** Print train/val/test dates + sample counts + ESS before training
- **R4 (LSTM stride=1):** Per-stock, stride=1, in verbose
- **R5:** CONSTRAINTS.md is source of truth
- **R6 (Global):** Test from 2026-01-01, Train/Val 80/20 from pre-2026. Console + learning curves for ALL models.

## Global Data Split Rules (R6)

**Applied to ALL models (LSTM, GNN, MLP, HAR, etc.):**

```
GLOBAL TEST SET: 2026-01-01 onwards (fixed boundary)

TRAIN/VAL SPLIT (from data before 2026-01-01):
  - Ratio: 80% Train, 20% Val
  - Example: If 2,400 samples before 2026-01-01
    → Train: first 1,920 samples
    → Val: last 480 samples
```

**Console Output:**
```
Epoch 001 | Train: loss=0.0234 [H1=0.0234] | Val: loss=0.0189 [H1=0.0189]
Epoch 002 | Train: loss=0.0198 [H1=0.0198] | Val: loss=0.0156 [H1=0.0156]
...
```

**Learning Curves:** Embedded in training code, saved as PNG after each model finishes training.

## Implementation Plan

### Phase 1: Data Filtering & Analysis

**File:** `scripts/eda/filter_stocks_by_ess.py`

Steps:
1. Load all VN30 stock price data from `data/raw/prices/{ticker}_ohlcv.csv`
2. Calculate N_raw for each stock = len(data from config.data_start to 2025-12-31)
3. Calculate ESS_h=1 for each stock = N_raw / 1 = N_raw
4. Filter stocks where ESS_h=1 > 3500
5. Print results table with columns: ticker, n_raw, ess_h1, qualifies (Y/N)
6. Save filtered stock list to `results/stocks_ess_h1_over_3500.csv`

Output:
- Console: ESS summary table
- CSV: `results/stocks_ess_h1_over_3500.csv` (ticker, n_raw, ess_h1)
- **Expected result:** Likely ZERO qualifying stocks (all VN30 have ~2900 trading days < 3500). This is a diagnostic verification step.

### Phase 2: Train Per-Stock LSTM (h=1 only)

**File:** `baselines/train_lstm_h1.py` (new CONSTRAINTS.md-compliant version)

Architecture & Config:
```python
LOOKBACK = 20
HORIZONS = [1]  # Single horizon (h=1 only)
MAX_H = 1
STRIDE = 1

# Global data split (R6) — ALL data before 2026-01-01 eligible for train/val
GLOBAL_TEST_START = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8  # 80/20

# Hyperparameters
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 32
PATIENCE = 20
SEED = 42
OPTIMIZER = "Adam"
LOSS_FN = "MSELoss"

# For each qualifying stock:
# 1. Load all data up to 2025-12-31
# 2. Split into train_samples (80%) and val_samples (20%)
# 3. Test: from 2026-01-01 onwards

LSTMModel:
  - Input: (batch, 20, 1)  # 20 days of past RV
  - LSTM: hidden=32, dropout=0.1
  - Output head: Linear(32 → 1)  # Single RV prediction (h=1)
```

Training Loop (per stock):
1. Load stock prices → compute log-returns → compute RV(h=1)
2. Split data:
   - **Before 2026-01-01:** All historical data
     - **Train (80%):** First 80% of historical samples
     - **Val (20%):** Last 20% of historical samples
   - **From 2026-01-01 onwards:** Test set
3. Print data split with ESS: dates, sample counts, ESS per split
   ```
   Stock VCB:
     Train: 2014-07-01 -> 2025-11-30 (1,920 samples, ESS=1920)
     Val:   2025-12-01 -> 2025-12-31 (480 samples, ESS=480)
     Test:  2026-01-01 onwards (TBD samples)
   ```
4. Create train/val/test sequences with stride=1 and LOOKBACK=20
5. Initialize LSTM model with hidden=32
6. Train for 100 epochs with validation monitoring:
   - **Print each epoch to console** (rule R6):
     ```
     Epoch 001 | Train: 0.0234 [H1=0.0234] | Val: 0.0189 [H1=0.0189] | LR: 1.00e-03
     Epoch 002 | Train: 0.0198 [H1=0.0198] | Val: 0.0156 [H1=0.0156] | LR: 1.00e-03
     ```
   - Track train_loss and val_loss per epoch
   - Early stopping (patience=20) based on val_loss
7. **Plot & save learning curve** (rule R2 + R6):
   - Learning curve plotting code embedded in training script
   - Save to `results/lstm_h1_curves/{ticker}_curve.png`
   - Include: train loss, val loss, per-horizon breakdown
8. Evaluate on test set (2026-01-01 onwards) → compute MAE, RMSE, R², Directional Accuracy
9. Save trained model to `models/lstm_h1/{ticker}_model.pt`

### Phase 3: Evaluation & Results Summary

**File:** `scripts/eda/evaluate_lstm_h1.py`

For each trained stock:
1. Load test data (2026-01-01 to latest available data)
2. Generate predictions
3. Compute metrics: MAE, RMSE, R², Directional Accuracy
4. Compare to simple baseline: RV_t = mean(train_rv)
5. Aggregate results table

Output:
- Console: Results table (ticker, MAE, RMSE, R², DA, ESS_train)
- CSV: `results/lstm_h1_results.csv`
- Plot: `results/lstm_h1_summary_metrics.png` (MAE/R² boxplot across stocks)

## Critical Files

| Task | File Path |
|------|-----------|
| Config | `config.yaml` (read for dates) |
| Data Loading | `src/volatility_labels.py:load_close_prices(), compute_rv()` |
| Stock List | `gnn/build_graph.py:VN30_TICKERS` |
| Filtering Script | `scripts/eda/filter_stocks_by_ess.py` (NEW) |
| LSTM Training | `baselines/train_lstm_h1.py` (NEW, CONSTRAINTS.md-compliant) |
| Evaluation | `scripts/eda/evaluate_lstm_h1.py` (NEW) |
| Output | `results/lstm_h1_*` (NEW folder) |

## Verification

1. **Filter Phase:**
   - Run `python scripts/eda/filter_stocks_by_ess.py`
   - Verify ESS table printed to console
   - Check `results/stocks_ess_h1_over_3500.csv` created (may be empty if no stocks qualify)

2. **Training Phase:**
   - Run `python baselines/train_lstm_h1.py --stocks-from results/stocks_ess_h1_over_3500.csv`
   - Verify per-stock training output (data split + epoch logs printed to console)
   - Check training curves saved to `results/lstm_h1_curves/`
   - Check models saved to `models/lstm_h1/`

3. **Evaluation Phase:**
   - Run `python scripts/eda/evaluate_lstm_h1.py`
   - Verify results CSV + metrics plot created
   - Compare results to v1/v2 GNN results (for reference)

## Notes

- All scripts follow `scripts/eda/` convention (feedback_scripts.md)
- Each script produces charts/tables for reusability
- MIMO architecture used (even with single h=1) for consistency with CONSTRAINTS.md R1
- stride=1 maximizes samples; ESS will be printed per CONSTRAINTS.md R4
- **Global data split (R6):** 80/20 train/val from pre-2026 data; test from 2026-01-01 onwards
- **Console output (R6):** All training must print epoch progress to observable console output
- **Learning curves (R6):** Plotting code embedded in training scripts, saved as PNG for all models
- **Data sufficiency:** ESS > 3500 likely results in zero qualifying stocks (all VN30 have ~2900 days). If Phase 1 yields empty dataset, this confirms data limitation and validates the threshold assumption.
