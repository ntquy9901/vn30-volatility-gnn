# Plan: Train LSTM MIMO for All Stocks (h=1,5,10,20)

## Context

Train per-stock LSTM targeting **all 4 horizons simultaneously** (HORIZONS=[1,5,10,20])
using the full MIMO architecture from CONSTRAINTS.md R1.

This is the **primary LSTM experiment** — h=1 was a diagnostic step that confirmed
1-day RV is too noisy for per-stock univariate LSTM (avg R2=-0.033).
h=20 is the main comparison horizon used by GNN, MLP, and Moirai2.

**h=1 results (reference):**
```
Avg R2 = -0.033  |  Avg DA = 0.282  |  LSTM beats baseline MAE: 8/14 stocks
Conclusion: h=1 RV is near-white-noise, expected to fail per CONSTRAINTS.md ESS analysis
```

**Why h=20 should be better:**
- RV_h20 = std of 20 log-returns (ddof=1) — Central Limit Theorem smooths noise
- Autocorrelation of RV_h20 is high (HAR-RV model has R2 ~ 0.4-0.6 on this data)
- ESS = N_train / 20, but signal/noise ratio is ~6x higher than h=1

## User Requirements (non-negotiable)

- **CLI:** Script phải chạy được trực tiếp từ terminal bằng `python baselines/train_lstm_h20.py`
- **Real-time output:** Mỗi epoch phải hiện ngay lập tức trong terminal, không bị buffer.
  Bắt buộc thêm `sys.stdout.reconfigure(line_buffering=True)` ở đầu script.
- **Learning curve PNG:** Sau khi mỗi stock train xong, tự động lưu learning curve ra file PNG.
  Path: `results/lstm_h20_curves/{ticker}_curve.png`
  Không cần chạy script riêng — vẽ và lưu ngay trong training loop.

## Key Requirements (CONSTRAINTS.md R1-R6)

- **R1 (MIMO):** HORIZONS=[1,5,10,20] — 1 model, 4 heads, single forward pass
- **R2 (Monitoring):** Print per-horizon loss every epoch + save PNG per stock
- **R3 (Data Split):** Print train/val/test dates + counts + ESS before training
- **R4 (LSTM stride=1):** stride=1, verbose per stock
- **R5:** CONSTRAINTS.md is source of truth
- **R6 (Global):** Test from 2026-01-01, Train/Val 80/20 from pre-2026

## RV Label Formula

```
RV_t(h) = std(log_ret[t+1 .. t+h], ddof=1)

Implementation: compute_rv(close, h=h) from src/volatility_labels.py
  rv = log_ret.rolling(h, min_periods=h).std(ddof=1).shift(-h)

For h=1: special case — rolling(1).std(ddof=1) = NaN
  => use abs(log_ret).shift(-1)  [same as train_lstm_h1.py]

For h=5, 10, 20: compute_rv() works correctly, no special case needed.
```

## ESS Analysis for h=20

```
ESS_h = N_train / max_h = N_train / 20

Stock examples (from h=1 training):
  VNM:  N_train=3960  -> ESS_h20 = 198   (sufficient)
  ACB:  N_train=3810  -> ESS_h20 = 190   (sufficient)
  MBB:  N_train=2831  -> ESS_h20 = 141   (sufficient)
  All 30 VN30: N_train ~ 1900-4000  -> ESS_h20 = 95-200

Threshold: ESS_h20 >= 100 (Lopez de Prado: minimum for reliable estimation)
Expected: most VN30 stocks qualify (vs only 14/30 for ESS_h1 > 3500)
```

## Global Data Split (R6)

```
GLOBAL_TEST_START = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

For each stock (example VNM with N_pre=4975):
  Train: first 3980 pre-test samples  (ESS = 3980/20 = 199)
  Val:   last  995 pre-test samples   (ESS = 995/20  = 49)
  Test:  from 2026-01-01 onwards      (~86 samples)
```

## Implementation Plan

### Phase 1: ESS Filter for h=20

**File:** `scripts/eda/filter_stocks_by_ess_h20.py`

Steps:
1. Load all VN30 prices from `data/raw/prices/`
2. For each stock: N_raw = len(data before 2026-01-01), ESS_h20 = N_raw / 20
3. Filter: ESS_h20 >= 100
4. Print table: ticker, n_raw, ess_h20, qualifies
5. Save `results/stocks_ess_h20_over100.csv`

Expected: most or all 30 stocks qualify (N_raw > 2000 for almost all VN30).

Console output format:
```
ticker  n_raw  ess_h20  qualifies
   VNM   4976      248        YES
   ACB   4764      238        YES
   ...
```

### Phase 2: Train Per-Stock LSTM MIMO (h=1,5,10,20)

**File:** `baselines/train_lstm_h20.py`

Config:
```python
LOOKBACK   = 20
HORIZONS   = [1, 5, 10, 20]   # R1: full MIMO
MAX_H      = 20
STRIDE     = 1                  # R4

GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

EPOCHS     = 150                # more epochs since h=20 convergence is slower
LR         = 1e-3
BATCH_SIZE = 32
PATIENCE   = 25
SEED       = 42
HIDDEN     = 64                 # larger hidden than h=1 (64 vs 32) for 4-head MIMO
DROPOUT    = 0.2
```

MIMO architecture (R1):
```python
class LSTMModelMIMO(nn.Module):
    # Input:  (batch, 20, 1)   -- 20 days of past RV_h1
    # LSTM:   hidden=64, dropout=0.2
    # Heads:  4 x Linear(64 -> 1)  for h=1,5,10,20
    # Output: (batch, 4)
```

Multi-horizon label construction:
```python
# Compute RV for each horizon separately
rv_h1  = abs(log_ret).shift(-1)          # special case for h=1
rv_h5  = compute_rv(close, h=5)
rv_h10 = compute_rv(close, h=10)
rv_h20 = compute_rv(close, h=20)

# Sequence target at position i:
# y[i] = [rv_h1[i+LOOKBACK], rv_h5[i+LOOKBACK], rv_h10[i+LOOKBACK], rv_h20[i+LOOKBACK]]
# Valid i: 0 <= i <= N - LOOKBACK - MAX_H  (ensures all horizons have labels)
```

Training per stock (R2, R3, R4):
1. Load stock prices -> compute rv_h1, rv_h5, rv_h10, rv_h20
2. Align: find common valid indices (no NaN in any rv_hX)
3. Split: pre-2026 -> 80% train / 20% val; 2026+ -> test (R6)
4. Print data split with ESS_h20 (R3)
5. Build stride=1 sequences (R4): X = past 20 days rv_h1, y = 4-horizon targets
6. Normalize each horizon independently on train stats (mean/std per horizon)
7. Train MIMO LSTM, print per-horizon loss every epoch (R2):
   ```
   Epoch  01/150 | Train: 0.4231 [H1=0.8419 H5=0.3201 H10=0.2103 H20=0.1108] | Val: 0.3912 [...] | LR: 1.00e-03
   ```
8. Early stopping on total val loss (patience=25)
9. Save learning curve PNG: `results/lstm_h20_curves/{ticker}_curve.png` (R2)
10. Evaluate on test set: MAE, RMSE, R2, DA per horizon
11. Save model: `models/lstm_h20/{ticker}_model.pt`

Data split print (R3):
```
Stock VNM:
  Train: 2006-06-28 -> 2022-01-15  (3980 samples, ESS_h20=199)
  Val:   2022-01-17 -> 2025-12-31  ( 995 samples, ESS_h20=49)
  Test:  2026-01-05 onwards        (  86 samples)
  stride=1 | LOOKBACK=20 | HORIZONS=[1,5,10,20] | MAX_H=20
```

### Phase 3: Evaluation vs HAR-RV

**File:** `scripts/eda/evaluate_lstm_h20.py`

For each trained model:
1. Load model + normalization constants
2. Rebuild test sequences (pre-test tail as context)
3. Predict all 4 horizons
4. Compute per-horizon metrics: MAE, RMSE, R2, DA
5. Compare vs:
   - Naïve baseline: mean(train_rv_hX)
   - HAR-RV: fit on train, predict on test (use har_rv_baseline.py functions)
6. Focus metric: **h=20 R2** (primary comparison horizon with GNN/Moirai2)

Output:
- Console: per-stock metrics table (all 4 horizons)
- CSV: `results/lstm_h20_results.csv`
- Plot: `results/lstm_h20_summary_metrics.png` — 4-panel per horizon (h=1,5,10,20)
- Plot: `results/lstm_h20_vs_har.png` — LSTM vs HAR-RV R2 by stock and horizon

## Critical Files

| Task | File |
|---|---|
| Config | `config.yaml` (horizon=20) |
| RV labels | `src/volatility_labels.py:compute_rv()` |
| HAR baseline | `baselines/har_rv_baseline.py:build_har_features(), fit_har()` |
| Stock list | `gnn/build_graph.py:VN30_TICKERS` |
| ESS filter | `scripts/eda/filter_stocks_by_ess_h20.py` (NEW) |
| Training | `baselines/train_lstm_h20.py` (NEW) |
| Evaluation | `scripts/eda/evaluate_lstm_h20.py` (NEW) |
| h=1 reference | `results/lstm_h1_results.csv` (existing) |

## Verification

1. **Phase 1:**
   - `python scripts/eda/filter_stocks_by_ess_h20.py`
   - Expect: most/all 30 stocks qualify (ESS_h20 >= 100)

2. **Phase 2:**
   - `python baselines/train_lstm_h20.py --stocks-from results/stocks_ess_h20_over100.csv`
   - Check: per-epoch output shows 4 horizon losses
   - Check: `results/lstm_h20_curves/` has PNG per stock
   - Check: `models/lstm_h20/` has .pt per stock

3. **Phase 3:**
   - `python scripts/eda/evaluate_lstm_h20.py`
   - Primary metric: h=20 R2 — should be positive and > HAR-RV for some stocks
   - Compare: `results/lstm_h20_results.csv` vs `results/lstm_h1_results.csv`

## Expected Results

```
h=20 LSTM (per-stock):
  Expected R2 ~ 0.1 - 0.4 (much better than h=1)
  HAR-RV R2   ~ 0.3 - 0.5 (strong baseline for h=20)
  LSTM likely still below HAR-RV per-stock (limited ESS)
  But MIMO enables multi-horizon report for thesis

h=20 is the primary metric for:
  - Comparison table vs GNN (pooled 30 stocks)
  - Comparison table vs Moirai2 (foundation model)
  - Thesis conclusion: data efficiency argument
```

## Notes

- No emoji or Unicode icons in any script (CLAUDE.md rule)
- ASCII-only console output (`->`, `--`, `[OK]`, `[WARN]`)
- All scripts in correct paths per feedback_scripts.md convention
- HAR-RV comparison is more meaningful than naive mean for h=20
- MIMO output enables thesis to report all 4 horizons in one table
- Keep `train_lstm_h20.py` independent from `train_lstm_h1.py` (no shared state)
