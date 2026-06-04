# Plan: LSTM with HAR Features Input (train_lstm_har_features.py)

## Context

This experiment is a **controlled ablation** of `train_lstm_h20.py`.

The only change: input representation.

| | train_lstm_h20.py (existing) | train_lstm_har_features.py (this plan) |
|---|---|---|
| Input | `rv_h1[t-19..t]` — 20 days of abs(log_ret), input_size=1 | `[rv_d, rv_w, rv_m][t-19..t]` — 20 days x 3 HAR features, input_size=3 |
| MIMO heads | 4 (h=1,5,10,20) | 4 (h=1,5,10,20) |
| HIDDEN | 64 | 64 |
| Everything else | identical | identical |

**Research question:** Does using HAR-style features as LSTM input close the gap vs HAR-RV?

**Expected result (from literature, Branco et al. 2024):**
- LSTM (abs_ret input): avg h=20 R2 = -1.28 (current result)
- LSTM (HAR features):  avg h=20 R2 = 0.5 - 0.85 (expected)
- HAR-RV:               avg h=20 R2 = 0.90 (current result)

**Thesis argument this enables:**
> "The failure of vanilla LSTM at h=20 is not a model capacity problem.
>  It is an information problem. When LSTM receives the same features as HAR-RV,
>  it achieves comparable performance — confirming that feature engineering
>  dominates over architecture choice at this data scale (ESS=90-198)."

---

## User Requirements (non-negotiable, same as train_lstm_h20.py)

- **CLI:** `python baselines/train_lstm_har_features.py --all`
- **Real-time output:** `sys.stdout.reconfigure(line_buffering=True)` mandatory
- **Learning curve PNG:** saved automatically per stock inside training loop
  Path: `results/lstm_har_curves/{ticker}_curve.png`

---

## Key Requirements (CONSTRAINTS.md R1-R6)

- **R1 (MIMO):** HORIZONS=[1,5,10,20], 1 backbone, 4 heads
- **R2 (Monitoring):** Per-horizon loss every epoch + PNG per stock
- **R3 (Data Split):** Print train/val/test dates + ESS before each stock
- **R4 (LSTM stride=1):** stride=1
- **R6 (Global):** Test from 2026-01-01, Train/Val 80/20

---

## Input Feature Construction (Key Difference)

HAR features at each timestep t (using only data available at t):

```python
rv_d[t] = rv_h1[t]                          # daily: abs(log_ret[t+1]) -- same as current
rv_w[t] = rv_h1[t-4..t].mean()              # weekly: 5-day rolling mean
rv_m[t] = rv_h1[t-19..t].mean()             # monthly: 20-day rolling mean

# X[t] = [rv_d[t], rv_w[t], rv_m[t]]        shape: (3,)
# Sequence: X[t-LOOKBACK..t]                  shape: (LOOKBACK, 3)
```

Implementation using pandas rolling:

```python
def build_har_features(rv_h1: pd.Series) -> pd.DataFrame:
    rv_d = rv_h1                                     # daily
    rv_w = rv_h1.rolling(5,  min_periods=5).mean()   # weekly avg
    rv_m = rv_h1.rolling(20, min_periods=20).mean()  # monthly avg
    return pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})
```

Note: this is the same rolling window logic as `har_rv_baseline.py:build_har_features()`
but computed forward-aligned (not shift(1)), since the LSTM sequence itself provides the lag.

---

## Architecture (minimal diff from train_lstm_h20.py)

```python
N_FEATURES = 3                    # rv_d, rv_w, rv_m  (vs 1 in train_lstm_h20.py)
LOOKBACK    = 20
HORIZONS    = [1, 5, 10, 20]
HIDDEN      = 64
DROPOUT     = 0.2

class LSTMModelMIMO(nn.Module):
    def __init__(self, n_features=N_FEATURES, horizons=HORIZONS, hidden=HIDDEN, dropout=DROPOUT):
        self.lstm  = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x):           # x: (B, LOOKBACK, 3) -> (B, 4)
        out, _ = self.lstm(x)
        feat   = self.drop(out[:, -1, :])
        return torch.cat([h(feat) for h in self.heads], dim=1)
```

Only change: `input_size=3` (vs 1). Everything else identical.

---

## Normalization (per-feature)

```python
feat_mu:  dict[str, float]   # keys: "rv_d", "rv_w", "rv_m"
feat_sig: dict[str, float]   # keys: "rv_d", "rv_w", "rv_m"

# Fit on train, apply to val/test
for feat in ["rv_d", "rv_w", "rv_m"]:
    feat_mu[feat]  = float(train_har[feat].mean())
    feat_sig[feat] = float(train_har[feat].std()) + 1e-8

# Normalize X: (N, LOOKBACK, 3)
def norm_X(har_df):
    out = np.zeros((len(har_df), 3), dtype=np.float32)
    for j, f in enumerate(["rv_d", "rv_w", "rv_m"]):
        out[:, j] = (har_df[f].values - feat_mu[f]) / feat_sig[f]
    return out
```

Target y normalization: same as train_lstm_h20.py (per-horizon rv_mu/rv_sig).

---

## Sequence Construction (R4: stride=1)

```python
def make_sequences(x_arr, y_arr, lookback):
    # x_arr: (N, 3)  -- HAR features, normalized
    # y_arr: (N, 4)  -- multi-horizon targets, normalized
    X, y = [], []
    for i in range(len(x_arr) - lookback):
        X.append(x_arr[i : i + lookback])      # (LOOKBACK, 3)
        y.append(y_arr[i + lookback])           # (4,)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    # X: (M, LOOKBACK, 3)  -- ready for LSTM input_size=3
```

---

## Data Flow

```
close prices
    |
    v
compute_all_rv(close)           -- rv_h1, rv_h5, rv_h10, rv_h20 (same as train_lstm_h20.py)
    |
    +--- rv_h1 ---> build_har_features() ---> [rv_d, rv_w, rv_m]  (INPUT)
    |
    +--- rv_h1, rv_h5, rv_h10, rv_h20 ---> targets y             (OUTPUT)
    |
    v
dropna() on BOTH features and targets aligned
    |
    v
train/val/test split (R6)
    |
    v
normalize features (per feat) + normalize targets (per horizon)
    |
    v
make_sequences(X_har, y_all, LOOKBACK)
    |
    v
LSTMModelMIMO(input_size=3)
```

---

## Config

```python
N_FEATURES = 3                    # rv_d, rv_w, rv_m
LOOKBACK   = 20
HORIZONS   = [1, 5, 10, 20]
MAX_H      = 20

GLOBAL_TEST_START     = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

EPOCHS     = 150
LR         = 1e-3
BATCH_SIZE = 32
PATIENCE   = 25
SEED       = 42
HIDDEN     = 64
DROPOUT    = 0.2

CURVES_DIR = "results/lstm_har_curves/"
MODELS_DIR = "models/lstm_har/"
CSV_OUT    = "results/lstm_har_results.csv"
```

---

## Checkpoint Format

```python
torch.save({
    "state_dict":  model.state_dict(),
    "horizons":    HORIZONS,
    "lookback":    LOOKBACK,
    "hidden":      HIDDEN,
    "n_features":  N_FEATURES,
    "rv_mu":       rv_mu,     # dict: {1: float, 5: float, 10: float, 20: float}
    "rv_sig":      rv_sig,    # dict: {1: float, 5: float, 10: float, 20: float}
    "feat_mu":     feat_mu,   # dict: {"rv_d": float, "rv_w": float, "rv_m": float}
    "feat_sig":    feat_sig,  # dict: {"rv_d": float, "rv_w": float, "rv_m": float}
}, MODELS_DIR / f"{ticker}_model.pt")
```

---

## Phase Structure

### Phase 1: train_lstm_har_features.py

**File:** `baselines/train_lstm_har_features.py`

Steps (per stock):
1. Load prices -> compute rv_h1, rv_h5, rv_h10, rv_h20 (same as train_lstm_h20.py)
2. Build HAR features from rv_h1: `[rv_d, rv_w, rv_m]`
3. Align: dropna() on HAR features AND all rv_hX targets together
4. Split: pre-2026 -> 80% train / 20% val; 2026+ -> test (R6)
5. Print data split with ESS_h20 (R3)
6. Normalize: feat_mu/feat_sig per feature on train; rv_mu/rv_sig per horizon on train
7. Build stride=1 sequences: X=(M, LOOKBACK, 3), y=(M, 4)
8. Train MIMO LSTM (input_size=3), print per-horizon loss every epoch (R2)
9. Early stopping on total val loss (patience=25)
10. Save learning curve PNG to `results/lstm_har_curves/{ticker}_curve.png` (R2)
11. Evaluate on test: denormalize, compute MAE/RMSE/R2/DA per horizon
12. Save model to `models/lstm_har/{ticker}_model.pt`

### Phase 2: evaluate_lstm_har_features.py

**File:** `scripts/eda/evaluate_lstm_har_features.py`

Mirrors `evaluate_lstm_h20.py` but:
- Loads from `models/lstm_har/`
- Reads `feat_mu`, `feat_sig` from checkpoint for HAR feature normalization
- Same HAR-RV comparison
- Output: `results/lstm_har_results.csv`, `results/lstm_har_summary_metrics.png`

### Phase 3: compare_lstm_variants.py

**File:** `scripts/eda/compare_lstm_variants.py`

**Primary thesis output.** Loads both result CSVs + HAR-RV results and produces:

```
3-way comparison per horizon (avg over all stocks):
  Model              h=1 R2   h=5 R2  h=10 R2  h=20 R2
  LSTM (abs_ret)     -0.056   -0.282   -0.590   -1.282
  LSTM (HAR feats)   ?????    ?????    ?????    ?????
  HAR-RV             -0.053    0.611    0.808    0.896
```

Outputs:
- Console: comparison table
- `results/lstm_comparison_r2.png` — grouped bar chart, 4 panels (1 per horizon)
- `results/lstm_comparison_table.csv` — 3-model x 4-horizon R2/MAE/RMSE

---

## Critical Files

| Task | File |
|---|---|
| RV labels | `src/volatility_labels.py:compute_rv()` |
| HAR baseline | `baselines/har_rv_baseline.py:build_har_features()` |
| Reference LSTM | `baselines/train_lstm_h20.py` |
| Stock list | `gnn/build_graph.py:VN30_TICKERS` |
| Existing results | `results/lstm_h20_results.csv` (R2=-1.28 at h=20) |
| Training (NEW) | `baselines/train_lstm_har_features.py` |
| Evaluation (NEW) | `scripts/eda/evaluate_lstm_har_features.py` |
| Comparison (NEW) | `scripts/eda/compare_lstm_variants.py` |

---

## Verification

```
# Phase 1
python baselines/train_lstm_har_features.py --all
# Check: results/lstm_har_curves/ has PNG per stock
# Check: models/lstm_har/ has .pt per stock
# Expect: val loss converges faster than train_lstm_h20.py

# Phase 2
python scripts/eda/evaluate_lstm_har_features.py
# Expect: h=20 R2 significantly better than -1.28
# Target: R2 > 0.5 for majority of stocks

# Phase 3
python scripts/eda/compare_lstm_variants.py
# Expect: table shows clear gap between the two LSTM variants
```

---

## Expected Results

```
Hypothesis (from Branco et al. 2024 + HAR-LSTM literature):

  h=20 avg R2:
    LSTM (abs_ret input):  -1.28     <- current, confirmed
    LSTM (HAR features):   +0.60     <- expected, range 0.4-0.80
    HAR-RV:                +0.90     <- current, confirmed

  If hypothesis holds:
    -> Feature engineering explains ~85% of the HAR-LSTM gap
    -> Model non-linearity explains remaining ~15%
    -> Thesis conclusion: per-stock data scarcity + feature quality dominate
```

---

## Architectural Decision Notes

**Why not just add rv_h20 lagged as a 4th feature?**
Keeping features to the classic 3 HAR components (rv_d, rv_w, rv_m) makes
the ablation clean: same information as HAR-RV OLS, just a different learner.
Adding rv_h20 lag would be a 2nd ablation step (can be added later).

**Why keep LOOKBACK=20?**
Consistent with train_lstm_h20.py. The rolling window of HAR features over 20
days gives the LSTM richer sequential signal than HAR-RV's single-point features.
If LSTM-HAR outperforms HAR-RV, this is the mechanism: temporal patterns in HAR.

**Why input_size=3 not 1?**
A single rv_h20 feature per timestep (input_size=1) would work but loses the
multi-scale (daily/weekly/monthly) decomposition that HAR relies on.
3 features preserve interpretability and match the literature's HAR-LSTM design.

---

## Notes

- No emoji or Unicode icons in any script (CLAUDE.md rule)
- ASCII-only console output
- Keep `train_lstm_har_features.py` independent from `train_lstm_h20.py`
- HAR feature construction uses pandas rolling (not for-loops) for efficiency
- Minimum rows for training: LOOKBACK + 20 (HAR warm-up) + BATCH_SIZE = 72 rows
