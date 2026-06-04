# GNNHAR Original Paper vs Our Implementation - Detailed Comparison

**Date:** 2026-05-31
**Source:** https://github.com/chaozhang-ox/GNNHAR
**Paper:** "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)

---

## 1. MODEL ARCHITECTURE COMPARISON

### Original Paper (GNNHAR.py)

#### HAR Model (Lines 149-163)
```python
class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.relu = nn.ReLU()  # ACTIVE ReLU in output!

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)  # (batch_size, N, 3)
        res = self.relu(H1)             # <-- ReLU ACTIVATION
        return res.squeeze(-1)
```

**KEY FINDING:** Original paper **HAS ReLU activation** on HAR output!

#### GHAR Model (Lines 167-186)
```python
class GHAR(nn.Module):
    def __init__(self, n_hid):
        super(GHAR, self).__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)      # Local HAR
        H2 = self.gcn1(node_feat, adj)    # Graph spillover
        res = H1 + H2                    # Residual sum
        res = self.relu(res)              # <-- ReLU ON OUTPUT
        return res.squeeze(-1)
```

#### GNNHAR1L Model (Lines 189-212)
```python
class GNNHAR1L(nn.Module):
    def __init__(self, n_hid):
        super(GNNHAR1L, self).__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H2 = self.gcn1(node_feat, adj)
        H2 = self.relu(H2)                # ReLU after GCN
        H2 = self.mlp1(H2)                # Linear projection
        res = H1 + H2                    # Residual
        res = self.relu(res)              # <-- ReLU ON OUTPUT
        return res.squeeze(-1)
```

### Our Implementation (gnn/gnnhar_models.py)

**WE REMOVED ReLU from all models** (bug fix 2026-05-31):

```python
class HAR(nn.Module):
    """HAR baseline - FIX: Removed ReLU activation (2026-05-31)"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        # No ReLU - allow linear output for z-scored residuals

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        H1 = self.linear1(node_feat)
        return H1.squeeze(-1)  # No activation, pure linear output
```

**CRITICAL DIFFERENCE:**
- **Original:** Has ReLU on output → predictions ≥ 0
- **Ours:** No ReLU → predictions can be negative (z-scored space)

---

## 2. TEST SET METHODOLOGY COMPARISON

### Original Paper (GNNHAR.py Lines 279-312, 409-420)

#### Rolling Window Approach
```python
def Compute_Adj(ret_df, vech_df, date, date_l):
    timestamp = date_l.index(date)
    s_p = max(timestamp-1000, 0)        # 1000-day lookback
    v_p = timestamp - opt.valid_len      # validation start
    f_p = min(timestamp + opt.window, len(date_l)-1)  # test end

    s_date = date_l[s_p]
    v_date = date_l[v_p]
    f_date = date_l[f_p]

    subret = ret_df[ret_df.index < date]
    subret = subret[subret.index >= s_date]

    return adj_df, s_p, v_p, timestamp, f_p
```

#### Data Split (Lines 410-420)
```python
def Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date):
    train_idx = range(s_p, v_p)      # Training indices
    val_idx = range(v_p, timestamp)    # Validation indices
    test_idx = range(timestamp, f_p)  # Test indices

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_idx), shuffle=False)
```

#### Rolling Window Loop (Lines 550-555)
```python
date_l = targets.index.tolist()
idx = date_l.index('2011-07-01')

for date in date_l[idx::opt.window]:  # Every 22 days
    print(' * ' * 20 + date + ' * ' * 20)
    adj_df, s_p, v_p, timestamp, f_p = Compute_Adj(...)
    Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date)
```

**KEY PARAMETERS:**
- `--window`: 22 (rolling window size, default)
- `--valid_len`: 22 (validation period)
- `--horizon`: 1 (forecasting horizon)
- **Test period:** 22-day rolling windows

### Our Implementation (gnn/gnnhar_paper/vic/train_vic_ensemble.py)

#### Fixed Test Period
```python
TRAIN_END_DATE = "2026-04-30"  # Train through April
TEST_START_DATE = "2026-05-01"   # Test only May
TEST_END_DATE = "2026-05-31"

# Build snapshots
X_train, y_train = build_snapshots_for_period(
    rv, start_date=rv.index[0], end_date=train_end_ts, stride=1
)
X_test, y_test = build_snapshots_for_period(
    rv, start_date=test_start_ts, end_date=test_end_ts, stride=1
)
```

**ACTUAL RESULTS:**
- Train: 3,689 samples (2007-10-26 to 2022-08-12)
- Validation: 923 samples (2022-08-15 to 2026-04-29)
- Test: **15 samples only** (2026-05-04 to 2026-05-18)

**CRITICAL DIFFERENCE:**
- **Original:** Rolling 22-day test windows throughout entire test period
- **Ours:** Single fixed test period (May 2026 only, 15 samples)

---

## 3. DATA PROCESSING COMPARISON

### Original Paper (compute_vol.py)

#### RV Computation
```python
# From 5-min data to daily variance
def compute_variance(sub_data):
    stocks_l = [i for i in sub_data.columns if i not in ['Date', 'Time']]
    sq_data = sub_data[stocks_l] ** 2
    var_sum = sq_data.sum(min_count=1)
    var_sum = pd.DataFrame(var_sum).T
    return var_sum

# Compute multi-horizon RV
def Compute_Horizon(path, univese, ret_vol, horizon):
    daily_var_data = pd.read_csv('daily_variance.csv', index_col=0)
    var_data = 0
    for i in range(horizon):
        var_data += daily_var_data.shift(-i)
    var_data.dropna(inplace=True)
```

#### HAR Feature Computation (GNNHAR.py Lines 85-89)
```python
def get_lag_avg(df, lag):
    res = pd.DataFrame(columns=df.columns, index=df.index).fillna(0)
    for l in range(1, lag + 1):
        res += (1 / lag) * df.shift(l)
    return res
```

**Usage (Lines 526-533):**
```python
if opt.horizon == 1:
    lag1 = get_lag_avg(feature_df, 1).iloc[22:]
    lag5 = get_lag_avg(feature_df, 5).iloc[22:]
    lag22 = get_lag_avg(feature_df, 22).iloc[22:]
else:
    e_idx = -opt.horizon + 1
    lag1 = get_lag_avg(feature_df, 1).iloc[22:e_idx]
    lag5 = get_lag_avg(feature_df, 5).iloc[22:e_idx]
    lag22 = get_lag_avg(feature_df, 22).iloc[22:e_idx]
```

### Our Implementation (src/volatility_labels.py)

```python
def compute_rv(close: pd.DataFrame, h: int = 20) -> pd.DataFrame:
    """
    Realized volatility: RV_t(h) = std(r_{t+1}, ..., r_{t+h}, ddof=1)
    Implementation:
        rolling(h).std() at index t = std of window [t-h+1, t]
        .shift(-h)        at index t = that value from index t+h
    """
    log_ret = compute_log_returns(close)
    rv = log_ret.rolling(h, min_periods=h).std(ddof=1).shift(-h)
    return rv
```

**DIFFERENCES:**
- **Original:** Sum of squared returns, then multi-horizon aggregation
- **Ours:** Rolling standard deviation with shift (more standard)

---

## 4. ENSEMBLE METHOD COMPARISON

### Original Paper (GNNHAR.py Lines 474-487)

```python
def Screen_Ensemble(date, thres_perc=50):
    loss_l = []
    for j in range(opt.numNN):
        loss_df = pd.read_csv(f'loss_{date}_index{j}.csv')
        loss_l.append(loss_df['Valid'].iloc[-1])  # <-- Last val loss

    threshold_loss = np.percentile(loss_l, thres_perc)
    select_l = []
    for j in range(opt.numNN):
        if loss_l[j] <= threshold_loss:
            select_l.append(j)  # <-- Keep models below 50th percentile
        else:
            pass
    return select_l
```

**Connect Predictions (Lines 500-508):**
```python
def connect_pred():
    for date in dates_l:
        select_l = Screen_Ensemble(date)
        tmp_pred_df_l = []
        for j in select_l:
            tmp_test_pred_df = pd.read_csv(f'Pred_{date}_Ens{j}.csv')
            tmp_pred_df_l.append(tmp_test_pred_df)

        # Average over screened models
        test_pred_df = pd.DataFrame(np.stack(tmp_pred_df_l).mean(0), ...)
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l) * opt.horizon  # <-- Multiply by horizon!
```

### Our Implementation (gnn/gnnhar_paper/vic/train_vic_ensemble.py Lines 341-349)

```python
# Screen by validation loss (keep top 50%)
median_val_loss = np.median(val_losses_list)
screened_indices = [i for i, vl in enumerate(val_losses_list) if vl <= median_val_loss]
screened_preds = [predictions_list[i] for i in screened_indices]

# Average predictions
ensemble_pred = np.mean(screened_preds, axis=0)

# Ensemble metrics
r2_ensemble = 1 - np.sum((y_test - ensemble_pred)**2) / np.sum((y_test - y_test.mean())**2)
```

**DIFFERENCES:**
- **Original:** Screens by val loss, then multiplies predictions by horizon
- **Ours:** Screens by val loss, but doesn't multiply by horizon (we predict RV directly)

---

## 5. KEY DIFFERENCES SUMMARY

| Aspect | Original Paper | Our Implementation | Impact |
|--------|----------------|---------------------|---------|
| **HAR ReLU** | ✅ HAS ReLU on output | ❌ Removed (bug fix) | **MAJOR** - Changes prediction space |
| **Test set** | Rolling 22-day windows | Fixed May 2026 (15 samples) | **CRITICAL** - Too small test set |
| **Test period** | Entire test period (e.g., 2011-2021) | Single month | **CRITICAL** - Not representative |
| **RV computation** | Sum of squared returns | Rolling std with shift | Minor - Different formula |
| **Ensemble** | 50th percentile screening | 50th percentile screening | ✓ Same |
| **Prediction scaling** | Multiplied by horizon | Not scaled | **IMPORTANT** - May affect results |

---

## 6. CRITICAL FINDING: ReLU Activation

**The original paper HAS ReLU activation on all model outputs:**

```python
# HAR (line 161)
res = self.relu(H1)

# GHAR (line 184)
res = self.relu(H1 + H2)

# GNNHAR1L (line 210)
res = self.relu(H1 + H2)
```

**This means:**
- Original predictions are **constrained to be ≥ 0**
- This makes sense if predicting RV (realized volatility is always positive)

**Our "fix" may be wrong!** We removed ReLU because we thought it was a bug, but:
1. The paper explicitly uses ReLU
2. RV is always positive, so ReLU constraint is appropriate
3. We're training on **z-scored residuals** (which can be negative), so ReLU would break

**Root cause of our failure:**
- We train on z-scored residuals (mean=0, std=1)
- Z-scored residuals **can be negative**
- ReLU(z-scored_negative) = 0 → kills gradient flow → R² = -100

**The paper's approach:**
- Train on raw RV (always positive)
- Use ReLU to ensure non-negative predictions
- No z-scoring needed

---

## 7. ACTION PLAN

### Phase 1: Verify Training Data

**Question:** Are we training on raw RV or z-scored residuals?

Check `train_vic_ensemble.py`:
```python
# If you see this, we're using z-scoring:
y_train = (y_train - mean) / std

# If you see this, we're using raw RV:
y_train = rv.values
```

### Phase 2: Decide on Approach

**Option A: Match paper exactly (recommended)**
- Remove z-scoring
- Add ReLU back to all models
- Train on raw RV
- Expected: Predictions ≥ 0, better R²

**Option B: Our approach (fix was correct)**
- Keep z-scoring (handles volatility scaling across stocks)
- Keep ReLU removed (allows negative predictions in residual space)
- Need to verify this is theoretically sound

### Phase 3: Fix Test Set

**Regardless of Option A vs B, MUST fix test set:**

Current: 15 samples (May 2026 only)
Should be: ~120 samples (Jan-May 2026, or use rolling windows)

```python
# Update train_vic_ensemble.py
TRAIN_END_DATE = "2025-12-31"  # Use all pre-2026 data for training
TEST_START_DATE = "2026-01-01"   # Match paper's GLOBAL_TEST_START
TEST_END_DATE = "2026-05-31"    # Current data end
```

---

## 8. NEXT STEPS

1. **Check training data format** - Are we using z-scoring?
2. **Re-add ReLU** if training on raw RV
3. **Extend test window** to full 2026 data
4. **Re-train models** with corrected setup
5. **Compare results** with paper

**Expected outcome:** R² should become positive with proper test set and ReLU (if training on raw RV).
