# GNNHAR Paper Full Replication Plan

**Date:** 2026-05-29
**Source:** Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks (IJF 2024)
**Reference code:** `GNNHAR/GNNHAR.py`

---

## 1. Architecture Overview

### 1.1 Core Components

```
Input: (N, 3) -- HAR features [rv_d, rv_w, rv_m] for N stocks at time t
Models:
  - HAR:     Linear(3,1) + ReLU (baseline)
  - GHAR:    Linear(3,1) + GCN(3,n_hid) + ReLU (residual)
  - GNNHAR1L: Linear(3,1) + GCN(3,n_hid) + ReLU + MLP(n_hid,1) + ReLU
  - GNNHAR2L: Linear(3,1) + 2xGCN + MLP(n_hid,1) + ReLU
  - GNNHAR3L: Linear(3,1) + 3xGCN + MLP(n_hid,1) + ReLU
```

### 1.2 GraphConvLayer (GCN)

```python
# Paper's implementation (line 128-146):
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)

    def forward(self, node_feature, adj):
        h = torch.matmul(node_feature, self.weight)    # (N, out_features)
        output = torch.matmul(adj, h)                  # message passing
        if self.bias is not None:
            return output + self.bias
        return output
```

Key difference from current GraphSAGE:
- GCN: `AXW` (spectral, requires normalized adjacency)
- GraphSAGE: `concat(h_i, mean(h_j))W` (spatial, inductive)

### 1.3 Model Variants Detail

**HAR (baseline):**
```python
H1 = Linear(3, 1)(node_feat)    # (N, 1)
output = ReLU(H1)
```

**GHAR (Graph-augmented HAR):**
```python
H1 = Linear(3, 1)(node_feat)             # local HAR
H2 = GCN(3, n_hid)(node_feat, adj)       # spillover from graph
output = ReLU(H1 + H2)                    # residual connection
```

**GNNHAR1L (1-layer GNN):**
```python
H1 = Linear(3, 1)(node_feat)                        # local HAR
H2 = GCN(3, n_hid)(node_feat, adj)                 # spillover
H2 = ReLU(H2)
H2 = MLP(n_hid, 1)(H2)                             # nonlinear transform
output = ReLU(H1 + H2)                             # residual
```

**GNNHAR2L (2-layer GNN):**
```python
H1 = Linear(3, 1)(node_feat)
H2 = ReLU(GCN(3, n_hid)(node_feat, adj))          # 1-hop neighbors
H2 = ReLU(GCN(n_hid, n_hid)(H2, adj))              # 2-hop neighbors
H2 = MLP(n_hid, 1)(H2)
output = ReLU(H1 + H2)
```

**GNNHAR3L (3-layer GNN):** Same pattern with 3 GCN layers.

---

## 2. Adjacency Matrix Construction

### 2.1 GLASSO (Graphical Lasso)

Paper uses `GraphicalLassoCV` from sklearn to build sparse precision matrix:
```python
def GLASSO_Precision(subret):
    from sklearn.covariance import GraphicalLassoCV
    n = subret.shape[1]
    tickers = subret.columns
    cov = GraphicalLassoCV().fit(subret)
    print('Alpha in GLASSO: %.3f' % cov.alpha_)
    corr = cov.precision_ != 0
    print('Sparsity of Adj: %.3f' % corr.mean())
    corr_adj = corr - np.identity(n)
    # Symmetric normalization
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum(1)+1e-8)))
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv),
                          columns=tickers, index=tickers)
    return adj_df
```

**Properties:**
- Input: return data (NOT volatility)
- Output: sparse symmetric adjacency matrix
- Sparsity: typically 5-15% non-zero edges
- Self-loops removed (`- np.identity(n)`)

### 2.2 Rolling Window Adjacency

Paper recomputes adjacency for each training window:
```python
def Compute_Adj(ret_df, vech_df, date, date_l):
    timestamp = date_l.index(date)
    s_p = max(timestamp-1000, 0)      # 1000-day lookback
    v_p = timestamp - valid_len
    f_p = min(timestamp + window, len(date_l)-1)

    subret = ret_df[(ret_df.index >= s_date) & (ret_df.index < date)]
    adj_df = GLASSO_Precision(subret)
    return adj_df, s_p, v_p, timestamp, f_p
```

---

## 3. Training Scheme

### 3.1 Rolling Window Approach

```
For each date in test_period (every `window` days):
  1. Build adjacency from 1000-day lookback returns
  2. Train: s_date to v_date
  3. Validate: v_date to current_date
  4. Test: current_date to f_date (22-day window)
  5. Save model checkpoint
```

**Parameters:**
- `window=22`: rolling window size
- `valid_len=22`: validation period
- Lookback: 1000 days for GLASSO

### 3.2 Ensemble Training

```python
def Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date):
    train_idx = range(s_p, v_p)
    val_idx = range(v_p, timestamp)
    test_idx = range(timestamp, f_p)

    # Train multiple models with different seeds
    for iii in range(opt.ens * opt.numNN, (opt.ens + 1) * opt.numNN):
        seed = np.random.randint(low=1, high=10000)
        loss_df = Train_Single(train_loader, val_loader,
                                model_index=iii, seed=seed, date=date)

        # Restart if training diverged
        while (np.abs(loss_df['Valid'].diff()) < 1e-6).mean() > 0.5:
            seed = np.random.randint(low=1, high=10000)
            loss_df = Train_Single(...)
```

### 3.3 Ensemble Screening

```python
def Screen_Ensemble(date, thres_perc=50):
    loss_l = []
    for j in range(opt.numNN):
        loss_df = pd.read_csv(f'loss_{date}_index{j}.csv')
        loss_l.append(loss_df['Valid'].iloc[-1])

    threshold_loss = np.percentile(loss_l, thres_perc)
    select_l = [j for j in range(opt.numNN) if loss_l[j] <= threshold_loss]
    return select_l
```

**Logic:** Keep models with validation loss below 50th percentile, discard poorly converged models.

### 3.4 Prediction Aggregation

```python
def connect_pred():
    for date in dates_l:
        tmp_pred_df_l = []
        select_l = Screen_Ensemble(date)
        for j in select_l:
            tmp_test_pred_df = pd.read_csv(f'Pred_{date}_Ens{j}.csv')
            tmp_pred_df_l.append(tmp_test_pred_df)

        # Average over screened models
        test_pred_df = pd.DataFrame(np.stack(tmp_pred_df_l).mean(0))
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l) * opt.horizon
```

---

## 4. Implementation Plan

### Phase 1: Core Components (2-3 days)

**File: `moirai/gnn/gcn_layer.py`**
- [ ] `GraphConvLayer` class matching paper exactly
- [ ] Xavier initialization with ReLU gain
- [ ] Optional bias with ones initialization
- [ ] Forward: `h = XW -> AXW`

**File: `moirai/gnn/gnnhar_models.py`**
- [ ] `HAR` baseline model
- [ ] `GHAR` model (linear + GCN, no MLP)
- [ ] `GNNHAR1L` model (linear + 1 GCN + MLP)
- [ ] `GNNHAR2L` model (linear + 2 GCN + MLP)
- [ ] `GNNHAR3L` model (linear + 3 GCN + MLP)
- [ ] `count_params()` method for each

**File: `moirai/gnn/glasso_adjacency.py`**
- [ ] `GLASSO_Precision()` function
- [ ] Symmetric normalization
- [ ] Sparsity logging
- [ ] Input validation (return data, not RV)

### Phase 2: Rolling Window Infrastructure (2 days)

**File: `moirai/gnn/rolling_window.py`**
- [ ] `Compute_Adj()` function with date indexing
- [ ] `get_lag_avg()` for HAR features (paper uses simple avg)
- [ ] Window management: s_p, v_p, f_p calculation
- [ ] Adjacency caching per window

**File: `moirai/gnn/rolling_datasets.py`**
- [ ] `build_rolling_snapshots()` function
- [ ] Return: (X, y, train_idx, val_idx, test_idx) per window
- [ ] Handle variable window sizes
- [ ] Memory-efficient for large rolling windows

### Phase 3: Ensemble Training (2 days)

**File: `moirai/gnn/ensemble_trainer.py`**
- [ ] `Train_Single()` function with seed control
- [ ] Divergence detection (`valid.diff() < 1e-6`)
- [ ] Learning curve saving per model
- [ ] Checkpoint saving: `Best_Model_{date}_index{iii}.pt`

**File: `moirai/gnn/ensemble_screen.py`**
- [ ] `Screen_Ensemble()` with percentile threshold
- [ ] Load all loss files for a date
- [ ] Return list of valid model indices
- [ ] Configurable threshold (default 50)

### Phase 4: Main Training Script (2 days)

**File: `moirai/gnn/train_gnnhar_paper.py`**
- [ ] Argument parser matching paper
- [ ] Data loading (var, return, feature CSVs)
- [ ] HAR feature construction (lag1, lag5, lag22 averages)
- [ ] Main loop: for date in test_period
- [ ] Per-window: Train -> Screen -> Predict
- [ ] Aggregate predictions across windows
- [ ] Per-horizon loop (h=1,5,10,20)

### Phase 5: Evaluation & Comparison (1 day)

**File: `moirai/gnn/evaluate_gnnhar.py`**
- [ ] Load predictions from all windows
- [ ] Concatenate into full test period
- [ ] Per-stock metrics: R2, MAE, RMSE
- [ ] Compare vs HAR baseline
- [ ] Compare vs existing GraphSAGE results
- [ ] Summary statistics table

---

## 5. Key Differences from Current Implementation

| Aspect | Current | Paper | Change Required |
|--------|---------|-------|-----------------|
| GNN Layer | GraphSAGE | GCN (AXW) | New `GraphConvLayer` |
| Models | Single 2-layer | HAR/GHAR/1L/2L/3L | Add 4 new model classes |
| Graph | Static Pearson | Rolling GLASSO | New adjacency builder |
| Training | Static 80/20 | Rolling window | New dataset builder |
| Ensemble | None | 5-10 models + screening | New ensemble system |
| HAR Features | Custom | Simple lag average | Match paper exactly |

---

## 6. Design Decisions & Trade-offs

### 6.1 GCN vs GraphSAGE

**Paper choice (GCN):**
- Pro: Matches published results exactly
- Pro: Simpler (no aggregator logic)
- Con: Requires normalized adjacency
- Con: Transductive only

**Current (GraphSAGE):**
- Pro: Inductive (works on new nodes)
- Pro: Robust to graph structure changes
- Con: Different from paper

**Decision:** Implement GCN for paper replication, keep GraphSAGE as comparison.

### 6.2 GLASSO vs Static Pearson

**Paper (GLASSO rolling):**
- Pro: Data-driven sparse graph
- Pro: Captures changing correlations
- Con: Expensive (recompute every window)
- Con: Unstable on small windows

**Current (static Pearson):**
- Pro: Fast (compute once)
- Pro: Stable over time
- Con: May miss regime changes

**Decision:** Implement GLASSO rolling for paper replication, compare with static.

### 6.3 HAR Feature Construction

**Paper:**
```python
lag1 = get_lag_avg(feature_df, 1)   # avg of last 1 day
lag5 = get_lag_avg(feature_df, 5)   # avg of last 5 days
lag22 = get_lag_avg(feature_df, 22)  # avg of last 22 days
```

**Current:**
```python
rv_d = |log_return[t]|           # daily absolute return
rv_w = std(log_ret[t-4:t+1])    # 5-day rolling std
rv_m = std(log_ret[t-19:t+1])   # 20-day rolling std
```

**Difference:** Paper uses simple average of past RV values, current uses rolling std of returns.

**Decision:** Match paper exactly for replication, keep current for comparison.

---

## 7. File Structure

```
moirai/gnn/
├── gcn_layer.py              # GraphConvLayer implementation
├── gnnhar_models.py          # HAR/GHAR/1L/2L/3L model classes
├── glasso_adjacency.py       # GLASSO adjacency builder
├── rolling_window.py         # Rolling window utilities
├── rolling_datasets.py       # Snapshot builder for rolling scheme
├── ensemble_trainer.py       # Single model training
├── ensemble_screen.py        # Ensemble screening logic
├── train_gnnhar_paper.py     # Main training script
├── evaluate_gnnhar.py        # Evaluation & comparison
├── har_model.py              # [KEEP] existing GraphSAGE
├── har_graph.py              # [KEEP] existing graph builder
└── train_gnn_har.py          # [KEEP] existing training

moirai/models/
└── gnnhar_paper/
    ├── h1/
    │   ├── HAR/
    │   ├── GHAR/
    │   ├── GNNHAR1L/
    │   ├── GNNHAR2L/
    │   └── GNNHAR3L/
    ├── h5/ ...
    ├── h10/ ...
    └── h20/ ...

moirai/results/
└── gnnhar_paper/
    ├── predictions/
    ├── losses/
    └── gnnhar_paper_results.csv
```

---

## 8. Hyperparameters (from paper)

```python
window      = 22     # rolling window size
valid_len   = 22     # validation period
n_epochs    = 5000   # training epochs
n_hid       = 9      # hidden neurons
batch_size  = 128    # batch size
lr          = 1e-3   # learning rate
ens         = 0      # ensemble index
numNN       = 5      # number of neural networks per ensemble
loss        = "MSE"  # or "QLIKE" (deprecated in our project)
```

**Our adjustments:**
- `n_hid=16` (match current, better than 9 for ESS)
- `n_epochs=500` (5000 is excessive with early stopping)
- `numNN=5` (balance between compute and stability)
- `weight_decay=1e-3` (add regularization, paper uses 1e-5)

---

## 9. Success Criteria

1. **Implementation Completeness:**
   - All 5 model variants (HAR, GHAR, 1L, 2L, 3L) implemented
   - GLASSO adjacency construction working
   - Rolling window training functional
   - Ensemble screening operational

2. **Results Quality:**
   - GNNHAR variants outperform HAR baseline
   - Results comparable to paper (directionally, not exact values)
   - Consistent improvements across horizons

3. **Comparison:**
   - Clear comparison table: Paper GCN vs Current GraphSAGE
   - Per-stock breakdown
   - Statistical significance testing (if time permits)

---

## 10. Open Questions

1. **Data leakage in paper?** Paper uses `get_lag_avg()` on `feature_df` which is RV. Does this include current day's RV? (Potential lookahead)
2. **QLIKE removal:** Paper supports QLIKE loss. Our project removed it. Use MSE only?
3. **Horizon handling:** Paper runs separate experiments per horizon. Should we do MIMO or SISO?
4. **Compute budget:** Full ensemble with rolling windows is expensive. Can we parallelize?

**Answers:**
1. Will audit feature construction carefully for leakage
2. Use MSE only (per project decision 2026-05-27)
3. Start SISO (one model per horizon) to match paper, MIMO later
4. Parallelize per-window training if needed

---

## 11. Timeline Estimate

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| 1 | GCN layer + 5 model classes | 2 days | - |
| 2 | GLASSO + rolling datasets | 2 days | Phase 1 |
| 3 | Ensemble trainer + screening | 2 days | Phase 1 |
| 4 | Main training script | 2 days | Phase 2,3 |
| 5 | Evaluation + comparison | 1 day | Phase 4 |
| 6 | Run experiments | 3-5 days | Phase 4 |
| **Total** | | **12-14 days** | |

**Accelerated path:** Skip Phase 2 (rolling window), use static 80/20 split with GLASSO adjacency → saves 2 days, simpler to debug.

---

## 12. Next Steps

1. Start with Phase 1: implement `GraphConvLayer` and 5 model classes
2. Test with synthetic data: verify forward pass works
3. Move to Phase 2: GLASSO adjacency on VN30 returns
4. Debug rolling window on small sample (3-4 windows)
5. Run full experiment for h=1 first, validate results
6. Extend to h=5,10,20

**First commit:** `gcn_layer.py` + `gnnhar_models.py` with unit tests
