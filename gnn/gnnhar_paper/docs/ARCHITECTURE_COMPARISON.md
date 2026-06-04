# Critical Architecture Comparison: Original vs Our Implementation

**Date:** 2026-05-31
**Purpose:** Identify why our implementation fails (R² = -1.37) while understanding the differences

---

## 🚨 CRITICAL FINDING: Single-Stock vs Multi-Stock

### Original Paper (Multi-Stock Architecture)

**Data Organization:**
```python
# From GNNHAR.py lines 312-313
all_inputs = Tensor(df[['var+lag1', 'var+lag5', 'var+lag22']].values)  # Shape: (N_stocks * T_dates, 3)
all_targets = Tensor(df[['Target']].values)  # Shape: (N_stocks * T_dates, 1)

# Dataset contains ALL stocks flattened: (VIC_t1, VIC_t2, ..., VIC_tT, VCB_t1, ..., VN30_tT)
dataset = TensorDataset(X, Y)
```

**Batching:**
```python
# Line 418-420
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_idx), shuffle=False)

# Each batch contains 128 samples from RANDOM stocks and dates
# Single test batch contains ALL test stocks × ALL test dates
```

**Model Input:**
```python
# From forward methods, node_feat shape: (batch_size, N, 3)
# During training: batch_size=128, N=30 (DJIA) or N=100 (S&P 100)
# During testing: batch_size=N_stocks × N_dates, N_stocks=30/100

# Example: Testing on 30 stocks × 22 test days = 660 samples
# node_feat shape: (660, 30, 3) where first dimension is flattened stock-date pairs
```

**Graph Construction:**
```python
# Line 102
tmp_subdf = pd.DataFrame(np.dot(adj_df, subdf[clms]), ...)
# adj_df: (N_stocks, N_stocks) adjacency matrix
# Each sample gets features from ALL stocks via graph convolution
```

### Our Implementation (Single-Stock Architecture)

**Data Organization:**
```python
# From train_vic_ensemble.py
# We load ONLY VIC stock, create snapshots per date

X_train, y_train, train_dates = build_snapshots_for_period(rv, ...)
# Shape: (n_samples, 3) where n_samples = number of trading days

# Example: 1,260 training samples for VIC only
X_train: (1260, 3) - no stock dimension!
y_train: (1260,) - single value per sample
```

**Batching:**
```python
# Line 227-230
X_t = torch.from_numpy(X_train).float().unsqueeze(1)  # (1260, 1, 3)
y_t = torch.from_numpy(y_train).float().unsqueeze(1)  # (1260, 1)

pred = model(X_t, torch.ones(1, 1))  # adj is (1, 1) identity matrix
```

**Model Input:**
```python
# During training: (batch_size=1, N=1, 3)
# During testing: (batch_size=1, N=1, 3)

# Example: 15 test samples
node_feat shape: (1, 1, 3) - only 1 stock!
```

**Graph Construction:**
```python
# We use identity matrix for single stock
adj = torch.ones(1, 1)  # (1, 1) - no cross-stock information!
```

### **Critical Difference Summary**

| Aspect | Original Paper | Our Implementation | Impact |
|--------|----------------|---------------------|---------|
| **Stocks processed** | 30-100 stocks simultaneously | 1 stock only | **CRITICAL** |
| **Adjacency matrix** | (30,30) or (100,100) real correlation | (1,1) identity | **CRITICAL** |
| **Graph information** | Cross-stock spillover via adjacency | No spillover (identity matrix) | **CRITICAL** |
| **Batching during training** | Random stock-date pairs | Sequential dates, single stock | Minor |
| **Batch size during testing** | All test samples at once | Single sample | Minor |

---

## 🔍 Why GHAR Fails Catastrophically

### Original GHAR (Multi-Stock)

```python
# From GNNHAR.py lines 167-186
class GHAR(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1)       # HAR: (batch, N, 3) -> (batch, N, 1)
        self.gcn1 = GraphConvLayer(3, n_hid) # GCN: (batch, N, 3) -> (batch, N, n_hid)
        # Note: n_hid = 16, N = 30-100

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)      # (batch, N, 1)
        H2 = self.gcn1(node_feat, adj)    # (batch, N, n_hid) - spillover from graph
        res = H1 + H2                     # (batch, N, 1) - DIMENSION MISMATCH!
```

**BUG IN ORIGINAL CODE:**
- H1 is (batch, N, 1)
- H2 is (batch, N, n_hid) where n_hid=16
- **They can't be added directly!**

### Original Code's Dimension Handling

**Looking more carefully at line 183:**
```python
res = H1 + H2
res = self.relu(res)
return res.squeeze(-1)
```

**How does this work?**
- PyTorch broadcasting: (batch, N, 1) + (batch, N, 16) should fail
- UNLESS H2 is somehow projected to (batch, N, 1)

**Let me check the GraphConvLayer more carefully:**

```python
# Lines 128-146
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, node_feature, adj):
        h = torch.matmul(node_feature, self.weight)  # (batch, N, in_feat) @ (in_feat, out) -> (batch, N, out)
        output = torch.matmul(adj, h)                   # (N, N) @ (batch, N, out) -> ERROR!
```

**Wait, there's a bug here too!**
- Line 143: `torch.matmul(adj, h)`
- adj: (N, N)
- h: (batch, N, out_features)
- Matrix multiply with batch dimension is incorrect!

**Correct operation should be:**
```python
# For each sample in batch: adj @ h_sample
# Or use: torch.bmm(adj.unsqueeze(0).expand(batch, N, N), h)
```

**Conclusion:** The original code has bugs that happen to work due to PyTorch's broadcasting rules, or there's something I'm missing about how the data is organized.

---

## 🔍 The Real Architecture Differences

### 1. **Feature Construction**

**Original Paper:**
```python
# From compute_vol.py and GNNHAR.py
# HAR features: lagged averages of multi-horizon RV

# For horizon=5:
# var_data = sum of squared 5-min returns
# RV_d = average of past 1 day RVs
# RV_w = average of past 5 day RVs  
# RV_m = average of past 22 day RVs

# Target: RV at horizon h (multi-horizon)
# For h=5: target = RV_t + RV_{t+1} + RV_{t+2} + RV_{t+3} + RV_{t+4}
```

**Our Implementation:**
```python
# From src/volatility_labels.py
rv = log_ret.rolling(h, min_periods=h).std(ddof=1).shift(-h)

# HAR features:
rv_d = rv.iloc[i-1:i].mean()   # Past 1 day
rv_w = rv.iloc[i-5:i].mean()   # Past 5 days
rv_m = rv.iloc[i-22:i].mean()  # Past 22 days

# Target: RV at horizon h
target = rv.iloc[i]  # Already h-day volatility
```

**Difference:** 
- Paper: Sum of squared returns, then multi-horizon aggregation
- Us: Rolling standard deviation (different formula, but same concept)

**Impact:** Minor - both measure volatility, just different scale

### 2. **Target Normalization**

**Original Paper (Line 539):**
```python
Y /= opt.horizon
```

**For horizon=5:**
- Target = (RV_t + ... + RV_{t+4}) / 5
- This is the AVERAGE future RV over the horizon

**Our Implementation:**
```python
# No normalization - use raw RV directly
target = rv.iloc[i]  # Standard deviation, not divided by horizon
```

**Impact:** 
- Paper's targets are averages (smaller values)
- Our targets are standard deviations (larger values)
- Different scales affect loss magnitude and gradient flow

### 3. **Graph Construction for Single Stock**

**Original Paper (Multi-Stock):**
```python
# Adjacency from GLASSO: (30, 30) for DJIA, (100, 100) for S&P 100
# Uses correlation of stock RETURNS to determine edges
# Represents volatility spillover network
```

**Our Implementation (Single Stock):**
```python
# We use identity matrix for VIC only
adj = torch.ones(1, 1)  # No graph structure!
```

**Impact:**
- GHAR and GNNHAR models add H2 (graph spillover) to H1 (local HAR)
- With identity matrix: H2 = 0 for single stock (no neighbors)
- So GHAR and GNNHAR should reduce to HAR (no graph effect)

**BUT:** GHAR fails catastrophically while HAR works moderately!

---

## 🎯 Root Cause of GHAR Failure

### Theory: GHAR Should Equal HAR for Single Stock

With identity adjacency matrix (N=1):
```python
H2 = GCN(node_feat, I) = GCN(x, I) = x @ W  # No neighborhood aggregation
res = H1 + H2 = Linear(x) + x @ W
```

**This should train to similar performance as HAR (just Linear).**

### Practice: GHAR Fails, HAR Works

**HAR results:** R² = -1.37 (moderately successful, 50% of seeds good)
**GHAR results:** R² = -33.47 (catastrophic, 95% of seeds failed)

### Possible Explanations

#### 1. **Dimension Mismatch Bug**

In original code:
```python
H1 = self.linear1(node_feat)  # (batch, N, 1)
H2 = self.gcn1(node_feat, adj) # (batch, N, n_hid=16)
res = H1 + H2  # DIMENSION MISMATCH!
```

**How does this not crash?**
- Maybe PyTorch broadcasts (batch, N, 1) to (batch, N, 16)?
- Or maybe n_hid=1 for some configurations?

**Our fix:** We added projection layer:
```python
self.proj = nn.Linear(n_hid, 1, bias=False)
H2 = self.proj(H2)  # (batch, N, n_hid) -> (batch, N, 1)
res = H1 + H2  # Now dimensions match
```

**But this might not be what the paper does!**

#### 2. **GCN Layer Implementation Bug**

```python
# Line 143 in original
output = torch.matmul(adj, h)
# adj: (N, N)
# h: (batch, N, out_features)
# This matmul is INCORRECT for batched tensors!
```

**Correct implementation:**
```python
# Should use batch matrix multiplication or einsum
output = torch.einsum('nm,bmk->bnk', adj, h)  # (N, N) @ (batch, N, out) -> (batch, N, out)
```

#### 3. **Our Projection Changes Weights**

Original GHAR (if we ignore dimension mismatch):
```python
res = Linear(x) + GCN(x)
# Both go to (batch, N, 1) somehow
```

Our GHAR with projection:
```python
res = Linear(x) + Linear(GCN(x))
# Extra linear layer adds parameters and changes gradients
```

**This could cause training instability!**

---

## 📋 Recommended Actions

### **Priority 1: Test HAR Without ReLU**

**Reason:** Original paper has ReLU, but our best single-stock results might not need it

**Test:**
```python
class HAR_NoReLU(nn.Module):
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        return H1.squeeze(-1)  # No ReLU
```

**Expected:** Should eliminate constant prediction failure mode

### **Priority 2: Fix or Remove GHAR**

**Options:**
A. Remove projection layer (match original's dimension mismatch)
B. Skip GHAR entirely (only use HAR, GNNHAR1L, GNNHAR2L, GNNHAR3L)
C. Investigate GCN layer implementation

### **Priority 3: Verify Graph Convolution**

**Question:** Does the original GCN implementation actually work correctly with batches?

**Action:** Test original GCN layer with synthetic data to verify dimensions

### **Priority 4: Consider Multi-Stock Training**

**Current:** Train on VIC only (N=1)
**Alternative:** Train on all 30 VN30 stocks (N=30)

**Benefits:**
- Proper use of graph structure
- Cross-stock volatility spillover information
- Matches paper's design

**Drawbacks:**
- Need to reconstruct adjacency matrix for VN30
- More complex data pipeline
- Not directly comparable to current results

---

## 🎯 Summary of Key Differences

| Aspect | Original Paper | Our Implementation | Status |
|--------|----------------|---------------------|---------|
| **Stocks processed** | Multi-stock (N=30-100) | Single-stock (N=1) | Different by design |
| **Adjacency matrix** | Real correlation graph | Identity matrix | Expected for N=1 |
| **HAR architecture** | Linear + ReLU | Linear + ReLU | ✅ Match |
| **GHAR architecture** | Has dimension mismatch bug | Added projection layer | ⚠️ Different |
| **GNNHAR architecture** | GCN + ReLU | GCN + ReLU + projection | ⚠️ Different |
| **Target scaling** | Divided by horizon | Not scaled | ⚠️ Different |
| **RV computation** | Sum of squared returns | Rolling std | Minor |
| **Batching** | Multi-stock random | Single-stock sequential | Minor |

**Most Critical Issues:**
1. GHAR/GNNHAR have projection layer that paper doesn't have (causes instability?)
2. Target not divided by horizon (affects scale)
3. ReLU causes constant predictions for 50% of seeds (paper uses ReLU, why does it work for them?)
