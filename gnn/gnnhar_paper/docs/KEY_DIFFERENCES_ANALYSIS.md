# Key Differences: Why Our Implementation Fails

**Date:** 2026-05-31
**Current Status:** R² = -1.37 (train through Apr 2026, test May 2026)
**Goal:** Identify architectural differences causing failure

---

## 🔴 CRITICAL ISSUE: Projection Layer

### Original GHAR (From GNNHAR.py Lines 167-186)

```python
class GHAR(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1)       # H1: (batch, N, 3) -> (batch, N, 1)
        self.gcn1 = GraphConvLayer(3, n_hid) # H2: (batch, N, 3) -> (batch, N, n_hid)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)  # (batch, N, 1)
        H2 = self.gcn1(node_feat, adj) # (batch, N, n_hid=16)
        
        res = H1 + H2  # DIMENSION MISMATCH: (batch, N, 1) + (batch, N, 16)
        res = self.relu(res)
        
        return res.squeeze(-1)
```

**QUESTION:** How does (batch, N, 1) + (batch, N, 16) not crash?

### Our GHAR (With Projection Layer)

```python
class GHAR(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.proj = nn.Linear(n_hid, 1, bias=False)  # ADDED TO FIX DIMENSION
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)   # (1, 1, 3) -> (1, 1, 1)
        H2 = self.gcn1(node_feat, adj)  # (1, 1, 3) -> (1, 1, 16)
        H2 = self.proj(H2)              # (1, 1, 16) -> (1, 1, 1)
        
        res = H1 + H2  # (1, 1, 1) + (1, 1, 1) -> (1, 1, 1)
        return self.relu(res.squeeze(-1))  # (1, 1)
```

**CRITICAL:** We added `self.proj` but the original code doesn't have it!

### How Original Code Doesn't Crash

**Possibility 1: PyTorch Broadcasting**
```python
# (batch, N, 1) + (batch, N, 16)
# PyTert might broadcast to (batch, N, 16) and ignore the dimension mismatch
# Then squeeze(-1) would give (batch, N), not (batch, N, 1)
```

**Possibility 2: Different n_hid for GHAR**
```python
# Maybe n_hid=1 for GHAR specifically?
# Need to check what opt.n_hid is set to
```

**Possibility 3: Bug in Original Code**
```python
# The original code might have a latent bug that only manifests in certain conditions
# It might "work" during training but produce wrong results
```

---

## 🔴 CRITICAL ISSUE: GCN Matrix Multiply Bug

### From GraphConvLayer.forward (Lines 141-143)

```python
def forward(self, node_feature, adj):
    h = torch.matmul(node_feature, self.weight)
    output = torch.matmul(adj, h)  # <-- THIS IS INCORRECT
```

**Analysis:**
```python
node_feature: (batch_size, N, in_features)
self.weight: (in_features, out_features)
h: (batch_size, N, out_features)

adj: (N, N)

# torch.matmul(adj, h) where:
# adj: (N, N)
# h: (batch, N, out_features)
```

**Matrix multiplication rules:**
- `torch.matmul(A, B)` where A shape (m, n), B shape (n, p) → (m, p)
- But `h` has 3 dimensions, not 2!

**What actually happens:**
```python
# PyTorch treats last 2 dimensions as matrices
# So: (N, N) @ (batch, N, out) is interpreted as:
# Multiply (N, N) by (N, out) for each element in batch dimension
# Result: (batch, N, N) @ (batch, N, out) = (batch, N, out)
# Wait, that doesn't work either...
```

**Correct Implementation Should Be:**
```python
# Option 1: Batch matrix multiply
output = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), h)

# Option 2: Einsum
output = torch.einsum('nm,bmk->bnk', adj, h)

# Option 3: Manual loop
output = torch.stack([torch.matmul(adj, h[i]) for i in range(batch_size)])
```

**Or maybe the adjacency is applied per-sample:**
```python
# If adj is (batch, N, N), then:
output = torch.bmm(adj, h)  # (batch, N, N) @ (batch, N, out) -> (batch, N, out)
```

---

## 🔴 CRITICAL ISSUE: Target Scaling

### Original Code (Line 539)

```python
Y /= opt.horizon
```

**For horizon=1:** No effect (dividing by 1)
**For horizon=5:** Target is divided by 5 (average of 5-day RV)

### Our Implementation

```python
# No division by horizon
target = rv.iloc[i]  # Raw standard deviation
```

**Impact:**
- Paper's targets are scaled (smaller values, ~0.003-0.005 range for averages)
- Our targets are raw (larger values, ~0.015-0.025 range for standard deviations)
- Affects loss magnitude and gradient flow

---

## 🔸 DIFFERENCE: RV Computation Method

### Original Paper (compute_vol.py)

```python
# From 5-minute intraday data
daily_return_data = ret_data.groupby(by='Date').sum(min_count=1)

# Compute daily variance
def compute_variance(sub_data):
    sq_data = sub_data[stocks_l] ** 2
    var_sum = sq_data.sum(min_count=1)  # Sum of squared returns
    return var_sum

# Multi-horizon aggregation
for i in range(horizon):
    var_data += daily_var_data.shift(-i)
var_data.dropna(inplace=True)  # Sum of h daily variances
```

**Result:** RV_t(h) = Σ_{τ=1}^{h×M} r_{t,τ}² (sum of squared returns over h days)

### Our Implementation (src/volatility_labels.py)

```python
def compute_rv(close, h=20):
    log_ret = compute_log_returns(close)
    rv = log_ret.rolling(h, min_periods=h).std(ddof=1).shift(-h)
    return rv
```

**Result:** RV_t(h) = std(r_{t+1}, ..., r_{t+h}) (standard deviation over h days)

**Mathematical Difference:**
- **Paper:** Sum of variances = Σ Var(r) ≠ Var(Σ r)
- **Us:** Variance of returns = std(r)

**Relationship:**
- For i.i.d. returns: std²(r) ≈ mean(r²)
- But with autocorrelation: std²(r) ≠ mean(r²)

**Impact:** Minor - both measure volatility, but different scale

---

## 🟡 MINOR DIFFERENCE: Data Organization

### Original Paper (Multi-Stock Flattened)

```python
# All stocks and dates flattened into single dataset
# Shape: (N_stocks × N_dates, 3)

# Example: 30 stocks × 1000 dates = 30,000 samples
# Each batch of 128 contains random stock-date pairs
```

### Our Implementation (Single Stock Sequential)

```python
# Single stock, sequential dates
# Shape: (N_dates, 3)

# Example: 1,260 training samples for VIC
# Each batch = 1 sample, processed sequentially
```

**Impact:** Minor - shouldn't affect model architecture

---

## 🎯 SUMMARY: Key Architectural Differences

### 1. **GHAR Projection Layer** ⚠️ CRITICAL
- **Original:** Has dimension mismatch bug (or hidden fix)
- **Ours:** Added projection layer to fix dimensions
- **Impact:** Might cause training instability!

### 2. **GCN Matrix Multiply** ⚠️ CRITICAL  
- **Original:** Uses `torch.matmul(adj, h)` which is incorrect for 3D tensors
- **Ours:** Same issue (copied from original)
- **Impact:** Might produce incorrect gradients

### 3. **Target Scaling** ⚠️ IMPORTANT
- **Original:** Targets divided by horizon
- **Ours:** Raw targets (no scaling)
- **Impact:** Different value ranges, affects loss

### 4. **RV Computation** 🟢 MINOR
- **Original:** Sum of squared returns
- **Ours:** Rolling standard deviation  
- **Impact:** Different formula, similar concept

### 5. **ReLU Activation** ✅ MATCH
- **Original:** Has ReLU on all outputs
- **Ours:** Added ReLU back (after initially removing)
- **Impact:** Constrains predictions ≥ 0

---

## 🔍 Why GHAR Fails Catastrophically (R² = -33.47)

**Working Hypothesis:**

Our added projection layer changes the gradient flow in a way that causes instability:

**Original (if it works):**
```python
H1 = Linear(x)     # (batch, N, 1)
H2 = GCN(x)        # (batch, N, 16) somehow becomes (batch, N, 1)?
res = H1 + H2      # If this works, both are (batch, N, 1)
```

**Ours (with projection):**
```python
H1 = Linear(x)           # (1, 1, 1)
H2 = Linear(GCN(x))     # (1, 1, 1) - extra linear layer
res = H1 + H2           # (1, 1, 1)
```

**Problem:** The extra linear layer adds parameters that can diverge during training, especially with:
- Small dataset (1,260 samples)
- High learning rate (1e-3)
- ReLU causing gradient sparsity

**Evidence:** 19/20 seeds fail with R² = -36.49 (constant predictions), suggesting the projection layer causes catastrophic forgetting or collapse.

---

## 📋 Recommended Next Steps

### **Option A: Remove Projection Layer (Match Original's Bug)**

**Rationale:** If original code works despite dimension mismatch, maybe PyTorch handles it

**Implementation:**
```python
class GHAR_NoProj(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H2 = self.gcn1(node_feat, adj)  # Don't project
        # Try to add dimensions that don't match
        res = H1 + H2  # Let PyTorch broadcast or fail
        res = self.relu(res)
        return res.squeeze(-1)
```

**Expected:** Might crash, or might work if PyTorch broadcasts

### **Option B: Remove GHAR from Ensemble**

**Rationale:** GHAR fundamentally broken for single-stock case

**Implementation:**
```python
MODELS_TO_TRAIN = ['HAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']  # Skip GHAR
```

**Expected:** Avoid worst performer, focus on models that work moderately

### **Option C: Divide Targets by Horizon**

**Rationale:** Match paper's target scaling

**Implementation:**
```python
# In train_vic_ensemble.py
y_train_full = y_train_full / HORIZON
y_val_full = y_val_full / HORIZON
X_test, y_test, test_dates = build_snapshots_for_period(...)
y_test = y_test / HORIZON
```

**Expected:** Different scale, might help with stability

### **Option D: Remove ReLU (Final Fix)**

**Rationale:** ReLU causes 50% seed failures with constant predictions

**Implementation:**
```python
# Remove self.relu and relu() calls from all models
return res.squeeze(-1)  # No activation
```

**Expected:** Should eliminate constant prediction mode, allow negative predictions

---

## 🎯 Most Likely Root Causes (In Order of Priority)

1. **ReLU activation** - Causes constant predictions for 50% of seeds
2. **Projection layer** - Adds instability to GHAR (19/20 seeds fail)
3. **Target scaling** - Different value ranges affect gradient flow
4. **GCN bug** - Matrix multiply issue (might be handled by PyTorch automatically)

**Recommendation:** Start with Option D (remove ReLU), then Option B (skip GHAR), then test Option C (scale targets).
