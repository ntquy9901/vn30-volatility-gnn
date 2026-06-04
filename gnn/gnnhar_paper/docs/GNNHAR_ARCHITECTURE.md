# GNN-HAR Architecture Documentation

## Reference
Paper: "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)

## Overview

GNN-HAR models combine HAR (Heterogeneous AutoRegressive) features with Graph Neural Networks to capture:
- **H1 pathway**: Stock-specific volatility dynamics (local HAR features)
- **H2 pathway**: Cross-stock spillover effects (graph-based neighbor aggregation)

The models use a residual design: `output = H1 + H2`, allowing the model to balance local and spillover information.

---

## Input/Output

### Input Features (HAR)
For each stock `i` at time `t`:
```
node_feat[i] = [rv_d[i,t], rv_w[i,t], rv_m[i,t]]
```
- `rv_d`: Daily realized volatility (lag 1)
- `rv_w`: Weekly realized volatility (lag 5)  
- `rv_m`: Monthly realized volatility (lag 22)

**Shape:** `(batch_size, N=30, 3)` for 30 VN30 stocks

### Graph Structure
- **Nodes:** 30 VN30 stocks
- **Edges:** GLASSO-constructed correlations from historical returns
- **Adjacency:** `(N, N)` symmetric matrix, diagonal=0 (no self-loops)

### Output
Predicted RV (z-scored residuals) for each stock:
```
output[i] = predicted_rv_residual[i]
```

**Shape:** `(batch_size, N=30)`

---

## Model Variants

### 1. HAR (Baseline)

**Purpose:** Linear baseline without graph information

**Architecture:**
```
Input: (batch, 30, 3)
    |
    v
Linear(3 -> 1)        # H1: HAR prediction
    |
    v
ReLU                  # Ensure non-negative
    |
    v
Output: (batch, 30)
```

**Code Reference:** `gnnhar_models.py` Lines 41-76
```python
class HAR(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(3, 1, bias=True)  # 4 params
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)  # (batch, N, 1)
        res = self.relu(H1)            # (batch, N)
        return res.squeeze(-1)
```

**Parameters:** 4 (3 weights + 1 bias)

---

### 2. GHAR (Graph-augmented HAR)

**Purpose:** Test if linear graph spillover helps

**Architecture:**
```
Input: (batch, 30, 3)
    |
    +-------------------+
    |                   |
    v                   v
Linear(3 -> 1)    GCN(3 -> n_hid)       # H1: local    H2: spillover
    |                   |
    v                   v
ReLU              Projection(n_hid -> 1)  # Bug fix from paper
    |                   |
    +-------+-----------+
            |
            v
        H1 + H2                         # Residual sum
            |
            v
Output: (batch, 30)
```

**Code Reference:** `gnnhar_models.py` Lines 79-137
```python
class GHAR(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1, bias=True)     # H1 branch
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.proj = nn.Linear(n_hid, 1, bias=False)   # Dimension fix
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)    # (batch, N, 1)
        H1 = self.relu(H1)              # FIX: Activate BEFORE residual
        
        H2 = self.gcn1(node_feat, adj)  # (batch, N, n_hid)
        H2 = self.proj(H2)              # (batch, N, 1)
        
        res = H1 + H2                   # Residual sum
        return res.squeeze(-1)
```

**GCN Layer Detail:** `gcn_layer.py` Lines 61-93
```python
# GraphConvLayer.forward:
# Step 1: Linear transform
h = torch.matmul(node_feature, self.weight)  # X @ W
# Step 2: Message passing
output = torch.matmul(adj, h)                 # A @ (X @ W)
# Step 3: Add bias
if self.bias is not None:
    output = output + self.bias
```

**Parameters:** 4 + 48 + 16 = 68 (for n_hid=16)

---

### 3. GNNHAR1L (1-Hop with MLP)

**Purpose:** Capture 1-hop neighbor spillover with nonlinearity

**Architecture:**
```
Input: (batch, 30, 3)
    |
    +-------------------+
    |                   |
    v                   v
Linear(3 -> 1)    GCN(3 -> n_hid)       # H1: local    H2: 1-hop neighbors
    |                   |
    v                   v
ReLU              ReLU                     # Activate both paths
    |                   |
    v              MLP(n_hid -> 1)
    |                   |
    |                   v
    |              ReLU                    # Activate after projection
    |                   |
    +-------+-----------+
            |
            v
        H1 + H2                         # Residual sum
            |
            v
Output: (batch, 30)
```

**Code Reference:** `gnnhar_models.py` Lines 140-195
```python
class GNNHAR1L(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1, bias=True)     # H1 branch
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)   # MLP projection
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        # H1 branch
        H1 = self.linear1(node_feat)
        H1 = self.relu(H1)              # FIX: Activate BEFORE residual
        
        # H2 branch (1-hop)
        H2 = self.gcn1(node_feat, adj)  # Aggregate neighbors
        H2 = self.relu(H2)              # Nonlinearity
        H2 = self.mlp1(H2)              # Project to scalar
        H2 = self.relu(H2)              # FIX: Activate BEFORE residual
        
        res = H1 + H2                   # Residual sum (no activation after)
        return res.squeeze(-1)
```

**Parameters:** 4 + 48 + 16 = 68 (for n_hid=16)

---

### 4. GNNHAR2L (2-Hop)

**Purpose:** Capture 2-hop neighbor effects (neighbors of neighbors)

**Architecture:**
```
Input: (batch, 30, 3)
    |
    +-------------------+
    |                   |
    v                   v
Linear(3 -> 1)    GCN(3 -> n_hid)       # 1-hop aggregation
    |                   |
    v                   v
ReLU              ReLU
    |                   |
    |              GCN(n_hid -> n_hid)     # 2-hop aggregation
    |                   |
    |                   v
    |              ReLU
    |                   |
    |              MLP(n_hid -> 1)
    |                   |
    |                   v
    |              ReLU
    |                   |
    +-------+-----------+
            |
            v
        H1 + H2
            |
            v
Output: (batch, 30)
```

**Code Reference:** `gnnhar_models.py` Lines 198-259
```python
class GNNHAR2L(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.gcn2 = GraphConvLayer(n_hid, n_hid, bias=False)  # 2nd GCN layer
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H1 = self.relu(H1)
        
        # 2-hop propagation
        H2 = self.relu(self.gcn1(node_feat, adj))  # 1-hop
        H2 = self.relu(self.gcn2(H2, adj))         # 2-hop
        
        H2 = self.mlp1(H2)
        H2 = self.relu(H2)
        
        res = H1 + H2
        return res.squeeze(-1)
```

**Parameters:** 4 + 48 + 256 + 16 = 324 (for n_hid=16)

---

### 5. GNNHAR3L (3-Hop)

**Purpose:** Capture 3-hop neighbor effects (maximum range in VN30 graph)

**Architecture:**
```
Input: (batch, 30, 3)
    |
    +-------------------+
    |                   |
    v                   v
Linear(3 -> 1)    GCN(3 -> n_hid)       # Layer 1: 1-hop
    |                   |
    v                   v
ReLU              ReLU
    |                   |
    |              GCN(n_hid -> n_hid)    # Layer 2: 2-hop
    |                   |
    |                   v
    |              ReLU
    |                   |
    |              GCN(n_hid -> n_hid)    # Layer 3: 3-hop
    |                   |
    |                   v
    |              ReLU
    |                   |
    |              MLP(n_hid -> 1)
    |                   |
    |                   v
    |              ReLU
    |                   |
    +-------+-----------+
            |
            v
        H1 + H2
            |
            v
Output: (batch, 30)
```

**Code Reference:** `gnnhar_models.py` Lines 262-325
```python
class GNNHAR3L(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.gcn2 = GraphConvLayer(n_hid, n_hid, bias=False)
        self.gcn3 = GraphConvLayer(n_hid, n_hid, bias=False)  # 3rd GCN layer
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H1 = self.relu(H1)
        
        # 3-hop propagation
        H2 = self.relu(self.gcn1(node_feat, adj))  # 1-hop
        H2 = self.relu(self.gcn2(H2, adj))         # 2-hop
        H2 = self.relu(self.gcn3(H2, adj))         # 3-hop
        
        H2 = self.mlp1(H2)
        H2 = self.relu(H2)
        
        res = H1 + H2
        return res.squeeze(-1)
```

**Parameters:** 4 + 48 + 256 + 256 + 16 = 580 (for n_hid=16)

---

## Comparison Table

| Model | H1 Path | H2 Path | GCN Layers | Hop Range | Parameters |
|-------|---------|---------|------------|-----------|------------|
| HAR | Linear(3→1) + ReLU | None | 0 | Local only | 4 |
| GHAR | Linear(3→1) + ReLU | GCN + Linear | 1 | 1-hop | 68 |
| GNNHAR1L | Linear(3→1) + ReLU | GCN + ReLU + MLP + ReLU | 1 | 1-hop | 68 |
| GNNHAR2L | Linear(3→1) + ReLU | 2×GCN + ReLU + MLP + ReLU | 2 | 2-hop | 324 |
| GNNHAR3L | Linear(3→1) + ReLU | 3×GCN + ReLU + MLP + ReLU | 3 | 3-hop | 580 |

---

## Key Design Decisions

### 1. Residual Connection (H1 + H2)
**Why:** Allows model to balance local vs spillover information
- If graph is noisy: H2 ≈ 0, model behaves like HAR
- If graph is informative: H2 adds useful signal

### 2. Activation BEFORE Residual (Issue #3 Fix)
**Before (wrong):** `res = H1 + H2; res = ReLU(res)`
**After (correct):** `H1 = ReLU(H1); H2 = ReLU(H2); res = H1 + H2`

**Why:** Prevents "dying ReLU"
- If H1 + H2 < 0, ReLU zeros out entire sum
- Activating first ensures both paths contribute positively

### 3. No Self-Loops in Graph
**Why:** H1 already captures self-information (local HAR features)
- Self-loops would duplicate H1's role
- GLASSO adjacency has diagonal = 0

### 4. Bias=False in GCN
**Why:** Reduce parameters, prevent overfitting
- Bias in Linear(3→1) already provides baseline
- GCN focuses on neighbor aggregation, not adding bias

---

## Training Configuration

```python
# From train_gnnhar_paper.py
N_HID = 16           # Hidden dimension for GCN layers
N_EPOCHS = 5000      # Maximum training epochs
PATIENCE = 500        # Early stopping patience
LR = 1e-3            # Learning rate
WEIGHT_DECAY = 1e-3  # L2 regularization
NUM_MODELS = 10       # Ensemble size
```

---

## Data Flow Example

For a single snapshot (batch_size=1, N=30 stocks):

```
Input:
  node_feat: (1, 30, 3)  # 30 stocks × 3 HAR features
  adj: (30, 30)          # GLASSO correlation graph

GNNHAR1L Forward Pass:

1. H1 branch:
   H1 = Linear(3→1)(node_feat)  # (1, 30, 3) → (1, 30, 1)
   H1 = ReLU(H1)                 # (1, 30, 1) → (1, 30, 1)

2. H2 branch:
   H2 = GCN(node_feat, adj)     # (1, 30, 3) → (1, 30, 16)
   H2 = ReLU(H2)                 # (1, 30, 16) → (1, 30, 16)
   H2 = MLP(16→1)(H2)            # (1, 30, 16) → (1, 30, 1)
   H2 = ReLU(H2)                 # (1, 30, 1) → (1, 30, 1)

3. Residual sum:
   res = H1 + H2                 # (1, 30, 1) + (1, 30, 1) → (1, 30, 1)
   output = res.squeeze(-1)      # (1, 30, 1) → (1, 30)

Result:
  output: (1, 30)  # Predicted RV for 30 stocks
```

---

## Graph Construction (GLASSO)

**File:** `glasso_adjacency.py`

**Process:**
1. Collect stock returns data (training period only)
2. Apply Graphical LASSO to estimate sparse precision matrix
3. Convert precision matrix to adjacency (partial correlations)
4. Normalize: each row sums to 1
5. Set diagonal to 0 (no self-loops)

**Key parameters:**
- `alpha_range=(0.01, 1.0)`: Controls sparsity
- Higher alpha → sparser graph (fewer edges)

---

## Bug Fixes Applied

### Issue #3: Activation After Residual
**Fixed in:** All 5 models (HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L)

### GHAR Dimension Fix
**Problem:** Paper's GHAR has mismatch (H1: N×1, H2: N×n_hid)
**Solution:** Added projection layer `n_hid → 1` for proper residual

---

## References

1. Paper: "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)
2. GCN: Kipf & Welling (2017) "Semi-Supervised Classification with Graph Convolutional Networks"
3. HAR: Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"
