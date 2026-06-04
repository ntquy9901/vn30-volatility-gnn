# GNN-HAR Architecture Diagrams for Thesis

## Complete Architecture (All Models)

```
                    INPUT LAYER
              ┌─────────────────┐
              │  HAR Features   │
              │  (batch, 30, 3) │
              │  rv_d, rv_w,    │
              │  rv_m per stock │
              └────────┬────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────┐           ┌─────────────────┐
│   H1 PATHWAY  │           │   H2 PATHWAY    │
│  (Local HAR)  │           │  (Graph Spillover)│
└───────────────┘           └─────────────────┘
        │                             │
        ▼                             ▼
┌───────────────┐           ┌─────────────────┐
│ Linear(3 → 1) │           │   GCN Layers     │
│    + ReLU     │           │   (1, 2, or 3)   │
└───────┬───────┘           └──────┬──────────┘
        │                          │
        │              ┌───────────┴───────────┐
        │              │                       │
        │              ▼                       ▼
        │      ┌──────────────┐      ┌──────────────┐
        │      │ GCN1:        │      │ GHAR:        │
        │      │ 3 → n_hid    │      │ Linear only   │
        │      │ + ReLU       │      │ (no ReLU)    │
        │      └──────┬───────┘      └──────┬───────┘
        │             │                     │
        │             ▼                     ▼
        │      ┌──────────────┐      ┌──────────────┐
        │      │ GNNHAR1L:    │      │  Projection   │
        │      │ MLP(n_hid→1) │      │  (n_hid→1)   │
        │      │ + ReLU       │      └──────┬───────┘
        │      └──────┬───────┘             │
        │             │                     │
        │             ▼                     │
        │      ┌──────────────┐             │
        │      │ GNNHAR2L:    │             │
        │      │ GCN2 + ReLU  │             │
        │      └──────┬───────┘             │
        │             │                     │
        │             ▼                     │
        │      ┌──────────────┐             │
        │      │ GNNHAR3L:    │             │
        │      │ GCN3 + ReLU  │             │
        │      └──────┬───────┘             │
        │             │                     │
        │             ▼                     │
        │      ┌──────────────┐             │
        │      │ MLP(n_hid→1) │             │
        │      │ + ReLU       │             │
        │      └──────┬───────┘             │
        │             │                     │
        └──────┬──────┴─────────────────────┘
               │
               ▼
        ┌─────────────┐
        │ RESIDUAL    │
        │   H1 + H2   │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  OUTPUT     │
        │ (batch, 30) │
        │ Predicted   │
        │ RV per stock│
        └─────────────┘
```

---

## Model Comparison (Side-by-Side)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL VARIANTS                                  │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│    HAR      │    GHAR     │  GNNHAR1L   │  GNNHAR2L   │    GNNHAR3L     │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│             │             │             │             │                 │
│  Input      │  Input      │  Input      │  Input      │  Input          │
│ (30, 3)     │ (30, 3)     │ (30, 3)     │ (30, 3)     │ (30, 3)         │
│             │             │             │             │                 │
│  Linear     │  Linear     │  Linear     │  Linear     │  Linear         │
│  3→1        │  3→1        │  3→1        │  3→1        │  3→1            │
│  + ReLU     │  + ReLU     │  + ReLU     │  + ReLU     │  + ReLU         │
│    │        │    │        │    │        │    │        │    │            │
│    │        │    │        │    │        │    │        │    │            │
│    │        │    │        │    │        │    │        │    │            │
│    └──┬─────┘    │        │    └──┬─────┘    │        │    └──┬─────────┤
│       │         │        │       │         │        │       │         │
│       ▼         │        ▼       ▼         ▼        ▼       ▼         ▼
│    [H1]        │        │   GCN1:3→16   GCN1:3→16  GCN1:3→16        │
│       │         │        │   + ReLU       + ReLU     + ReLU           │
│       │         │        │       │         │        │       │         │
│       │         │        │       │    GCN2:16→16  GCN2:16→16        │
│       │         │        │       │    + ReLU     + ReLU              │
│       │         │        │       │         │        │       │         │
│       │         │        │       │         │        │  GCN3:16→16     │
│       │         │        │       │         │        │  + ReLU         │
│       │         │        │       │         │        │       │         │
│       │         │        ▼       ▼         ▼        ▼       ▼         ▼
│       │         │    [H2: Linear]  MLP:16→1   MLP:16→1  MLP:16→1       │
│       │         │         │       + ReLU     + ReLU    + ReLU           │
│       │         │         │         │         │        │               │
│       └────┬────┴─────────┴─────────┴─────────┴────────┴───────┐     │
│              ▼                                                   ▼     │
│           H1 + H2 = OUTPUT(30)                                 OUTPUT │
│                                                               (30)    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow (Single Snapshot)

```
TIMESTEP t:

Stock Features (before aggregation):

┌──────┬──────────┬──────────┬──────────┐
│Stock │   rv_d   │   rv_w   │   rv_m   │
├──────┼──────────┼──────────┼──────────┤
│ VCB  │  0.0012  │  0.0015  │  0.0018  │
│ VIC  │  0.0008  │  0.0011  │  0.0014  │
│ VNM  │  0.0015  │  0.0013  │  0.0016  │
│ ...  │   ...    │   ...    │   ...    │
└──────┴──────────┴──────────┴──────────┘
          │
          ▼
    H1: Linear + ReLU
    [Local prediction per stock]
          │
          ├─────────────────────────────────┐
          │                                 │
          ▼                                 ▼
    H1[i] = ReLU(w1*rv_d + w2*rv_w + w3*rv_m + b)
          │                                 │
          │                    Graph Adjacency (GLASSO):
          │                    ┌─────┬─────┬─────┬───┐
          │                    │     │ 0.3 │     │   │ VCB neighbors
          │                    ├─────┼─────┼─────┼───┤
          │                    │ 0.3 │     │ 0.1 │   │
          │                    ├─────┼─────┼─────┼───┤
          │                    │     │ 0.1 │     │   │
          │                    └─────┴─────┴─────┴───┘
          │                                 │
          │                    Message Passing (GCN):
          │                    H2[i] = Σ A[i,j] * (X[j] @ W)
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
                    OUTPUT = H1 + H2
                         │
                         ▼
              Predicted RV[i] (z-scored)
```

---

## Layer-wise Transformations (GNNHAR1L)

```
LAYER 0: Input
Shape: (batch=100, N=30, features=3)
Value: [rv_d, rv_w, rv_m] for each stock


LAYER 1a: H1 Linear Transform
H1_0 = Linear(3→1)(X)
Shape: (100, 30, 1)
Operation: For each stock i: h1[i] = w1*rv_d + w2*rv_w + w3*rv_m + b


LAYER 1b: H1 Activation
H1 = ReLU(H1_0)
Shape: (100, 30, 1)
Operation: h1[i] = max(0, h1_0[i])


LAYER 2a: H2 GCN (1-hop)
H2_0 = GCN(X, A)
Shape: (100, 30, 16)
Operation: h2_0[i] = Σ_j A[i,j] * (X[j] @ W_gcn)


LAYER 2b: H2 Activation 1
H2_1 = ReLU(H2_0)
Shape: (100, 30, 16)
Operation: h2_1[i] = max(0, h2_0[i])


LAYER 3: H2 Projection
H2_2 = MLP(16→1)(H2_1)
Shape: (100, 30, 1)
Operation: h2_2[i] = h2_1[i] @ W_mlp


LAYER 4: H2 Activation 2
H2 = ReLU(H2_2)
Shape: (100, 30, 1)
Operation: h2[i] = max(0, h2_2[i])


LAYER 5: Residual Sum
output = H1 + H2
Shape: (100, 30)
Operation: output[i] = h1[i] + h2[i]
```

---

## Parameter Count (n_hid=16)

```
┌────────────────────────────────────────────────────────────┐
│                    PARAMETER COUNTS                       │
├──────────────┬─────────────┬─────────────┬────────────────┤
│    Model     │   H1 Path   │   H2 Path   │    Total       │
├──────────────┼─────────────┼─────────────┼────────────────┤
│ HAR          │ 4 (3+1)     │ 0           │ 4              │
│ GHAR         │ 4           │ 64 (48+16)  │ 68             │
│ GNNHAR1L     │ 4           │ 64 (48+16)  │ 68             │
│ GNNHAR2L     │ 4           │ 320 (48+256+16) │ 324      │
│ GNNHAR3L     │ 4           │ 576 (48+256+256+16) │ 580  │
└──────────────┴─────────────┴─────────────┴────────────────┘

Breakdown:
- Linear(3→1): 3 weights + 1 bias = 4
- GCN(in→out): in × out weights (bias=False)
- MLP(n_hid→1): n_hid weights (bias=False)
```

---

## Training Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

DATA PREPARATION:
│
├─ Load VN30 prices (30 stocks, ~2500 days)
├─ Compute log returns
├─ Build HAR features: [rv_d, rv_w, rv_m]
├─ Compute RV targets (h=1,5,10,20)
├─ Z-score HAR features per stock
├─ Split: Train (80%) / Val (20%) / Test (from 2026-01-01)
│
└─ Verify temporal ordering (Issue #6 fix)


GRAPH CONSTRUCTION:
│
├─ Use training-period returns only
├─ Apply GLASSO (alpha_range=[0.01, 1.0])
├─ Estimate sparse precision matrix
├─ Convert to adjacency matrix (30×30)
├─ Normalize: row sums = 1
├─ Set diagonal = 0 (no self-loops)
│
└─ Result: Symmetric, sparse graph


FOR EACH HORIZON h IN [1, 5, 10, 20]:
│
├─ Build snapshots with stride_h
├─ FOR EACH MODEL IN [HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L]:
│  │
│  ├─ Initialize ensemble (NUM_MODELS=10)
│  │
│  ├─ FOR epoch IN 1..5000:
│  │  │
│  │  ├─ Forward pass: output = model(X_train, adj)
│  │  ├─ Compute loss: QLIKE(output, y_train)
│  │  ├─ Backward pass + optimizer step
│  │  ├─ Compute val loss
│  │  ├─ Early stopping check (patience=500)
│  │  └─ Save checkpoint if best
│  │
│  └─ Screen ensemble (keep top 50%)
│
└─ Save models + learning curves


EVALUATION:
│
├─ Load test data (stride=1, full evaluation)
├─ FOR EACH model:
│  ├─ Forward pass: predictions = model(X_test, adj)
│  ├─ Compute metrics: R², MAE, RMSE, QLIKE, HMSE, HMAE
│  ├─ Compare vs HAR baseline
│  └─ Per-stock analysis (30 stocks)
│
└─ Generate results + plots
```

---

## Citation for Thesis

```
@article{gnnhar2024,
  title={Forecasting Realized Volatility with Spillover Effects: 
         Perspectives from Graph Neural Networks},
  journal={International Journal of Forecasting},
  year={2024},
  note={(IJF 2024)}
}
```

---

## Figure Captions (Ready for Thesis)

**Figure 1: GNN-HAR Architecture Overview**
The GNN-HAR model combines HAR features with graph neural networks. 
H1 pathway captures stock-specific volatility dynamics through linear 
regression on HAR features. H2 pathway captures cross-stock spillover 
through GCN layers that aggregate information from correlated neighbors. 
The residual connection H1 + H2 allows flexible balancing of local 
and spillover information.

**Figure 2: Model Variants Comparison**
Five model variants from simple to complex: HAR (baseline without graph), 
GHAR (linear graph spillover), GNNHAR1L (1-hop with nonlinearity), 
GNNHAR2L (2-hop neighbors), GNNHAR3L (3-hop neighbors). Deeper GCN 
layers capture longer-range dependencies but risk over-smoothing.

**Figure 3: GCN Message Passing**
Each node aggregates information from its neighbors using the adjacency 
matrix. The adjacency is constructed via GLASSO from historical returns, 
encoding correlation structure. Node features are HAR features 
[rv_d, rv_w, rv_m] representing daily, weekly, monthly volatility.

**Figure 4: Training Pipeline**
Data is split temporally: train (80%), validation (20%), test (from 
2026-01-01). Graph construction uses training data only to prevent 
look-ahead bias. Ensemble of 10 models per variant trained with QLIKE 
loss, early stopping (patience=500).
