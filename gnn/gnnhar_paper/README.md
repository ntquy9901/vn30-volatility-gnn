# GNNHAR Paper Replication

Full replication of GNNHAR paper: "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks" (IJF 2024)

## Files

| File | Description |
|------|-------------|
| `gcn_layer.py` | GCN (Graph Convolutional Network) layer implementation |
| `gnnhar_models.py` | 5 model classes: HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L |
| `glasso_adjacency.py` | GLASSO adjacency matrix construction |
| `rolling_datasets.py` | Rolling window dataset builder |
| `ensemble_trainer.py` | Ensemble training with screening |
| `train_gnnhar_paper.py` | Main training script |

## Quick Start

```bash
cd moirai

# Full training (all horizons, all models, 5 ensemble each)
python gnn/gnnhar_paper/train_gnnhar_paper.py

# Expected time: 30-60 minutes (CPU only)
```

## Quick Test

Edit `train_gnnhar_paper.py` to reduce training:

```python
# Line ~60
HORIZONS = [1]              # Only h=1
MODEL_NAMES = ['GNNHAR1L']  # Only one model
NUM_MODELS = 2              # Reduce ensemble
N_EPOCHS = 100             # Reduce epochs
```

## Output

```
models/gnnhar_paper/
├── h1/, h5/, h10/, h20/
│   └── HAR/, GHAR/, GNNHAR1L/, GNNHAR2L/, GNNHAR3L/
│       └── *.pt  (ensemble models)

results/gnnhar_paper/
├── gnnhar_paper_results.csv  # Per-stock results
├── curve_*.png                # Learning curves
└── summary                    # Console summary
```

## Architecture

```
Input: (30, 3) HAR features [rv_d, rv_w, rv_m]

Models:
  HAR       Linear(3,1) -> ReLU -> Output
  GHAR      Linear + 1×GCN (linear spillover)
  GNNHAR1L  Linear + 1×GCN + MLP (1-hop)
  GNNHAR2L  Linear + 2×GCN + MLP (2-hop)
  GNNHAR3L  Linear + 3×GCN + MLP (3-hop)
```

## Reference

Original paper code: `GNNHAR/GNNHAR.py`
