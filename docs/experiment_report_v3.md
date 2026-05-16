# Experiment Report v3 — VN30 Volatility Forecasting
**Date:** 2026-05-16  
**Model stack:** Moirai2-small (frozen) + GNN/MLP head, DGL backend  
**Test set:** 2026-01-05 → 2026-04-07 (62 dates, 30 VN30 stocks, pooled)

---

## Final Results Table

| Model | MAE | RMSE | R² | QLIKE | Pearson_r | Notes |
|---|---|---|---|---|---|---|
| **Batch GNN** | **0.0074** | 0.0094 | +0.041 | 0.399 | 0.357 | H2 experiment |
| **Batch MLP** | 0.0078 | 0.0102 | -0.137 | 0.290 | 0.425 | H2 experiment |
| WalkFwd MLP+Moirai2 | 0.0121 | 0.0138 | -1.077 | 1.444 | 0.239 | last_context pooling |
| WalkFwd GNN+Moirai2 | 0.0143 | 0.0162 | -1.856 | 1.898 | 0.117 | last_context pooling |
| GARCH(1,1) | 0.0071 | 0.0088 | +0.159 | 0.301 | 0.535 | per-stock OLS |
| **HAR-RV (OLS)** | **0.0010** | 0.0016 | **+0.971** | 0.012 | **0.985** | benchmark |
| LSTM | 0.0011 | 0.0017 | +0.967 | 0.012 | 0.983 | benchmark |

---

## Experiment Timeline

### v1 — Baseline (raw RV scale, mean pooling)
- GNN MAE = 0.0153, model collapse (std = 0.000003, constant prediction)
- Root cause: MSE on raw RV scale → model predicts mean

### v2 — Log-RV normalization + mean pooling
- GNN MAE = 0.0117 (−23% vs v1), graph contribution +21.9%
- MLP MAE = 0.0148

### v3 — Hybrid lagged-RV features (384→387 dim) + log-RV
- GNN MAE = 0.0143, MLP MAE = 0.0142
- Graph contribution: −0.6% (GNN ≈ MLP, adding RV features diluted graph benefit)

### v4 — last_context pooling fix (+23.5% embedding RV correlation)
- Confirmed: Moirai2 is decoder-only with causal attention (14 patches: 13 context + 1 MASK)
- last_context (index −2) = last observed patch, richest single-vector representation
- MLP MAE = 0.0121 (−14.8%), GNN MAE = 0.0143 (no improvement — graph averages out gain)

### v5 — Batch training (H2 hypothesis test)
- 3,619 samples from 135 windows cached; trained with mini-batch SGD
- **Batch GNN MAE = 0.0074** (+48.4% vs walk-forward)
- **Batch MLP MAE = 0.0078** (+35.1% vs walk-forward)
- Graph now helps: Batch GNN < Batch MLP (sufficient data enables message-passing)

---

## Root Cause Analysis (D1–D4 Diagnostics)

### D1 — Scatter plot
Walk-forward GNN/MLP predictions form a **near-horizontal cloud** vs true RV → systematic underestimation of variance, not random noise.

### D2 — Moirai2 Embedding–RV Correlation Audit
- Median |corr(dim_i, RV)| = **0.128** (with last_context pooling; was 0.103 with mean pooling)
- 91/384 dims have |corr| > 0.20 (23.7%)
- H1 **partially confirmed**: embeddings carry diluted but non-zero RV signal

### D3 — Neural HAR-RV (3 features, walk-forward)
- Neural HAR MAE = 0.037 vs HAR-OLS MAE = 0.001 — **37× gap** from same features
- Confirms: walk-forward training regime, not feature quality, drives the gap
- Moirai2 embeddings DO help: GNN/MLP (0.014) < Neural HAR (0.037) in same walk-forward regime

### D4 — Train loss trajectory
- Initial loss = 71.8 → Best = 0.023 (window 125/135), 100% improvement
- Model learns during training; the test gap is not a broken training loop

---

## Hypothesis Verdicts

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | Moirai2 embeddings carry no RV signal | **PARTIAL** — median \|corr\|=0.128, signal exists but diluted |
| H2 | Walk-forward training starves neural models | **CONFIRMED** — batch training cuts MAE 35–48% |
| H3 | Lagged RV features dominate, Moirai2 adds noise | **NOT confirmed** — Moirai2 helps even with walk-forward |

---

## Architecture Notes

### Moirai2 Architecture (confirmed)
- **Decoder-only** with causal temporal attention (HuggingFace model card)
- Single TransformerEncoder stack, `packed_causal_attention_mask`
- For context_length=200, prediction_length=1: **14 patches** (13 context + 1 MASK token)
- `last_context` pooling = `reprs[:, -2, :]` — last observed patch before MASK

### GNN (DGL backend)
- 2-layer SAGEConv (384+3 → 64 → 32) + MLP head (32 → 16 → 1)
- Graph: Pearson corr > 0.4 OR same sector → edge; VNINDEX as hub node (31 total)
- DGL 1.1.2 (avoids torchdata/graphbolt dependency conflicts)

### Remaining gap vs HAR-RV
Batch GNN/MLP are 7–8× worse than HAR-RV (MAE 0.0074 vs 0.0010).  
Attributable to **feature quality**: HAR-RV uses direct past-RV (near-perfect autocorrelation);  
Moirai2 embeddings encode return-level dynamics, not second-moment structure.

---

## Files

| File | Purpose |
|---|---|
| `run_gnn_train.py` | Walk-forward GNN training |
| `run_baselines.py` | GARCH, HAR-RV, LSTM, MLP baselines |
| `run_evaluation.py` | Full model comparison |
| `run_batch_train.py` | Batch training experiment (H2) |
| `run_plots.py` | Bar chart, R² heatmap, time series |
| `diag_d1_d4.py` | Scatter plots + train loss trajectory |
| `diag_d2_embed_corr.py` | Embedding–RV correlation audit |
| `diag_d3_neural_har.py` | Neural HAR-RV ablation |
| `diag_pooling_compare.py` | Pooling strategy benchmark |
| `results/embed_cache/` | 135 cached window embeddings (387-dim) |
