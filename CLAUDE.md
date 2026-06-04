# Claude Instructions — VN30 Volatility Forecasting (Moirai2 + GNN)

## User Context (always apply)

The user is a thesis student learning DL/GNN as they go — limited prior experience with
deep learning, graph neural networks, and financial ML. Always:
- Explain ML/DL concepts in code comments as if the reader is encountering them for the first time
- Write learning docs in Vietnamese for new techniques (see Code Comment Rules below)
- Prefer clear over clever: explicit variable names, no terse one-liners for complex math
- When asked to explain code behavior, explain the statistical/ML intuition, not just syntax

## MANDATORY: Read before any code generation or planning

Before generating any code, creating any plan, or making implementation decisions,
you MUST read and follow: `CONSTRAINTS.md`

Key rules summary (full details in CONSTRAINTS.md):
- R1: MIMO multi-horizon — HORIZONS=[1,5,10,20] configurable, 1 model, no recursive
- R2: Print per-horizon loss every epoch + save training curve PNG
- R3: Print data split (dates + counts + ESS) before training starts
- R4: LSTM uses stride=1 per-stock; GNN/MLP uses stride=MAX_H batch
- R5: CONSTRAINTS.md is the single source of truth for design decisions
- R6: Global data split — Test from 2026-01-01, Train/Val 80/20 from pre-2026. Console + learning curves for ALL models.

## Code Style Rules (mandatory)

- No emoji or Unicode icons anywhere — not in print(), comments, or docstrings. Causes UnicodeEncodeError on Windows cp1252.
- Use ASCII-only status indicators: `[OK]`, `[WARN]`, `[ERROR]`, `->`, `--`.
- Console output (print statements): short and direct. No multi-line explanations.

## File Naming Rules (mandatory)

- **Timestamps in output filenames:** All generated files (learning curves PNG, results JSON, model checkpoints) MUST include timestamps to prevent overwrites
- **Format:** `{prefix}_{description}_{YYYYMMDD_HHMMSS}.{ext}`
- **Examples:**
  - `GNNHAR1L_seed42_learning_curve_20260601_213000.png`
  - `GNNHAR1L_gelu_h5_20260601_213000.json`
- **Consistency:** All files from a single training run must share the same timestamp
- **Rationale:** Multiple training runs create separate files, easy to identify when results were generated, timestamps link learning curves to JSON results

## Code Comment Rules (mandatory)

The user is learning DL/GNN. Code comments must explain the ML/statistical reasoning, not just
re-state what Python does. Two levels of explanation are required:

**Level 1 — Inline comments in code:**
Add a comment on any line/block that is non-obvious from a ML perspective. Examples:
- Why z-score? Why clip to [0,inf)? Why DropEdge? Why HAR residual training?
- What does each tensor shape mean? (e.g., # (n_snapshots, n_stocks, n_features))
- Why a specific hyperparameter value? (e.g., # weight_decay=1e-3: strong prior, ESS=123)
- Mathematical formula in words where code alone is ambiguous
One or two lines per block is enough. Do NOT comment on obvious Python (loops, imports, etc.).

**Level 2 — Learning documentation:**
When writing NEW model code or implementing a technique for the first time (GNN layer, graph
construction, residual training, normalization scheme, etc.), ALWAYS create or update a
corresponding learning document in `docs/learning/`. Format must match existing files:
- Header: topic in Vietnamese, date, source question
- Sections: concept explanation, formulas, applied to this project, common pitfalls
- Language: Vietnamese (for user study), with English technical terms preserved
- File naming: next sequential number + short topic (e.g., `05_graphsage_gnn.md`)
- DO NOT create a doc for trivial changes (bug fixes, config tweaks, adding a print statement)
Trigger: any file creation or substantial rewrite in `gnn/`, `src/models/`, `baselines/`.

## Project Context

- **Task:** Thesis — VN30 realized volatility forecasting
- **Data:** 30 VN30 stocks, OHLCV daily, ~2,500 days (2014-2026)
- **Models:** HAR-RV (baseline), LSTM, GNN, MLP, Moirai2 (foundation model)
- **Label:** RV_h = std(log_return[t:t+h]) for h in [1, 5, 10, 20]

## Key Paths

- `data/raw/prices/` — OHLCV CSV per stock
- `src/` — model implementations
- `results/` — metrics, predictions, charts
- `docs/learning/` — learning notes (review material)
- `CONSTRAINTS.md` — design rules (read before coding)

## Effective Sample Size (ESS) — always reference when discussing data sufficiency

```
ESS = N_raw / max_horizon   (Lopez de Prado 2018)
Per-stock LSTM (h=20, stride=1): ESS = 2,458 / 20 = 123
Cross-stock pooled (x30):        ESS = 123 * 30 = 3,690
HAR-RV (3 params, ESS=125):     obs/param = 41 -> BLUE
LSTM (4000 params, ESS=123):    obs/param = 0.03 -> overfit
```

## Model Design Checklist (mandatory before writing any new model)

Learned from comparing broken PyTorch rewrite (R2=-100) vs working Keras model (R2=0.815), and from GNN+HAR review. These are recurring failure modes in this project.

**C1 — Activation functions: verify explicitly when porting**
Keras `Dense()` default = linear (no activation). PyTorch `nn.Linear` also linear, but easy
to add `nn.ReLU()` accidentally. Adding ReLU where none existed -> dying neurons -> R2=-100.
Before finalizing any model: list every layer and its activation. Match the reference exactly.

**C2 — Feature type must match target type**
For RV prediction: features must use the SAME measurement type as the target.
- Target = std(log_returns, h-day window) -> features must also use rolling std, NOT |log_return|
- Using single-day |r_t| as feature for h=20 RV target = high noise mismatch
- Use `compute_past_rv(log_returns, h)` for features, not `log_returns.abs()`
- Source of truth: `src/volatility_labels.py:compute_past_rv()`

**C3 — Loss function for RV: MSE on z-scored HAR residuals**
- Train loss: MSE on z-scored HAR residuals (zero-mean, unit-variance per stock)
- Evaluation metrics: R2, MAE, RMSE only
- Z-scoring is mandatory: without it, high-vol stocks (GAS, NVL) dominate the loss,
  and the model ignores low-vol stocks entirely. Z-score ensures equal learning signal
  per stock regardless of volatility level.

**C4 — Always include naive baseline in evaluation**
Every evaluation script must print BOTH model metrics AND naive baseline metrics side by side.
- For RV: naive = predict training mean (constant)
- For price: naive = y_t = y_{t-1}
- A model with R2=0.8 that is worse than naive (R2=0.92) has zero predictive skill.

**C5 — Dropout calibration: rate x hidden >= 10 active neurons minimum**
dropout=0.3 on hidden=16 -> only 11 active neurons. Too aggressive for small models.
Default: dropout=0.1 for hidden <= 32. Only use dropout=0.3 for hidden >= 64.

**C6 — Count total gradient updates before declaring model "trained"**
total_updates = (n_train_samples / batch_size) * epochs
If total_updates < 50,000 for a neural model -> likely undertrained.
stride=20 -> ~100 snapshots -> 100 updates/epoch. stride=1 -> 2000 updates/epoch.
