# Plan: LSTM SISO Ablation (h=1, h=5, h=10)

## Context

CONSTRAINTS.md R1 exception approved 2026-05-18.

Learning curve analysis across 30 stocks revealed: h=10 and h=20 validation loss
diverges early and triggers early stopping before h=1 and h=5 converge.
Hypothesis: MIMO gradient interference is causing h=5 underperformance.

This experiment tests that hypothesis with a controlled ablation.

---

## Research Question

> "Is LSTM-HAR h=5 failure (-0.199 median R2) caused by MIMO gradient interference
>  from h=10/h=20 objectives sharing the same backbone, or is it an ESS bottleneck
>  that no architecture change can fix?"

| Outcome | Conclusion |
|---|---|
| SISO h=5 >> MIMO h=5 (e.g. median R2 > 0) | MIMO interference is the primary cause |
| SISO h=5 ~= MIMO h=5 (both still negative) | ESS bottleneck confirmed — irrelevant of architecture |

Both outcomes are thesis-relevant.

---

## Controlled Ablation Design

| Dimension | LSTM-HAR MIMO (existing) | LSTM SISO (this plan) |
|---|---|---|
| Input features | [rv_d, rv_w, rv_m] | **same** |
| LOOKBACK | 20 | **same** |
| HIDDEN | 64 | **same** |
| DROPOUT | 0.2 | **same** |
| LR, BATCH, EPOCHS, PATIENCE | same | **same** |
| Data split (R6) | same | **same** |
| HORIZONS | [1, 5, 10, 20] | **[H] only** |
| n_heads | 4 | **1** |
| Loss function | sum of 4 MSE losses | **MSE on H only** |
| ESS printed as | N_train // 20 | **N_train // H** |

Only change: number of output heads and loss objective.

---

## Files

| File | Purpose |
|---|---|
| `baselines/train_lstm_siso.py` | Parametric training: `--horizon H` (H in {1, 5, 10}) |
| `scripts/eda/compare_siso_vs_mimo.py` | 3-way comparison: SISO, MIMO, HAR-RV |

No separate evaluate script — training script includes HAR-RV evaluation inline.

---

## Output Paths (per horizon H)

| Output | Path |
|---|---|
| Results CSV | `results/lstm_siso_h{H}_results.csv` |
| Learning curves | `results/lstm_siso_h{H}_curves/{ticker}_curve.png` |
| Trained models | `models/lstm_siso_h{H}/{ticker}_model.pt` |
| Comparison CSV | `results/siso_vs_mimo_comparison.csv` |
| Comparison plots | `results/siso_vs_mimo_scatter.png`, `results/siso_vs_mimo_bar.png` |

---

## CSV Column Schema

`lstm_siso_h{H}_results.csv`:

| Column | Description |
|---|---|
| ticker | Stock symbol |
| n_train, n_val | Sample counts |
| ess_h{H} | N_train // H (higher than MIMO's N_train // 20) |
| best_epoch | Early stopping epoch |
| lstm_h{H}_r2, _mae, _rmse, _da | SISO test metrics |
| har_h{H}_r2, _mae, _rmse, _da | HAR-RV test metrics (inline) |

---

## Run Order

```bash
# Train all 3 horizons (can run sequentially)
python baselines/train_lstm_siso.py --horizon 1 --all
python baselines/train_lstm_siso.py --horizon 5 --all
python baselines/train_lstm_siso.py --horizon 10 --all

# Compare (requires lstm_har_results.csv from MIMO evaluate)
python scripts/eda/compare_siso_vs_mimo.py
```

---

## Expected Baseline for Interpretation

From current LSTM-HAR MIMO results (median R2):

| Horizon | MIMO median R2 | HAR-RV avg R2 |
|---|---|---|
| h=1  | -0.050 | ~-0.04 |
| h=5  | -0.199 | +0.598 |
| h=10 | -0.371 | +0.820 |

SISO threshold for "interference confirmed": median R2 > 0 at h=5 or h=10.

---

## Thesis Paragraph (template, fill after results)

If SISO improves significantly:
> "Separating the h=5 objective from h=10/h=20 gradients (SISO vs MIMO ablation)
>  improves h=5 median R2 from -0.199 to +X.XX, confirming that MIMO gradient
>  interference is a significant source of performance degradation. Nevertheless,
>  the remaining gap to HAR-RV (+0.598) reflects the fundamental ESS bottleneck."

If SISO similar to MIMO:
> "Training a dedicated SISO model for h=5 with identical features and hyperparameters
>  yields median R2 = -X.XX, statistically indistinguishable from the MIMO result
>  (-0.199). This rules out MIMO interference as the primary cause and confirms
>  that the ESS bottleneck (obs/param = 0.02-0.05) is irreducible per-stock."

---

*Created: 2026-05-18 | Status: IMPLEMENTED (scripts created, not yet run)*
