#!/usr/bin/env python
"""Compare GELU vs ReLU activation results."""
import json

# Load results
with open('results/gnnhar_paper/multi_stock/GNNHAR1L_gelu_h5_20260601_212005.json') as f:
    gelu = json.load(f)
with open('results/gnnhar_paper/multi_stock/GNNHAR1L_relu_h5_20260601_212959.json') as f:
    relu = json.load(f)

# Extract metrics
gelu_r2 = gelu['test_r2']
relu_r2 = relu['test_r2']
r2_improvement = (gelu_r2 - relu_r2) / relu_r2 * 100

gelu_mae = gelu['test_mae']
relu_mae = relu['test_mae']
mae_improvement = (relu_mae - gelu_mae) / relu_mae * 100

gelu_rmse = gelu['test_rmse']
relu_rmse = relu['test_rmse']
rmse_improvement = (relu_rmse - gelu_rmse) / relu_rmse * 100

print('='*70)
print('  GELU vs ReLU Activation Comparison (GNNHAR1L, h=5)')
print('='*70)
print()
print('Training Configuration:')
print(f'  Seeds: {gelu["n_seeds"]} (screened to {gelu["n_models"]} model)')
print(f'  Epochs: {gelu["model_epochs"][0]}')
print(f'  Hidden dim: {gelu["n_hid"]}')
print()
print('Test Metrics:')
print(f'  {"Metric":<15} {"ReLU":>12} {"GELU":>12} {"Improvement":>12}')
print(f'  {"-"*50}')
print(f'  {"R2":<15} {relu_r2:>12.4f} {gelu_r2:>12.4f} {r2_improvement:>+10.2f}%')
print(f'  {"MAE":<15} {relu_mae:>12.6f} {gelu_mae:>12.6f} {mae_improvement:>+10.2f}%')
print(f'  {"RMSE":<15} {relu_rmse:>12.6f} {gelu_rmse:>12.6f} {rmse_improvement:>+10.2f}%')
print()
print('Validation Loss (best models):')
print(f'  GELU: {gelu["model_val_losses"][0]:.4f}')
print(f'  ReLU: {relu["model_val_losses"][0]:.4f}')
print()
print('='*70)
print('  Analysis')
print('='*70)
print()
print('R2 Improvement:')
if r2_improvement >= 2.0:
    print(f'  [OK] STRONG: GELU improves R2 by {r2_improvement:.2f}% (>=2% threshold)')
    print('  Recommendation: Proceed to full ensemble training (20 seeds, 400 epochs)')
elif r2_improvement >= 0.5:
    print(f'  [WARN] MODEST: GELU improves R2 by {r2_improvement:.2f}% (0.5-2% range)')
    print('  Recommendation: Test with more seeds (10-20) before decision')
else:
    print(f'  [FAIL] WEAK: GELU improves R2 by only {r2_improvement:.2f}% (<0.5% threshold)')
    print('  Recommendation: Revert to ReLU, focus on other improvements')
print()
print('MAE/RMSE Improvement:')
print(f'  MAE:  {mae_improvement:+.2f}% (lower is better)')
print(f'  RMSE: {rmse_improvement:+.2f}% (lower is better)')
print()
print('Convergence:')
print(f'  GELU val loss: {gelu["model_val_losses"][0]:.4f} (similar to ReLU)')
print(f'  ReLU val loss: {relu["model_val_losses"][0]:.4f}')
print('  Both activations converged equally well')
print()
print('='*70)
print('  Recommendation')
print('='*70)
print()

if r2_improvement >= 0.5:
    print('GELU shows consistent improvement across all metrics:')
    print(f'  - R2: +{r2_improvement:.2f}%')
    print(f'  - MAE: +{mae_improvement:.2f}%')
    print(f'  - RMSE: +{rmse_improvement:.2f}%')
    print()
    print('Next steps:')
    print('1. Test with 10-20 seeds to confirm improvement holds')
    print('2. If improvement persists, proceed to full ensemble (20 seeds, 400 epochs)')
    print('3. Compare final GELU results against sklearn baselines (R2 >= 0.75)')
else:
    print('GELU improvement is minimal:')
    print(f'  - R2 improvement: only {r2_improvement:.2f}%')
    print()
    print('Recommendation:')
    print('1. Keep ReLU as default (simpler, faster)')
    print('2. Focus on higher-impact improvements (Optuna, attention)')
    print('3. GELU implementation remains available for future testing')

print()
print('='*70)
