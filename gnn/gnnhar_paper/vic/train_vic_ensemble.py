"""
VIC Ensemble Training - Train 20 models with different seeds, screen by val loss
This follows the GNNHAR paper's approach to handle training instability
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY, gnnhar_ratio_loss
from baselines.har_rv_baseline import fit_har, predict_har

print("\n" + "="*70)
print("  VIC ENSEMBLE TRAINING (MSE LOSS + NO RELU + APR 2026 TRAIN END)")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKER = 'VIC'
HORIZON = 5
TRAIN_START_DATE = "2020-01-01"  # Use more recent data for training
TRAIN_END_DATE = "2025-12-31"  # Train through end of 2025
TEST_START_DATE = "2026-01-01"   # Test all of 2026 (larger test set)
TEST_END_DATE = "2026-05-31"

MODELS_TO_TRAIN = ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']
N_HID = 16
N_EPOCHS = 1500
LR = 1e-3
WEIGHT_DECAY = 1e-5  # Paper uses 1e-5, not 1e-3 (100x less aggressive)
PATIENCE = 150
NUM_SEEDS = 20

SEEDS = [42, 123, 456, 789, 321, 111, 222, 333, 444, 555, 666, 777, 888, 999, 101, 202, 303, 404, 505, 606]

# =============================================================================
# LOAD DATA
# =============================================================================

print("[Data] Loading VIC data...")
close = load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices', tickers=[TICKER])
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()

def build_snapshots_for_period(rv_series, start_date, end_date, stride=1):
    """Build HAR snapshots for a specific period with stride.

    FIXED: Use full RV series for lookback, only filter targets by date.
    Loop through full series and filter by date (not by index calculation).
    """
    min_history = 22 + HORIZON

    X_list, y_list, date_list = [], [], []

    # Use full RV series for lookback
    full_rv = rv_series
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # If end_date is beyond last data date, use last data date
    if end_ts > full_rv.index[-1]:
        end_ts = full_rv.index[-1]

    # Loop through full series, filter targets by date range
    for i in range(min_history, len(full_rv), stride):
        # Current date for this snapshot
        current_date = full_rv.index[i]

        # Only include if target date is in range
        if start_ts <= current_date <= end_ts:
            # RV_t is already h-day volatility, just use it directly
            target = full_rv.iloc[i]

            # Lookback features (use full series)
            rv_d = full_rv.iloc[i-1:i].mean()
            rv_w = full_rv.iloc[i-5:i].mean()
            rv_m = full_rv.iloc[i-22:i].mean()

            X_list.append([rv_d, rv_w, rv_m])
            y_list.append(target)
            date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

train_start_ts = pd.Timestamp(TRAIN_START_DATE)
train_end_ts = pd.Timestamp(TRAIN_END_DATE)
test_start_ts = pd.Timestamp(TEST_START_DATE)
test_end_ts = pd.Timestamp(TEST_END_DATE)

print(f"\n[Split] Building train/val/test splits (stride=1)...")

# Build all snapshots from TRAIN_START_DATE to TRAIN_END_DATE
X_train_full, y_train_full, train_dates_full = build_snapshots_for_period(
    rv, start_date=train_start_ts, end_date=train_end_ts, stride=1
)

split_point = int(len(X_train_full) * 0.8)
X_train = X_train_full[:split_point]
y_train = y_train_full[:split_point]
train_dates = train_dates_full[:split_point]
X_val = X_train_full[split_point:]
y_val = y_train_full[split_point:]
val_dates = train_dates_full[split_point:]

# Scale targets by horizon (paper line 539: Y /= opt.horizon)
# This converts sum/horizon RV to average RV, making loss scale-invariant
y_train = y_train / HORIZON
y_val = y_val / HORIZON

X_test, y_test, test_dates = build_snapshots_for_period(
    rv, start_date=test_start_ts, end_date=test_end_ts, stride=1
)

# Scale test targets by horizon (must match training scale)
y_test = y_test / HORIZON

print(f"  Train: {len(X_train)} samples ({train_dates[0].date()} to {train_dates[-1].date()})")
print(f"  Val:   {len(X_val)} samples ({val_dates[0].date()} to {val_dates[-1].date()})")
print(f"  Test:  {len(X_test)} samples ({test_dates[0].date()} to {test_dates[-1].date()})")

print(f"\n[Distribution] Analysis (targets scaled by horizon={HORIZON}):")
print(f"  Train mean RV: {y_train.mean():.6f}")
print(f"  Val mean RV:   {y_val.mean():.6f}")
print(f"  Test mean RV:  {y_test.mean():.6f}")
print(f"  Test vs Train shift: {(y_test.mean() - y_train.mean()) / y_train.mean() * 100:+.1f}%")

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_learning_curves(model_name, train_histories):
    """Plot and save learning curves for all seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training loss
    ax1 = axes[0]
    for history in train_histories:
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], alpha=0.3, linewidth=1)

    # Plot average
    max_epochs = max(len(h['train_losses']) for h in train_histories)
    avg_train_loss = []
    for epoch in range(max_epochs):
        epoch_losses = [h['train_losses'][epoch] for h in train_histories if epoch < len(h['train_losses'])]
        if epoch_losses:
            avg_train_loss.append(np.mean(epoch_losses))
        else:
            avg_train_loss.append(avg_train_loss[-1] if avg_train_loss else 0)

    epochs_avg = range(1, len(avg_train_loss) + 1)
    ax1.plot(epochs_avg, avg_train_loss, 'b-', linewidth=2.5, label='Average')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (MSE)')
    ax1.set_title(f'{model_name} Training Learning Curves ({len(train_histories)} seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot validation loss
    ax2 = axes[1]
    for history in train_histories:
        epochs = range(1, len(history['val_losses']) + 1)
        ax2.plot(epochs, history['val_losses'], alpha=0.3, linewidth=1)

    # Plot average
    max_epochs = max(len(h['val_losses']) for h in train_histories)
    avg_val_loss = []
    for epoch in range(max_epochs):
        epoch_losses = [h['val_losses'][epoch] for h in train_histories if epoch < len(h['val_losses'])]
        if epoch_losses:
            avg_val_loss.append(np.mean(epoch_losses))
        else:
            avg_val_loss.append(avg_val_loss[-1] if avg_val_loss else 0)

    epochs_avg = range(1, len(avg_val_loss) + 1)
    ax2.plot(epochs_avg, avg_val_loss, 'r-', linewidth=2.5, label='Average')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss (MSE)')
    ax2.set_title(f'{model_name} Validation Learning Curves ({len(train_histories)} seeds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save figure
    output_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'vic_learning_curves'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{model_name}_learning_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [Saved] Learning curves to {output_file}")

# =============================================================================
# ENSEMBLE TRAINING FUNCTION
# =============================================================================

def train_ensemble(model_name, n_hid, seeds, X_train, y_train, X_val, y_val, save_dir=None):
    """Train multiple models with different seeds and track learning curves."""
    predictions_list = []
    val_losses_list = []
    train_histories = []  # Store training history for each seed
    trained_models = []  # Store trained model states

    for seed_idx, seed in enumerate(seeds):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        if model_name == 'HAR':
            model = MODEL_REGISTRY[model_name]()
        else:
            model = MODEL_REGISTRY[model_name](n_hid=n_hid)

        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # Use MSE loss (QL loss incompatible with NO ReLU - requires positive predictions)
        # QL loss causes NaN when predictions go negative (log of negative ratio)
        # Project standard: MSE on z-scored residuals (CLAUDE.md C3)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train).float().unsqueeze(1)
        y_t = torch.from_numpy(y_train).float().unsqueeze(1)
        X_v = torch.from_numpy(X_val).float().unsqueeze(1)
        y_v = torch.from_numpy(y_val).float().unsqueeze(1)

        # Track training history
        train_losses_epoch = []
        val_losses_epoch = []

        # Train with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(N_EPOCHS):
            model.train()
            optimizer.zero_grad()
            pred = model(X_t, torch.ones(1, 1))
            train_loss = criterion(pred, y_t)
            train_loss.backward()
            optimizer.step()

            # Store training loss
            train_losses_epoch.append(train_loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v, torch.ones(1, 1))
                val_loss = criterion(val_pred, y_v).item()

            # Store validation loss
            val_losses_epoch.append(val_loss)

            # Print progress every 10% of epochs (75 epochs for 750 total)
            if (epoch + 1) % (N_EPOCHS // 10) == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{N_EPOCHS}: train_loss={train_loss.item():.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        # Get test predictions
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
            test_pred = model(X_test_t, torch.ones(1, 1))
            predictions_list.append(test_pred.squeeze(0).squeeze(-1).numpy())

        val_losses_list.append(best_val_loss)
        train_histories.append({
            'seed': seed,
            'train_losses': train_losses_epoch,
            'val_losses': val_losses_epoch,
            'total_epochs': len(train_losses_epoch)
        })

        # Save trained model state
        trained_models.append({
            'seed': seed,
            'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'val_loss': best_val_loss
        })

    # Save all models if save_dir provided
    if save_dir is not None:
        model_dir = save_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, model_data in enumerate(trained_models):
            model_file = model_dir / f'model_{i}.pt'
            torch.save(model_data['state_dict'], model_file)

        print(f"  [Saved] {len(trained_models)} models to {model_dir}")

    return predictions_list, val_losses_list, train_histories, trained_models

# =============================================================================
# TRAIN ALL MODELS
# =============================================================================

print(f"\n{'='*70}")
print(f"  ENSEMBLE TRAINING ({NUM_SEEDS} MODELS WITH DIFFERENT SEEDS)")
print(f"{'='*70}\n")

results = {}

for model_name in MODELS_TO_TRAIN:
    print(f"\n[{model_name}] Training ensemble of {NUM_SEEDS} models...")

    # Create save directory
    ensemble_save_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'vic_ensemble_models'

    predictions_list, val_losses_list, train_histories, trained_models = train_ensemble(
        model_name, N_HID, SEEDS, X_train, y_train, X_val, y_val, save_dir=ensemble_save_dir
    )

    # Plot learning curves
    plot_learning_curves(model_name, train_histories)

    # Print individual results
    print(f"  Individual model results:")
    for i, (seed, val_loss, pred) in enumerate(zip(SEEDS, val_losses_list, predictions_list)):
        r2 = 1 - np.sum((y_test - pred)**2) / np.sum((y_test - y_test.mean())**2)
        mae = np.mean(np.abs(y_test - pred))
        status = "GOOD" if r2 > -5 else "POOR" if r2 > -100 else "FAIL"
        print(f"    Seed {seed}: val_loss={val_loss:.6f}, R²={r2:+8.2f}, MAE={mae:.6f} [{status}]")

    # Screen by validation loss (keep top 50%)
    median_val_loss = np.median(val_losses_list)
    screened_indices = [i for i, vl in enumerate(val_losses_list) if vl <= median_val_loss]
    screened_preds = [predictions_list[i] for i in screened_indices]

    print(f"  Screening: kept {len(screened_preds)}/{len(predictions_list)} models (val_loss <= {median_val_loss:.6f})")

    # Average predictions
    ensemble_pred = np.mean(screened_preds, axis=0)

    # Ensemble metrics
    r2_ensemble = 1 - np.sum((y_test - ensemble_pred)**2) / np.sum((y_test - y_test.mean())**2)
    mae_ensemble = np.mean(np.abs(y_test - ensemble_pred))
    rmse_ensemble = np.sqrt(np.mean((y_test - ensemble_pred)**2))

    print(f"  Ensemble ({len(screened_preds)} models): R² = {r2_ensemble:+.4f}, MAE = {mae_ensemble:.6f}")

    results[model_name] = {
        'r2': float(r2_ensemble),
        'mae': float(mae_ensemble),
        'rmse': float(rmse_ensemble),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'individual_r2': [float(1 - np.sum((y_test - p)**2) / np.sum((y_test - y_test.mean())**2)) for p in predictions_list],
        'val_losses': [float(vl) for vl in val_losses_list]
    }

    # Save ensemble metadata for inference
    ensemble_metadata = {
        'model_name': model_name,
        'n_hid': N_HID,
        'num_models': len(trained_models),
        'num_screened': len(screened_indices),
        'screened_indices': screened_indices,
        'seeds': SEEDS,
        'val_losses': [float(vl) for vl in val_losses_list],
        'train_config': {
            'n_epochs': N_EPOCHS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'patience': PATIENCE,
            'horizon': HORIZON
        }
    }

    model_dir = ensemble_save_dir / model_name
    metadata_file = model_dir / 'ensemble_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)

    print(f"  [Saved] Ensemble metadata to {metadata_file}")

# =============================================================================
# HAR OLS BASELINE
# =============================================================================

print(f"\n[HAR Baseline] Computing OLS HAR baseline...")

try:
    har_coeffs = fit_har(rv, train_end_ts)
    har_pred = predict_har(rv, har_coeffs, test_start_ts)
    har_pred_aligned = har_pred.reindex(test_dates).dropna()
    # Scale HAR predictions by horizon to match y_test scaling (line 130: y_test = y_test / HORIZON)
    har_pred_aligned = har_pred_aligned / HORIZON
    y_test_aligned = y_test[:len(har_pred_aligned)]

    if len(har_pred_aligned) > 0:
        ss_res = np.sum((y_test_aligned - har_pred_aligned.values) ** 2)
        ss_tot = np.sum((y_test_aligned - y_test_aligned.mean()) ** 2)
        har_r2 = 1 - (ss_res / (ss_tot + 1e-8))
        har_mae = np.mean(np.abs(y_test_aligned - har_pred_aligned.values))

        print(f"  HAR OLS: R² = {har_r2:+.4f}, MAE = {har_mae:.6f}")

        results['HAR_OLS'] = {
            'r2': float(har_r2),
            'mae': float(har_mae),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
except Exception as e:
    print(f"  Error: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"  SUMMARY: ENSEMBLE TRAINING RESULTS")
print(f"{'='*70}\n")

print(f"Ensemble Strategy: {NUM_SEEDS} models with different seeds")
print(f"Loss Function: MSE (QL loss incompatible with NO ReLU - requires positive predictions)")
print(f"Training Period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
print(f"Screening: Keep models with val_loss <= median")
print(f"Test Period: {TEST_START_DATE} to {TEST_END_DATE}")
print(f"\nModel Performance:")
print(f"{'Model':<15} {'R2':>10} {'MAE':>12} {'Improvement':>15}")
print(f"{'-'*60}")

baseline_r2 = results.get('HAR_OLS', {}).get('r2', 0)

for model_name in MODELS_TO_TRAIN + ['HAR_OLS']:
    if model_name in results and 'r2' in results[model_name]:
        r2 = results[model_name]['r2']
        mae = results[model_name]['mae']
        improvement = r2 - baseline_r2
        print(f"{model_name:<15} {r2:>+10.4f} {mae:>12.6f} {improvement:>+15.4f}")

print(f"\n{'='*70}\n")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'analysis'
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'vic_ensemble_training_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'strategy': 'ensemble_training',
        'train_start_date': TRAIN_START_DATE,
        'train_end_date': TRAIN_END_DATE,
        'test_period': f"{TEST_START_DATE} to {TEST_END_DATE}",
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'num_models': NUM_SEEDS,
        'seeds': SEEDS,
        'distribution_shift_pct': float((y_test.mean() - y_train.mean()) / y_train.mean() * 100),
        'results': results
    }, f, indent=2)

print(f"[Saved] Results saved to {output_file}\n")
