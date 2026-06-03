"""
Multi-Stock PyTorch GNNHAR Training

Trains GNNHAR models (HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L) with multi-stock data.
This is the CORRECT implementation that pools 30 stocks together for training,
unlike the single-stock training in train_gnnhar_paper.py.

Key differences from single-stock training:
    1. Data: 30 stocks pooled -> ~96,000 samples (vs ~1,200 per stock)
    2. Batch: Random samples from multiple stocks (not just one stock)
    3. Forward: Reshape to (batch, 30, 3) for GCN, extract batch stocks only
    4. Graph: Real adjacency matrix (30x30) instead of identity (1x1)

Expected results: All neural models should beat HAR OLS baseline (R² ≈ 0.75)

Usage:
    python train_multi_stock.py --model GHAR --n_seeds 20 --epochs 1500
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.gnnhar_models import (
    MODEL_REGISTRY,
    create_model,
    gnnhar_ratio_loss,
)
from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.graph_builder import GraphBuilder
from src.volatility_labels import compute_log_returns
from gnn.build_graph import VN30_TICKERS


class MultiStockDataset(Dataset):
    """
    Dataset for multi-stock training.

    Stores flattened data (N_samples, 3) with stock indices and dates.
    DataLoader randomly samples from ALL stocks, creating diverse batches.
    """

    def __init__(self, X, y, stocks, dates):
        """
        Args:
            X: (N_samples, 3) HAR features
            y: (N_samples,) RV targets
            stocks: (N_samples,) stock indices (0-29)
            dates: (N_samples,) date timestamps
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.stocks = torch.from_numpy(stocks).long()
        self.dates = dates  # Keep as pandas DatetimeIndex

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.stocks[idx], idx


def forward_pass_with_mask(
    model: nn.Module,
    batch_X: torch.Tensor,
    adj: torch.Tensor,
    batch_stocks: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass with stock masking for multi-stock training (VECTORIZED).

    Critical for GNNHAR: GCN layers expect (batch, N, features) where N=30 stocks.
    Each batch contains random samples from different stocks, so we need to:

    1. Create (batch, 30, 3) node_feat matrix
    2. Place actual features in correct stock positions (vectorized, no Python loop)
    3. Zero out features for stocks not in batch
    4. Forward through model
    5. Extract predictions for actual stocks in batch

    Args:
        model: GNNHAR model (HAR, GHAR, GNNHAR1L, etc.)
        batch_X: (batch_size, 3) HAR features (flattened)
        adj: (30, 30) adjacency matrix (full graph)
        batch_stocks: (batch_size,) stock indices for each sample

    Returns:
        (batch_size,) predictions for samples in batch
    """
    batch_size = batch_X.shape[0]
    n_stocks = adj.shape[0]  # 30

    # Step 1: Create node_feat matrix (batch, N, 3)
    node_feat = torch.zeros(batch_size, n_stocks, 3, device=batch_X.device)

    # Step 2: Place actual features in correct stock positions (VECTORIZED)
    # For each sample i, put its features at column [stock_id, :]
    # Use advanced indexing instead of Python loop (10x faster)
    batch_indices = torch.arange(batch_size, device=batch_X.device)
    node_feat[batch_indices, batch_stocks, :] = batch_X

    # Step 3: Forward through model
    # node_feat: (batch, N, 3) -> predictions: (batch, N)
    predictions = model(node_feat, adj)

    # Step 4: Extract predictions for actual stocks in batch
    # predictions[i] should be predictions[i, stock_id]
    batch_pred = predictions[torch.arange(batch_size), batch_stocks]

    return batch_pred


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_dataset: MultiStockDataset,
    adj: torch.Tensor,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    patience: int = 150,
    grad_clip: float = 1.0,
) -> dict:
    """
    Train a single model with early stopping.

    Args:
        model: GNNHAR model instance
        train_loader: Training data loader
        val_dataset: Validation dataset (for early stopping)
        adj: Adjacency matrix (30, 30)
        n_epochs: Maximum epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: 'cpu' or 'cuda'
        patience: Early stopping patience
        grad_clip: Gradient clipping max norm (0 to disable, default 1.0)

    Returns:
        dict with best_val_loss, n_epochs, train_loss_history, val_loss_history
    """
    model = model.to(device)
    adj = adj.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    # Prepare validation data
    val_X = val_dataset.X.to(device)
    val_y = val_dataset.y.to(device)
    val_stocks = val_dataset.stocks.to(device)

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch_X, batch_y, batch_stocks, _ in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_stocks = batch_stocks.to(device)

            # Forward pass with masking
            pred = forward_pass_with_mask(model, batch_X, adj, batch_stocks)

            # Clip predictions to prevent QL loss singularity
            # QL loss requires positive predictions (ratio-based)
            pred = torch.clamp(pred, min=1e-4, max=None)

            # QL loss
            loss = gnnhar_ratio_loss(pred, batch_y)

            # Check for NaN/Inf loss (numerical instability detection)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] NaN/Inf loss at epoch {epoch+1}, seed {seed}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Architectural guardrail: Gradient clipping for stability
            # Prevents gradient explosion from extreme ratios in loss function
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        # Average training loss for this epoch
        train_loss_avg = train_loss_sum / n_batches
        train_loss_history.append(train_loss_avg)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = forward_pass_with_mask(model, val_X, adj, val_stocks)
            val_pred = torch.clamp(val_pred, min=1e-4, max=None)
            val_loss = gnnhar_ratio_loss(val_pred, val_y).item()

            # Check for NaN/Inf validation loss
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"  [WARN] NaN/Inf val loss at epoch {epoch+1}")

        val_loss_history.append(val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"Train Loss={train_loss_avg:.6f}, Val Loss={val_loss:.6f}")

            # Architectural guardrail: Ratio monitoring for early warning
            # Compute ratio statistics to detect numerical instability
            with torch.no_grad():
                # Use last batch for ratio monitoring (representative sample)
                if pred is not None and batch_y is not None:
                    ratio = batch_y / (pred + 1e-4)
                    ratio_mean = ratio.mean().item()
                    ratio_std = ratio.std().item()
                    ratio_min = ratio.min().item()
                    ratio_max = ratio.max().item()

                    print(f"           Ratio: mean={ratio_mean:.4f}, std={ratio_std:.4f}, "
                          f"range=[{ratio_min:.4f}, {ratio_max:.4f}]")

                    # Warning if ratio is in extreme region
                    if ratio_max > 100:
                        print(f"           [WARN] Extreme ratio detected (max={ratio_max:.2f})")
                        print(f"                  Model may be predicting near-zero volatility")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Calculate validation R² at best epoch (for Optuna optimization)
    model.eval()
    with torch.no_grad():
        val_pred = forward_pass_with_mask(model, val_X, adj, val_stocks)
        val_pred = torch.clamp(val_pred, min=1e-4, max=None)

        # R² = 1 - MSE(model, y) / MSE(mean(y), y)
        val_y_numpy = val_y.cpu().numpy()
        val_pred_numpy = val_pred.cpu().numpy()

        # Check for empty validation dataset
        if len(val_y_numpy) == 0:
            print("  [WARN] Empty validation dataset, val_r2 set to 0")
            val_r2 = 0.0
        else:
            ss_res = np.sum((val_y_numpy - val_pred_numpy) ** 2)
            ss_tot = np.sum((val_y_numpy - val_y_numpy.mean()) ** 2)
            val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'best_val_loss': best_val_loss,
        'val_r2': val_r2,  # Add validation R² for Optuna
        'n_epochs': epoch + 1,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
    }


def plot_learning_curves(
    train_loss_history: list,
    val_loss_history: list,
    model_name: str,
    seed: int,
    save_path: Path,
    timestamp: str = None,
):
    """
    Plot training and validation loss curves on the same chart.

    Args:
        train_loss_history: List of training losses per epoch
        val_loss_history: List of validation losses per epoch
        model_name: Model name for title
        seed: Random seed for filename
        save_path: Directory to save the plot
        timestamp: Optional timestamp string for filename
    """
    epochs = range(1, len(train_loss_history) + 1)

    # Calculate y-axis range EXCLUDING first 5 epochs (ignore initial high losses)
    converged_train = train_loss_history[5:]  # Skip first 5 epochs
    converged_val = val_loss_history[5:]

    if len(converged_train) > 0:
        min_loss = min(min(converged_train), min(converged_val))
        max_loss = max(max(converged_train), max(converged_val))
        padding = (max_loss - min_loss) * 0.1  # 10% padding

        # Set y-axis range focusing on converged values
        y_min = max(0.0, min_loss - padding)
        y_max = min_loss + padding + 0.1
    else:
        # Fallback if too few epochs
        y_min = 1.0
        y_max = 2.0

    plt.figure(figsize=(12, 7))  # Larger figure for better visibility
    plt.plot(epochs, train_loss_history, 'b-', label='Train Loss', linewidth=3, marker='o', markersize=4)
    plt.plot(epochs, val_loss_history, 'r-', label='Val Loss', linewidth=3, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('QL Loss', fontsize=14)
    plt.title(f'{model_name} Learning Curve (Seed {seed})', fontsize=16)

    # Set y-axis limits to focus on converged range (exclude initial spikes)
    plt.ylim(y_min, y_max)

    # Set y-axis ticks with 0.05 increments for clarity (1.10, 1.15, 1.20, 1.25, ...)
    ax = plt.gca()
    import math
    y_ticks = []
    current = math.ceil(y_min * 20) / 20  # Round up to nearest 0.05
    while current <= y_max:
        y_ticks.append(round(current, 2))
        current += 0.05
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # Add horizontal grid lines at every tick for better readability
    ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)

    # Save plot with timestamp
    if timestamp:
        plot_file = save_path / f'{model_name}_seed{seed}_learning_curve_{timestamp}.png'
    else:
        plot_file = save_path / f'{model_name}_seed{seed}_learning_curve.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved learning curve: {plot_file}")


def plot_ensemble_learning_curves(
    model_train_losses: list,
    model_val_loss_histories: list,
    model_name: str,
    save_path: Path,
    timestamp: str = None,
):
    """
    Plot ensemble average learning curves with confidence bands.

    Args:
        model_train_losses: List of train loss histories for each seed
        model_val_loss_histories: List of val loss histories for each seed
        model_name: Model name for title
        save_path: Directory to save the plot
        timestamp: Optional timestamp string for filename
    """
    # Find maximum epochs across all seeds
    max_epochs = max(len(h) for h in model_train_losses)

    # Pad shorter histories with their last value
    train_losses_padded = []
    val_losses_padded = []

    for train_hist, val_hist in zip(model_train_losses, model_val_loss_histories):
        # Pad training history
        train_padded = list(train_hist) + [train_hist[-1]] * (max_epochs - len(train_hist))
        train_losses_padded.append(train_padded)

        # Pad validation history
        val_padded = list(val_hist) + [val_hist[-1]] * (max_epochs - len(val_hist))
        val_losses_padded.append(val_padded)

    # Convert to numpy array for statistical calculations
    train_losses_array = np.array(train_losses_padded)
    val_losses_array = np.array(val_losses_padded)

    # Compute mean and std
    train_mean = train_losses_array.mean(axis=0)
    train_std = train_losses_array.std(axis=0)
    val_mean = val_losses_array.mean(axis=0)
    val_std = val_losses_array.std(axis=0)

    epochs = range(1, max_epochs + 1)

    # Calculate y-axis range EXCLUDING first 5 epochs (ignore initial high losses)
    converged_train = [train_hist[5:] for train_hist in train_losses_padded]
    converged_val = [val_hist[5:] for val_hist in val_losses_padded]

    # Flatten to find min/max across all seeds and epochs
    flat_train = [val for sublist in converged_train for val in sublist]
    flat_val = [val for sublist in converged_val for val in sublist]

    if flat_train and flat_val:
        min_loss = min(min(flat_train), min(flat_val))
        max_loss = max(max(flat_train), max(flat_val))
        padding = (max_loss - min_loss) * 0.1  # 10% padding

        # Set y-axis range focusing on converged values
        y_min = max(0.0, min_loss - padding)
        y_max = min_loss + padding + 0.1
    else:
        # Fallback if too few epochs
        y_min = 1.0
        y_max = 2.0

    plt.figure(figsize=(12, 7))  # Larger figure for better visibility
    plt.plot(epochs, train_mean, 'b-', label='Train Loss (mean)', linewidth=3, marker='o', markersize=3)
    plt.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        color='b',
        alpha=0.2,
        label='Train Loss (±1 std)',
    )
    plt.plot(epochs, val_mean, 'r-', label='Val Loss (mean)', linewidth=3, marker='s', markersize=3)
    plt.fill_between(
        epochs,
        val_mean - val_std,
        val_mean + val_std,
        color='r',
        alpha=0.2,
        label='Val Loss (±1 std)',
    )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('QL Loss', fontsize=14)
    plt.title(f'{model_name} Ensemble Learning Curves ({len(model_train_losses)} seeds)', fontsize=16)

    # Set y-axis limits to focus on converged range (exclude initial spikes)
    plt.ylim(y_min, y_max)

    # Set y-axis ticks with 0.05 increments for clarity (1.10, 1.15, 1.20, 1.25, ...)
    ax = plt.gca()
    import math
    y_ticks = []
    current = math.ceil(y_min * 20) / 20  # Round up to nearest 0.05
    while current <= y_max:
        y_ticks.append(round(current, 2))
        current += 0.05
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # Add horizontal grid lines at every tick for better readability
    ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)

    # Save plot with timestamp
    if timestamp:
        plot_file = save_path / f'{model_name}_ensemble_learning_curve_{timestamp}.png'
    else:
        plot_file = save_path / f'{model_name}_ensemble_learning_curve.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved ensemble learning curve: {plot_file}")


def train_ensemble(
    model_name: str,
    train_dataset: MultiStockDataset,
    val_dataset: MultiStockDataset,
    test_dataset: MultiStockDataset,
    adj: torch.Tensor,
    n_seeds: int,
    n_hid: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    horizon: int,
    activation: str = 'relu',
    dropout: float = 0.0,
    grad_clip: float = 1.0,
    timestamp: str = None,
) -> dict:
    """
    Train ensemble of models with different random seeds.

    v1.1_GELU: Added activation parameter for testing GELU vs ReLU
    v1.2_DROPOUT: Added dropout parameter for regularization
    v1.3_LOSS_FIX: Added horizon parameter for model saving

    For each seed:
        1. Set random seed
        2. Create model
        3. Train with early stopping
        4. Save validation loss and predictions

    After training all seeds:
        1. Screen models by validation loss (keep top 50%)
        2. Average predictions from screened models
        3. Compute test metrics

    Args:
        model_name: Model name ('HAR', 'GHAR', 'GNNHAR1L', etc.)
        train_dataset: Training data
        val_dataset: Validation data
        test_dataset: Test data
        adj: Adjacency matrix
        n_seeds: Number of models in ensemble
        n_hid: Hidden dimension (ignored for HAR)
        n_epochs: Maximum epochs per model
        lr: Learning rate
        weight_decay: L2 regularization
        batch_size: Batch size
        device: 'cpu' or 'cuda'
        activation: Activation function ('relu' or 'gelu')
        dropout: Dropout rate for regularization (0.0-0.3)
        grad_clip: Gradient clipping max norm (0 to disable, default 1.0)

    Returns:
        dict with ensemble predictions, metrics, and individual model results
    """
    print(f"\n{'='*70}")
    print(f"  Training Ensemble: {model_name}")
    print(f"  Models: {n_seeds}, n_hid: {n_hid}, lr: {lr}, weight_decay: {weight_decay}")
    print(f"  Activation: {activation.upper()}")
    print(f"  Dropout: {dropout:.3f}")
    print(f"{'='*70}")
    print(f"  [INFO] Version: v1.3_LOSS_FIX")
    print(f"  [INFO] Using CORRECTED gnnhar_ratio_loss (y_true/y_pred)")
    print(f"  [INFO] Loss function: GNNHAR Ratio Loss (NOT standard QLIKE)")
    print(f"  [INFO] Guardrails: ratio clipping=YES, gradient clipping=max_norm={grad_clip}")
    print(f"  [INFO] Monitoring: ratio stats every 10 epochs")
    print(f"  [WARNING] Models trained before 2026-06-01 used INCORRECT loss")
    print(f"  [WARNING] Old results are INVALID - do not compare with new results")
    print(f"{'='*70}\n")

    # Create data loader for training
    # Note: num_workers=0 for Windows compatibility (multiprocessing issues on Windows)
    # Vectorized forward pass provides speedup instead of parallel data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Windows: multiprocessing issues with PyTorch
        pin_memory=False,  # False for CPU training
    )

    # Create results directory for plots
    results_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'multi_stock'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Storage for individual models
    model_val_losses = []
    model_val_loss_histories = []  # Store full val loss histories
    model_train_losses = []  # Store train loss histories
    model_predictions = []
    model_epochs = []

    # Seeds for ensemble (from paper)
    seeds = [42, 123, 456, 789, 321, 111, 222, 333, 444, 555,
             666, 777, 888, 999, 101, 202, 303, 404, 505, 606][:n_seeds]

    for seed in seeds:
        print(f"[Seed {seed}] Training...")

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model with activation and dropout parameters
        model = create_model(model_name, n_hid=n_hid, activation=activation, dropout=dropout)

        # Train
        result = train_single_model(
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            adj=adj,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            grad_clip=grad_clip,
        )

        # Store validation loss and training history
        model_val_losses.append(result['best_val_loss'])
        model_val_loss_histories.append(result['val_loss_history'])
        model_train_losses.append(result['train_loss_history'])
        model_epochs.append(result['n_epochs'])

        # Save model checkpoint for inference
        models_dir = PROJECT_ROOT / 'models' / 'gnnhar_paper_multi_stock' / f'h{horizon}' / model_name
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f'seed{seed}_{timestamp}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'seed': seed,
            'val_loss': result['best_val_loss'],
            'n_hid': n_hid,
            'activation': activation,
            'dropout': dropout,
            'version': 'v1.3_LOSS_FIX',
        }, model_path)

        # Plot learning curve for this seed
        plot_learning_curves(
            train_loss_history=result['train_loss_history'],
            val_loss_history=result['val_loss_history'],
            model_name=model_name,
            seed=seed,
            save_path=results_dir,
            timestamp=timestamp,
        )

        # Test predictions
        model.eval()
        with torch.no_grad():
            test_X = test_dataset.X.to(device)
            test_stocks = test_dataset.stocks.to(device)
            test_pred = forward_pass_with_mask(model, test_X, adj, test_stocks)
            test_pred = torch.clamp(test_pred, min=1e-4, max=None)
            model_predictions.append(test_pred.cpu().numpy())

        print(f"[Seed {seed}] Val loss: {result['best_val_loss']:.6f}, "
              f"Epochs: {result['n_epochs']}")

    # Screen by validation loss (keep top 50%)
    median_val_loss = np.median(model_val_losses)
    screened_indices = [i for i, vl in enumerate(model_val_losses) if vl <= median_val_loss]
    screened_preds = [model_predictions[i] for i in screened_indices]

    print(f"\n[Ensemble] Screened {len(screened_preds)}/{n_seeds} models "
          f"(val loss <= {median_val_loss:.6f})")

    # Average predictions
    ensemble_pred = np.mean(screened_preds, axis=0)

    # Compute metrics
    test_y = test_dataset.y.numpy()
    ss_res = np.sum((test_y - ensemble_pred)**2)
    ss_tot = np.sum((test_y - test_y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(test_y - ensemble_pred))
    rmse = np.sqrt(np.mean((test_y - ensemble_pred)**2))

    print(f"[Ensemble] Test R2: {r2:+.4f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    # Plot ensemble learning curves
    print("\n[Ensemble] Generating learning curves...")
    plot_ensemble_learning_curves(
        model_train_losses=model_train_losses,
        model_val_loss_histories=model_val_loss_histories,
        model_name=model_name,
        save_path=results_dir,
        timestamp=timestamp,
    )

    return {
        'model_name': model_name,
        'ensemble_pred': ensemble_pred,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'n_models': len(screened_preds),
        'model_val_losses': model_val_losses,
        'model_val_loss_histories': model_val_loss_histories,
        'model_train_losses': model_train_losses,
        'model_epochs': model_epochs,
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-stock GNNHAR training')
    parser.add_argument('--model', type=str, default='GHAR',
                        choices=['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L'],
                        help='Model to train')
    parser.add_argument('--n_seeds', type=int, default=20,
                        help='Number of models in ensemble')
    parser.add_argument('--n_hid', type=int, default=16,
                        help='Hidden dimension (ignored for HAR)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Maximum epochs per model (models converge ~175, 400 gives 2.3x safety margin)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (larger = faster, 512 recommended for CPU)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forecast horizon (days)')
    parser.add_argument('--train_end', type=str, default='2025-12-31',
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2026-01-01',
                        help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--adj_method', type=str, default='pearson',
                        choices=['pearson', 'glasso'],
                        help='Adjacency matrix method')
    parser.add_argument('--adj_threshold', type=float, default=0.3,
                        help='Correlation threshold for Pearson adjacency')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu'],
                        help='Activation function (v1.1_GELU: GELU expected +2-5%% R² improvement)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for regularization (0.0-0.3, default 0.0)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm for stability (0 to disable, default 1.0)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  MULTI-STOCK PYTORCH GNNHAR TRAINING v1.2_DROPOUT")
    print("="*70 + "\n")

    print(f"[Config] Model: {args.model}")
    print(f"[Config] Ensemble: {args.n_seeds} models")
    print(f"[Config] Horizon: h={args.horizon}")
    print(f"[Config] Train end: {args.train_end}")
    print(f"[Config] Test start: {args.test_start}")
    print(f"[Config] Adjacency: {args.adj_method}, thresh={args.adj_threshold}")
    print(f"[Config] Activation: {args.activation.upper()}")
    print(f"[Config] Dropout: {args.dropout:.3f}")
    print(f"[Config] Device: {args.device}\n")

    # Step 1: Load multi-stock data
    print("[Step 1] Loading multi-stock data...")
    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=args.horizon,
        train_end=args.train_end,
        test_start=args.test_start,
    )
    loader.load_data()
    loader.build_features()
    loader.flatten_dataset()
    loader.split_train_val_test()

    # Use prepare_pytorch_data to get train/val/test split
    X_train, y_train, stocks_train, dates_train, \
    X_val, y_val, stocks_val, dates_val, \
    X_test, y_test, stocks_test, dates_test = loader.prepare_pytorch_data(val_split=0.2)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples\n")

    # Step 2: Build adjacency matrix
    print("[Step 2] Building adjacency matrix...")
    returns = compute_log_returns(loader.close)

    graph_builder = GraphBuilder(
        method=args.adj_method,
        threshold=args.adj_threshold,
    )
    adj = graph_builder.build_adjacency(returns, pd.Timestamp(args.train_end))
    adj_tensor = torch.from_numpy(adj).float()

    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Density: {(adj > 0).sum() / (adj.shape[0] * adj.shape[0]):.2%}\n")

    # Validate adjacency matrix
    if adj.shape[0] != len(VN30_TICKERS):
        raise ValueError(f"Adjacency shape {adj.shape} != n_stocks {len(VN30_TICKERS)}")

    row_sums = adj.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.1):
        print(f"  [WARN] Adjacency not row-normalized. Row sums: {row_sums[:5]}")

    # Step 3: Create datasets
    print("[Step 3] Creating datasets...")
    train_dataset = MultiStockDataset(X_train, y_train, stocks_train, dates_train)
    val_dataset = MultiStockDataset(X_val, y_val, stocks_val, dates_val)
    test_dataset = MultiStockDataset(X_test, y_test, stocks_test, dates_test)
    print(f"  Datasets created\n")

    # Step 4: Train ensemble
    print("[Step 4] Training ensemble...")

    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result = train_ensemble(
        model_name=args.model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        adj=adj_tensor,
        n_seeds=args.n_seeds,
        n_hid=args.n_hid,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        horizon=args.horizon,
        activation=args.activation,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        timestamp=timestamp,
    )

    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    results_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'multi_stock'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use the same timestamp from training (already generated above)
    result_file = results_dir / f'{args.model}_{args.activation}_h{args.horizon}_{timestamp}.json'

    save_data = {
        'model': args.model,
        'activation': args.activation,
        'version': 'v1.3_LOSS_FIX',  # Fixed ratio inversion, renamed to gnnhar_ratio_loss
        'dropout': args.dropout,
        'horizon': args.horizon,
        'adj_method': args.adj_method,
        'adj_threshold': args.adj_threshold,
        'n_seeds': args.n_seeds,
        'n_hid': args.n_hid,
        'test_r2': float(result['r2']),
        'test_mae': float(result['mae']),
        'test_rmse': float(result['rmse']),
        'n_models': result['n_models'],
        'model_val_losses': [float(x) for x in result['model_val_losses']],
        'model_epochs': result['model_epochs'],
    }

    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"  Results saved to: {result_file}\n")

    # Print summary
    print("="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  Model: {args.model}")
    print(f"  Ensemble: {result['n_models']} models (screened from {args.n_seeds})")
    print(f"  Test R2:   {result['r2']:+.4f}")
    print(f"  Test MAE:  {result['mae']:.6f}")
    print(f"  Test RMSE: {result['rmse']:.6f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
