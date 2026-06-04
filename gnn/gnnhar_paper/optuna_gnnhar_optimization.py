"""
Optuna Hyperparameter Optimization for GNNHAR1L

Optimizes hyperparameters for GNNHAR1L volatility forecasting:
- Learning rate (lr)
- Weight decay (L2 regularization)
- Hidden dimension (n_hid)
- Adjacency threshold (graph density)
- Dropout rate (optional)

Expected improvement: +5-10% R² over baseline
Optimization time: ~8 hours (100 trials)
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
from datetime import datetime
import argparse
import json

# Optuna imports
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("[ERROR] Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.gnnhar_models import create_model, gnnhar_ratio_loss
from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.graph_builder import GraphBuilder
from gnn.gnnhar_paper.train_multi_stock import (
    MultiStockDataset,
    forward_pass_with_mask,
    train_single_model,
)
from src.volatility_labels import compute_log_returns
from gnn.build_graph import VN30_TICKERS


def objective(
    trial: optuna.Trial,
    model_name: str,
    train_dataset: MultiStockDataset,
    val_dataset: MultiStockDataset,
    n_epochs: int,
    device: str,
    returns: pd.DataFrame,
    train_end: pd.Timestamp,
) -> float:
    """
    Optuna objective function: train model and return validation R².

    Args:
        trial: Optuna trial object
        model_name: Model to optimize ('GNNHAR1L', 'GHAR', etc.)
        train_dataset: Training data
        val_dataset: Validation data
        n_epochs: Maximum epochs per trial
        device: 'cpu' or 'cuda'
        returns: Log returns DataFrame for adjacency construction
        train_end: Training end date for adjacency construction

    Returns:
        Validation R² (maximize this)
    """

    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)  # Expanded range to 1e-3
    n_hid = trial.suggest_categorical('n_hid', [16, 32, 64, 128])  # Added 128 option
    batch_size = trial.suggest_categorical('batch_size', [256, 512])  # Optimize batch size
    activation = trial.suggest_categorical('activation', ['relu', 'gelu'])  # Optimize activation

    # For GNN models, also optimize graph parameters
    if model_name in ['GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L', 'GHAR']:
        adj_method = trial.suggest_categorical('adj_method', ['pearson', 'glasso'])  # Optimize graph method
        adj_threshold = trial.suggest_float('adj_threshold', 0.2, 0.5) if adj_method == 'pearson' else 0.3
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.3)

    # Create adjacency matrix with suggested method and threshold
    if model_name in ['GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L', 'GHAR']:
        builder = GraphBuilder(
            method=adj_method,
            threshold=adj_threshold,
        )
        adj = builder.build_adjacency(returns, train_end)
    else:
        # HAR doesn't use graph
        adj = None
        adj_method = None
        adj_threshold = None
        dropout_rate = 0.0

    # Create model with suggested hyperparameters
    model = create_model(
        model_name,
        n_hid=n_hid,
        activation=activation,
        dropout=dropout_rate
    )

    # Create data loader with suggested batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Windows compatibility
        pin_memory=False,
    )

    # Train model
    print(f"\n[Trial {trial.number}] Training with hyperparameters:")
    print(f"  lr={lr:.2e}, weight_decay={weight_decay:.2e}, n_hid={n_hid}")
    print(f"  activation={activation}, batch_size={batch_size}")
    if model_name in ['GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L', 'GHAR']:
        print(f"  adj_method={adj_method}, adj_threshold={adj_threshold:.3f}, dropout={dropout_rate:.3f}")

    try:
        result = train_single_model(
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            adj=torch.from_numpy(adj).float() if adj is not None else None,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            patience=50,  # Shorter patience for Optuna (faster pruning)
        )

        # Return validation R² (Optuna maximizes this)
        val_r2 = result.get('val_r2', 0.0)

        print(f"[Trial {trial.number}] Val R²: {val_r2:.4f}, Epochs: {result['n_epochs']}")

        return val_r2

    except Exception as e:
        print(f"[Trial {trial.number}] FAILED: {e}")
        return -1.0  # Penalize failed trials


def run_optuna_study(
    model_name: str,
    train_dataset: MultiStockDataset,
    val_dataset: MultiStockDataset,
    n_trials: int,
    n_epochs: int,
    device: str,
    study_name: str,
    storage: str,
    returns: pd.DataFrame,
    train_end: pd.Timestamp,
) -> dict:
    """
    Run Optuna hyperparameter optimization study.

    Args:
        model_name: Model to optimize
        train_dataset: Training data
        val_dataset: Validation data
        n_trials: Number of trials
        n_epochs: Maximum epochs per trial
        device: 'cpu' or 'cuda'
        study_name: Name for this study
        storage: Database URL for saving results

    Returns:
        dict with best hyperparameters and metrics
    """

    print("\n" + "="*70)
    print(f"  OPTUNA HYPERPARAMETER OPTIMIZATION: {model_name}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Trials: {n_trials}")
    print(f"  Max epochs per trial: {n_epochs}")
    print(f"  Device: {device}")
    print(f"  Optimizing: lr, weight_decay, n_hid, batch_size, activation, dropout, adj_method, adj_threshold\n")

    # Create study with TPE sampler (Tree-structured Parzen Estimator)
    sampler = TPESampler(seed=42)  # Reproducible sampling

    # Median pruner: stop unpromising trials early
    pruner = MedianPruner(
        n_startup_trials=10,  # Don't prune first 10 trials
        n_warmup_steps=30,    # Allow 30 epochs before pruning
    )

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize R²
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,   # Resume if study exists
    )

    print(f"Study name: {study_name}")
    print(f"Database: {storage}")
    print(f"Resuming from existing study: {len(study.trials)} trials completed\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            model_name=model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_epochs=n_epochs,
            device=device,
            returns=returns,
            train_end=train_end,
        ),
        n_trials=n_trials,
        n_jobs=1,  # Sequential trials (can parallelize with n_jobs > 1)
        show_progress_bar=True,
    )

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    print("\n" + "="*70)
    print("  OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest trial (#{best_trial.number}):")
    print(f"  Val R²: {best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    # Save results
    results_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'optuna'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = results_dir / f'{model_name}_optuna_{timestamp}.json'

    save_data = {
        'model': model_name,
        'study_name': study_name,
        'n_trials': n_trials,
        'best_trial_number': best_trial.number,
        'best_val_r2': float(best_value),
        'best_params': best_params,
        'timestamp': timestamp,
    }

    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_trial': best_trial,
        'study': study,
    }


def main():
    parser = argparse.ArgumentParser(description='Optuna optimization for GNNHAR models')
    parser.add_argument('--model', type=str, default='GNNHAR1L',
                        choices=['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L'],
                        help='Model to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs per trial (shorter than full training)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forecast horizon')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Study name (default: auto-generated)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_studies.db',
                        help='Database URL for Optuna storage')
    parser.add_argument('--train_end', type=str, default='2025-12-31',
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2026-01-01',
                        help='Test start date (YYYY-MM-DD)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*70 + "\n")

    print(f"[Config] Model: {args.model}")
    print(f"[Config] Trials: {args.n_trials}")
    print(f"[Config] Max epochs: {args.epochs}")
    print(f"[Config] Horizon: h={args.horizon}")
    print(f"[Config] Storage: {args.storage}")
    print(f"[Config] Optimizing: lr, weight_decay, n_hid, batch_size, activation, dropout, adj_method, adj_threshold\n")

    # Generate study name if not provided
    if args.study_name is None:
        args.study_name = f'{args.model}_h{args.horizon}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # Load data
    print("[Step 1] Loading multi-stock data...")
    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=args.horizon,
        train_end=args.train_end,
        test_start=args.test_start,
    )

    # Load and prepare data
    print("[Step 2] Loading and preparing data...")
    loader.load_data()           # Load close prices
    loader.build_features()      # Build HAR features
    loader.flatten_dataset()     # Flatten to (N_samples, 3)
    loader.split_train_val_test()  # Split by date

    print(f"  Close prices shape: {loader.close.shape}\n")

    # Build adjacency matrix (for graph construction during trials)
    print("[Step 3] Building adjacency matrix template...")

    # Compute log returns from close prices (needed for adjacency)
    returns = compute_log_returns(loader.close)

    builder = GraphBuilder(
        method='pearson',
        threshold=0.3,  # Default, will be optimized during trials
    )
    adj = builder.build_adjacency(returns, pd.Timestamp(args.train_end))

    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Density: {(adj > 0).sum() / (adj.shape[0] * adj.shape[0]):.2%}\n")

    # Create datasets
    print("[Step 4] Creating PyTorch datasets...")
    (X_train, y_train, stocks_train, dates_train,
     X_val, y_val, stocks_val, dates_val,
     X_test, y_test, stocks_test, dates_test) = loader.prepare_pytorch_data(val_split=0.2)

    train_dataset = MultiStockDataset(X_train, y_train, stocks_train, dates_train)
    val_dataset = MultiStockDataset(X_val, y_val, stocks_val, dates_val)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}\n")

    # Run Optuna optimization
    print("[Step 5] Running Optuna optimization...")
    result = run_optuna_study(
        model_name=args.model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=args.n_trials,
        n_epochs=args.epochs,
        device=args.device,
        study_name=args.study_name,
        storage=args.storage,
        returns=returns,
        train_end=pd.Timestamp(args.train_end),
    )

    print("\n" + "="*70)
    print("  RECOMMENDATION")
    print("="*70)
    print(f"\nBest hyperparameters found:")
    print(f"  Val R²: {result['best_value']:.4f}")
    print(f"\nTo train final model with these hyperparameters, run:")
    print(f"\npython gnn/gnnhar_paper/train_multi_stock.py \\")
    print(f"    --model {args.model} \\")
    for param, value in result['best_params'].items():
        if param == 'lr':
            print(f"    --lr {value:.4f} \\")
        elif param == 'weight_decay':
            print(f"    --weight_decay {value:.4f} \\")
        elif param == 'n_hid':
            print(f"    --n_hid {value} \\")
        elif param == 'batch_size':
            print(f"    --batch_size {value} \\")
        elif param == 'activation':
            print(f"    --activation {value} \\")
        elif param == 'adj_method':
            print(f"    --adj_method {value} \\")
        elif param == 'adj_threshold':
            print(f"    --adj_threshold {value:.3f} \\")
        elif param == 'dropout':
            print(f"    --dropout {value:.3f} \\")
    print(f"    --n_seeds 20 \\")
    print(f"    --epochs 400")
    print("\n")


if __name__ == '__main__':
    main()
