"""
Extended training script for GNNHAR paper replication.

This script implements the IMPROVED data split that addresses the volatility
regime shift issue identified in the original approach.

Key improvements over original train_gnnhar_paper.py:
1. Training extends through 2024 (instead of ending 2022) - includes 2022-2024 volatility patterns
2. Validation uses only 2025 data (instead of 2022-2025) - more representative of test period
3. No gap between validation and test (2025 → 2026 vs 2022 → 2026)

Expected benefits:
- Training samples: +30% increase (3,828 → ~4,900)
- Prediction bias: 48% → <15% underestimation
- Test R²: -0.5 → +0.1 to +0.3 improvement

Usage:
    python gnn/gnnhar_paper/train_extended.py --horizon 5 --model GNNHAR1L

or for all horizons:
    python gnn/gnnhar_paper/train_extended.py
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from os import makedirs

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer
from gnn.gnnhar_paper.glasso_adjacency import glasso_adjacency
from gnn.gnnhar_paper.rolling_datasets import build_static_snapshots
from baselines.har_rv_baseline import fit_har, predict_har
from gnn.gnnhar_paper.evaluation import compute_metrics as eval_metrics

# ==============================================================================
# CONFIGURATION
# ==============================================================================

HORIZONS = [1, 5, 10, 20]
GLOBAL_TEST_START = "2026-01-01"

# Extended split configuration (NEW!)
EXTENDED_TRAIN_END = "2024-12-31"  # Train through 2024 (not 2022)
EXTENDED_VAL_START = "2025-01-01"   # Validate on 2025 only
EXTENDED_VAL_END = "2025-12-31"     # End of 2025

# Model variants to train
MODEL_NAMES = ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']

# Training hyperparameters
N_HID = 16
N_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 40
NUM_MODELS = 5

# Adjacency options
ADJ_TYPE = 'static'
CORR_THRESHOLD = 0.4

# Paths
_ROOT = Path(__file__).parent.parent.parent
with open(_ROOT / 'config.yaml') as f:
    _CFG = yaml.safe_load(f)
DATA_DIR = _ROOT / _CFG['data']['prices_dir']
MODELS_DIR = _ROOT / 'models' / 'gnnhar_paper_extended'  # NEW: Extended models directory
RESULTS_DIR = _ROOT / 'results' / 'gnnhar_paper_extended'  # NEW: Extended results directory

N_STOCKS = len(VN30_TICKERS)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def build_extended_data_split(
    close: pd.DataFrame,
    log_returns: pd.DataFrame,
    horizon: int
) -> dict:
    """
    Build EXTENDED data split addressing volatility regime shift.

    Original split (PROBLEMATIC):
        Train: 2006-2022 (3,828 samples)
        Val:   2022-2025 (958 samples)
        Test:  2026-present (97 samples)
        Issue: 4-year gap, validation doesn't represent test conditions

    Extended split (IMPROVED):
        Train: 2006-2024 (~4,900 samples) - includes 2022-2024 patterns
        Val:   2025 only (~250 samples) - more representative of test
        Test:  2026-present (97 samples) - no gap with validation

    Args:
        close: (T, N) close prices
        log_returns: (T, N) log returns
        horizon: forecast horizon

    Returns:
        dict with X_train, y_train, train_dates, X_val, y_val, val_dates, etc.
    """
    test_start = pd.Timestamp(GLOBAL_TEST_START)
    train_end = pd.Timestamp(EXTENDED_TRAIN_END)
    val_start = pd.Timestamp(EXTENDED_VAL_START)
    val_end = pd.Timestamp(EXTENDED_VAL_END)

    print(f"\n  [Extended Split Configuration]")
    print(f"  Train end:     {train_end.date()}")
    print(f"  Validation:     {val_start.date()} to {val_end.date()}")
    print(f"  Test start:    {test_start.date()}")

    # Build full dataset with stride=1 for maximum samples
    X_full, y_full, dates_full = build_static_snapshots(
        close, log_returns, horizon, stride=1,
        date_end=None  # Use all available data
    )

    # Convert to pandas Series for easier date filtering
    dates_pd = pd.to_datetime(dates_full)

    # Create masks for each split
    train_mask = dates_pd <= train_end
    val_mask = (dates_pd >= val_start) & (dates_pd <= val_end)
    test_mask = dates_pd >= test_start

    # Extract data for each split
    X_train = X_full[train_mask]
    y_train = y_full[train_mask]
    train_dates = dates_full[train_mask]

    X_val = X_full[val_mask]
    y_val = y_full[val_mask]
    val_dates = dates_full[val_mask]

    X_test = X_full[test_mask]
    y_test = y_full[test_mask]
    test_dates = dates_full[test_mask]

    # Print split summary
    print(f"\n  [Data Split Summary]")
    print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} samples)")
    print(f"  Val:   {val_dates[0].date() if len(val_dates) > 0 else 'N/A'} to {val_dates[-1].date() if len(val_dates) > 0 else 'N/A'} ({len(val_dates)} samples)")
    print(f"  Test:  {test_dates[0].date() if len(test_dates) > 0 else 'N/A'} to {test_dates[-1].date() if len(test_dates) > 0 else 'N/A'} ({len(test_dates)} samples)")

    # Calculate ESS
    ess = len(train_dates) * N_STOCKS // horizon
    print(f"  ESS = {len(train_dates)} × {N_STOCKS} / {horizon} = {ess}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'train_dates': train_dates,
        'X_val': X_val,
        'y_val': y_val,
        'val_dates': val_dates,
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': test_dates,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
        'test_start': test_start
    }


def analyze_volatility_characteristics(data_split: dict, close: pd.DataFrame, horizon: int):
    """
    Analyze volatility characteristics across train/val/test periods.

    This helps identify if the extended split successfully addresses the
    volatility regime shift issue.

    Args:
        data_split: Dictionary with train/val/test data and dates
        close: Close prices for RV computation
        horizon: Forecast horizon
    """
    from src.volatility_labels import compute_rv

    print(f"\n  [Volatility Analysis h={horizon}]")
    print(f"  {'Period':<12} {'Mean RV':>10} {'Std RV':>10} {'Range':>25} {'Samples':>8}")
    print(f"  {'-'*70}")

    periods = {}

    # Training period
    train_close = close.loc[data_split['train_dates'][0]:data_split['train_dates'][-1]]
    train_rv = compute_rv(train_close, h=horizon)
    train_mean = train_rv.mean().mean()
    train_std = train_rv.std().mean()
    train_min = train_rv.min().min()
    train_max = train_rv.max().max()

    print(f"  {'Training':<12} {train_mean:>10.6f} {train_std:>10.6f} [{train_min:.6f}, {train_max:.6f}] {len(data_split['train_dates']):>8}")
    periods['Training'] = {'mean': train_mean, 'std': train_std, 'min': train_min, 'max': train_max}

    # Validation period
    if len(data_split['val_dates']) > 0:
        val_close = close.loc[data_split['val_dates'][0]:data_split['val_dates'][-1]]
        val_rv = compute_rv(val_close, h=horizon)
        val_mean = val_rv.mean().mean()
        val_std = val_rv.std().mean()
        val_min = val_rv.min().min()
        val_max = val_rv.max().max()

        print(f"  {'Validation':<12} {val_mean:>10.6f} {val_std:>10.6f} [{val_min:.6f}, {val_max:.6f}] {len(data_split['val_dates']):>8}")
        periods['Validation'] = {'mean': val_mean, 'std': val_std, 'min': val_min, 'max': val_max}

    # Test period
    if len(data_split['test_dates']) > 0:
        test_close = close.loc[data_split['test_dates'][0]:data_split['test_dates'][-1]]
        test_rv = compute_rv(test_close, h=horizon)
        test_mean = test_rv.mean().mean()
        test_std = test_rv.std().mean()
        test_min = test_rv.min().min()
        test_max = test_rv.max().max()

        print(f"  {'Test':<12} {test_mean:>10.6f} {test_std:>10.6f} [{test_min:.6f}, {test_max:.6f}] {len(data_split['test_dates']):>8}")
        periods['Test'] = {'mean': test_mean, 'std': test_std, 'min': test_min, 'max': test_max}

    # Analyze shifts
    print(f"\n  [Volatility Shift Analysis]")

    if 'Validation' in periods and 'Test' in periods:
        train_to_val_shift = (periods['Validation']['mean'] / periods['Training']['mean'] - 1) * 100
        val_to_test_shift = (periods['Test']['mean'] / periods['Validation']['mean'] - 1) * 100
        train_to_test_shift = (periods['Test']['mean'] / periods['Training']['mean'] - 1) * 100

        print(f"  Training -> Validation: {train_to_val_shift:+.1f}%")
        print(f"  Validation -> Test:    {val_to_test_shift:+.1f}%")
        print(f"  Training -> Test:      {train_to_test_shift:+.1f}%")

        # Alert on significant shifts
        if abs(val_to_test_shift) < 15:
            print(f"  ✅ GOOD: Validation and test have similar volatility")
        elif abs(val_to_test_shift) < 30:
            print(f"  ⚠️  MODERATE: Some volatility shift between val and test")
        else:
            print(f"  ❌ CONCERN: Large volatility shift between val and test")


def build_static_adjacency_extended(
    log_returns: pd.DataFrame,
    train_end: pd.Timestamp
) -> np.ndarray:
    """
    Build static Pearson adjacency matrix (same as original).

    Using training end date from extended split for adjacency construction.
    """
    from gnn.har_graph import build_static_graph_30
    from gnn.build_graph import SECTOR_MAP

    stock_lr = log_returns[[t for t in VN30_TICKERS if t in log_returns.columns]]
    train_lr = stock_lr[stock_lr.index <= train_end]

    corr_mat = train_lr.corr()
    n = N_STOCKS
    src_list, dst_list = [], []

    for i, ti in enumerate(VN30_TICKERS):
        for j, tj in enumerate(VN30_TICKERS):
            if i >= j:
                continue
            c = float(corr_mat.loc[ti, tj]) if (ti in corr_mat.index and tj in corr_mat.index) else 0.0
            same_sect = (SECTOR_MAP.get(ti) == SECTOR_MAP.get(tj) and SECTOR_MAP.get(ti) is not None)
            if c > CORR_THRESHOLD or same_sect:
                src_list += [i, j]
                dst_list += [j, i]

    # Convert to adjacency matrix
    adj = np.zeros((n, n), dtype=np.float32)
    for s, d in zip(src_list, dst_list):
        adj[s, d] = 1.0

    # Symmetric normalize
    edge_sums = adj.sum(axis=1)
    d_sqrt_inv = np.sqrt(1.0 / (edge_sums + 1e-8))
    adj_norm = np.diag(d_sqrt_inv) @ adj @ np.diag(d_sqrt_inv)

    return adj_norm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    return eval_metrics(y_true, y_pred, include_qlike=True, include_hetero=True)


def main(horizon=None):
    """
    Main training function with extended data split.

    Args:
        horizon: Single horizon to train (e.g., 5). If None, trains all horizons.
    """
    # Determine which horizons to train
    if horizon is not None:
        train_horizons = [horizon]
        print(f"\n{'='*70}")
        print(f"  EXTENDED GNNHAR Training - SINGLE HORIZON MODE")
        print(f"  Training horizon h={horizon} only")
    else:
        train_horizons = HORIZONS
        print(f"\n{'='*70}")
        print(f"  EXTENDED GNNHAR Training")
        print(f"  Horizons: {train_horizons} | Models: {MODEL_NAMES}")

    print(f"  {'='*70}")
    print(f"  EXTENDED DATA SPLIT MODE")
    print(f"  Train through: {EXTENDED_TRAIN_END}")
    print(f"  Validation:    {EXTENDED_VAL_START} to {EXTENDED_VAL_END}")
    print(f"  Test from:     {GLOBAL_TEST_START}")
    print(f"  N_HID={N_HID} | N_EPOCHS={N_EPOCHS} | NUM_MODELS={NUM_MODELS}")
    print(f"  {'='*70}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped folder for learning curves
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    CURVES_DIR = RESULTS_DIR / f'curves_{timestamp}'
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Curves] Learning curves -> {CURVES_DIR.name}/")

    # Load data
    print("\n[Data] Loading VN30 prices...")
    close = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    log_ret = compute_log_returns(close)
    print(f"  Shape: {close.shape[0]} dates x {close.shape[1]} stocks")

    # Build adjacency using extended training end date
    train_end_ts = pd.Timestamp(EXTENDED_TRAIN_END)
    print(f"\n[Graph] Building {ADJ_TYPE} adjacency (train_end={train_end_ts.date()})...")
    adj = build_static_adjacency_extended(log_ret, train_end_ts)
    print(f"  Nodes: {N_STOCKS} | Edges: {(adj != 0).sum()}")

    # Results storage
    all_results = []

    # Per-horizon loop
    for h in train_horizons:
        print(f"\n{'#'*70}")
        print(f"  HORIZON h = {h}")
        print(f"{'#'*70}\n")

        # Build EXTENDED data split
        data_split = build_extended_data_split(close, log_ret, h)

        # Analyze volatility characteristics
        analyze_volatility_characteristics(data_split, close, h)

        # Extract data components
        X_train = data_split['X_train']
        y_train = data_split['y_train']
        X_val = data_split['X_val']
        y_val = data_split['y_val']
        X_test = data_split['X_test']
        y_test = data_split['y_test']
        test_dates = data_split['test_dates']

        # Skip if insufficient data
        if len(X_train) < 100 or len(X_val) < 20 or len(X_test) < 10:
            print(f"  [SKIP] Insufficient data for h={h}")
            continue

        # Per-model loop
        for model_name in MODEL_NAMES:
            print(f"\n  [{model_name}] Training ensemble...")

            try:
                # Initialize trainer
                trainer = EnsembleTrainer(
                    model_name=model_name,
                    n_hid=N_HID,
                    n_epochs=N_EPOCHS,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    patience=PATIENCE,
                    device='auto'
                )

                # Train ensemble
                trainer.train(
                    X_train, y_train,
                    X_val, y_val,
                    adj,
                    num_models=NUM_MODELS,
                    verbose=False  # Reduce verbosity
                )

                # Predict on test set
                test_pred = trainer.predict(X_test, adj, selected=None)

                # Evaluate per-stock metrics
                print(f"\n  [{model_name}] Test Results:")
                print(f"   {'Ticker':<8} {'R2':>8} {'MAE':>9} {'RMSE':>10}")
                print(f"   {'-'*40}")

                for si, ticker in enumerate(VN30_TICKERS):
                    y_true_stock = y_test[:, si]
                    y_pred_stock = test_pred[:, si]

                    # Remove NaN values
                    valid = ~np.isnan(y_true_stock) & ~np.isnan(y_pred_stock)
                    if valid.sum() < 2:
                        continue

                    y_true_valid = y_true_stock[valid]
                    y_pred_valid = y_pred_stock[valid]

                    # Compute metrics
                    metrics = compute_metrics(y_true_valid, y_pred_valid)

                    print(f"   {ticker:<8} {metrics['r2']:>+7.4f} {metrics['mae']:>9.5f} {metrics['rmse']:>10.5f}")

                    # Store results
                    result_row = {
                        'horizon': h,
                        'model': model_name,
                        'ticker': ticker,
                        'n_test': valid.sum(),
                        **{f'gnn_{k}': v for k, v in metrics.items()}
                    }
                    all_results.append(result_row)

            except Exception as e:
                print(f"  [ERROR] {model_name} training failed: {e}")
                continue

    # Save combined results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        csv_out = RESULTS_DIR / f'extended_results_{timestamp}.csv'
        results_df.to_csv(csv_out, index=False)
        print(f"\n[Results] Saved to {csv_out}")

        # Print summary
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        for h in train_horizons:
            print(f"\n  Horizon h={h}:")
            h_df = results_df[results_df.horizon == h]
            for model_name in MODEL_NAMES:
                m_df = h_df[h_df.model == model_name]
                if m_df.empty:
                    continue
                avg_r2 = m_df['gnn_r2'].mean()
                n_stocks = len(m_df)
                n_positive = (m_df['gnn_r2'] > 0).sum()
                print(f"    {model_name:<10} R2={avg_r2:>+7.4f} | Positive: {n_positive:2d}/{n_stocks:2d} stocks")
    else:
        print(f"\n[Warning] No results to save")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended GNNHAR training with improved data split')
    parser.add_argument('--horizon', type=int, choices=[1, 5, 10, 20],
                       help='Single horizon to train (if not specified, trains all)')

    args = parser.parse_args()

    try:
        if args.horizon:
            main(horizon=args.horizon)
        else:
            main()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise