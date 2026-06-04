"""
Main training script for GNNHAR paper replication.

Paper: "Forecasting Realized Volatility with Spillover Effects:
         Perspectives from Graph Neural Networks" (IJF 2024)

This script implements the full training pipeline:
  1. Load VN30 data (close prices, returns)
  2. For each horizon h in [1, 5, 10, 20]:
     - Build HAR features and RV targets
     - Split data: train/val/test (R6: test from 2026-01-01)
     - Build adjacency (static Pearson or rolling GLASSO)
     - Train ensemble of models (HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L)
     - Evaluate per-stock metrics vs baseline

Output structure:
  models/gnnhar_paper/
    h{H}/
      {MODEL_NAME}/
        ensemble_*.pt  -- trained models
  results/gnnhar_paper/
    predictions_h{H}.csv -- test predictions per stock
    metrics_h{H}.csv    -- R2, MAE, RMSE per stock
    summary.csv        -- cross-model summary

Constraints:
  R1: SISO (one model per horizon, HORIZONS=[1,5,10,20])
  R2: Print per-horizon loss every epoch + learning curve
  R3: Print data split (dates + counts + ESS) before training
  R6: Global test from 2026-01-01, train/val 80/20 from pre-2026
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer
from gnn.gnnhar_paper.glasso_adjacency import glasso_adjacency, RollingAdjacency
from gnn.gnnhar_paper.rolling_datasets import build_static_snapshots
from baselines.har_rv_baseline import fit_har, predict_har

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

HORIZONS = [1, 5, 10, 20]
GLOBAL_TEST_START = "2026-01-01"
TRAIN_VAL_SPLIT = 0.8

# Model variants to train
MODEL_NAMES = ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']

# Training hyperparameters (matching paper where applicable)
N_HID = 16              # Hidden dimension (paper uses 9, we use 16 for ESS)
N_EPOCHS = 200          # DEV: Test with 200 epochs
LR = 1e-3               # Learning rate
WEIGHT_DECAY = 1e-3     # L2 regularization
PATIENCE = 40           # DEV: Adjusted for 200 epochs
NUM_MODELS = 5          # Ensemble size (paper default)

# Adjacency options
ADJ_TYPE = 'static'     # OPTIONS: 'glasso' (slower, paper method) or 'static' (faster, Pearson correlation)
CORR_THRESHOLD = 0.4    # For static Pearson adjacency

# Paths
_ROOT = Path(__file__).parent.parent.parent
with open(_ROOT / 'config.yaml') as f:
    _CFG = yaml.safe_load(f)
DATA_DIR = _ROOT / _CFG['data']['prices_dir']
MODELS_DIR = _ROOT / 'models' / 'gnnhar_paper'
RESULTS_DIR = _ROOT / 'results' / 'gnnhar_paper'

N_STOCKS = len(VN30_TICKERS)

# ==============================================================================
# UTILITIES
# ==============================================================================


def build_static_adjacency(
    log_returns: pd.DataFrame,
    train_end: pd.Timestamp,
) -> np.ndarray:
    """
    Build static Pearson adjacency (like current GraphSAGE implementation).
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

    # Convert to adjacency matrix (NO SELF-LOOPS - paper excludes them)
    adj = np.zeros((n, n), dtype=np.float32)
    for s, d in zip(src_list, dst_list):
        adj[s, d] = 1.0

    # Symmetric normalize
    edge_sums = adj.sum(axis=1)
    d_sqrt_inv = np.sqrt(1.0 / (edge_sums + 1e-8))
    adj_norm = np.diag(d_sqrt_inv) @ adj @ np.diag(d_sqrt_inv)

    return adj_norm


def save_learning_curve(
    history: dict,
    horizon: int,
    model_name: str,
    path: Path,
) -> None:
    """Save training/validation loss curve as PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train']) + 1)
    ax.plot(epochs, history['train'], label='Train', color='steelblue')
    ax.plot(epochs, history['val'], label='Val', color='darkorange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{model_name} h={horizon}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive evaluation metrics for volatility forecasting.

    Imports from evaluation.py which implements:
    - Standard metrics: R², MAE, RMSE
    - QLIKE metric (Patton 2011) - robust to noise
    - Heteroskedastic metrics: HMSE, HMAE (Lopez de Prado 2018)

    Returns:
        Dictionary with all metrics (lower is better except R²)
    """
    # Use absolute import to avoid module name conflicts
    from gnn.gnnhar_paper.evaluation import compute_metrics as eval_metrics
    return eval_metrics(y_true, y_pred, include_qlike=True, include_hetero=True)


def print_split(h: int, train_dates, val_dates, test_dates, n_train, n_val, n_test):
    """Print data split (R3)."""
    ess = n_train * N_STOCKS // h
    print(f"\n{'='*62}")
    print(f"  DATA SPLIT  h={h}")
    print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()} ({n_train} snaps)")
    print(f"  Val  : {val_dates[0].date()} -> {val_dates[-1].date()} ({n_val} snaps)")
    print(f"  Test : {test_dates[0].date()} -> {test_dates[-1].date()} ({n_test} snaps)")
    print(f"  ESS = {n_train}x{N_STOCKS}/{h} = {ess}")
    print(f"{'='*62}\n")


def verify_temporal_split(train_dates, val_dates, test_dates):
    """
    Verify temporal ordering of train/val/test split (Issue #6).

    Ensures no data leakage by checking:
    1. All date arrays are monotonically increasing
    2. Train end date < Val start date
    3. Val end date < Test start date

    Raises:
        AssertionError: If temporal ordering is violated
    """
    # Check monotonic increasing
    assert train_dates.is_monotonic_increasing, "Train dates not time-ordered!"
    assert val_dates.is_monotonic_increasing, "Val dates not time-ordered!"
    assert test_dates.is_monotonic_increasing, "Test dates not time-ordered!"

    # Check temporal boundaries
    train_end = train_dates[-1]
    val_start = val_dates[0]
    val_end = val_dates[-1]
    test_start = test_dates[0]

    assert train_end < val_start, f"Train/Val leakage: train_end={train_end} >= val_start={val_start}"
    assert val_end < test_start, f"Val/Test leakage: val_end={val_end} >= test_start={test_start}"

    print(f"  [OK] Temporal split verified: train < val < test")


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================


def main(horizon=None):
    """
    Main training function.

    Args:
        horizon: Single horizon to train (e.g., 5). If None, trains all horizons.
    """
    # Determine which horizons to train
    if horizon is not None:
        train_horizons = [horizon]
        print(f"\n{'='*62}")
        print(f"  GNNHAR Paper Replication - SINGLE HORIZON MODE")
        print(f"  Training horizon h={horizon} only")
    else:
        train_horizons = HORIZONS
        print(f"\n{'='*62}")
        print(f"  GNNHAR Paper Replication")
        print(f"  Horizons: {HORIZONS} | Models: {MODEL_NAMES}")
    print(f"\n{'='*62}")
    print(f"  GNNHAR Paper Replication")
    print(f"  Horizons: {HORIZONS} | Models: {MODEL_NAMES}")
    print(f"  N_HID={N_HID} | N_EPOCHS={N_EPOCHS} | NUM_MODELS={NUM_MODELS}")
    print(f"  ADJ_TYPE={ADJ_TYPE}")
    print(f"{'='*62}\n")
    print(f"  Training horizons: {train_horizons}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped folder for learning curves
    # Format: curves_DD_MM_YYYY_HH_MM
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M')
    CURVES_DIR = RESULTS_DIR / f'curves_{timestamp}'
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Curves] Learning curves will be saved to: {CURVES_DIR.name}/")

    # Load data
    print("[Data] Loading VN30 prices...")
    close = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    log_ret = compute_log_returns(close)
    print(f"  Shape: {close.shape[0]} dates x {close.shape[1]} stocks")

    test_ts = pd.Timestamp(GLOBAL_TEST_START)

    # Determine train_end for static adjacency (use h=1, most data)
    X_pre, _, dates_pre = build_static_snapshots(
        close, log_ret, horizon=1, stride=20,
        date_end=test_ts - pd.Timedelta(days=1),
    )
    n_train_total = int(len(dates_pre) * TRAIN_VAL_SPLIT)
    train_end_ts = dates_pre[n_train_total - 1]

    # Build adjacency
    print(f"\n[Graph] Building {ADJ_TYPE} adjacency (train_end={train_end_ts.date()})...")
    if ADJ_TYPE == 'static':
        adj = build_static_adjacency(log_ret, train_end_ts)
        print(f"  Nodes: {N_STOCKS} | Edges: {(adj != 0).sum()}")
    elif ADJ_TYPE == 'glasso':
        stock_lr = log_ret[[t for t in VN30_TICKERS if t in log_ret.columns]]
        train_lr = stock_lr[stock_lr.index <= train_end_ts]
        adj_df = glasso_adjacency(train_lr, verbose=True)
        adj = adj_df.values
    else:
        raise ValueError(f"Unknown ADJ_TYPE: {ADJ_TYPE}")

    # Results storage
    all_results = []

    # Per-horizon loop
    for h in train_horizons:
        print(f"\n{'#'*62}")
        print(f"  HORIZON h = {h}")
        print(f"{'#'*62}\n")

        # Build snapshots for this horizon
        stride = max(5, min(h, 20))  # Adaptive stride: at least 5, cap at 20
        X_pre, y_pre, dates_pre = build_static_snapshots(
            close, log_ret, h, stride,
            date_end=test_ts - pd.Timedelta(days=1),
        )

        n_pre = len(dates_pre)
        n_train = int(n_pre * TRAIN_VAL_SPLIT)
        n_val = n_pre - n_train

        train_dates = dates_pre[:n_train]
        val_dates = dates_pre[n_train:]

        X_train, y_train = X_pre[:n_train], y_pre[:n_train]
        X_val, y_val = X_pre[n_train:], y_pre[n_train:]

        # Test set (stride=1 for full evaluation)
        X_full, y_full, dates_full = build_static_snapshots(
            close, log_ret, h, stride=1,
        )
        # Filter test period
        test_mask = dates_full >= test_ts
        X_test, y_test, test_dates = X_full[test_mask], y_full[test_mask], dates_full[test_mask]
        n_test = len(test_dates)

        print_split(h, train_dates, val_dates, test_dates, n_train, n_val, n_test)
        # FIX: Verify temporal ordering to prevent data leakage (Issue #6)
        verify_temporal_split(train_dates, val_dates, test_dates)

        # QLIKE training: Use raw RV targets (no normalization)
        # QLIKE is scale-invariant and designed for log-normal RV distribution
        # High-vol stocks won't dominate due to QLIKE's ratio-based formulation
        print(f"[Preprocess] Using raw RV targets for QLIKE training...")

        # No normalization needed - QLIKE handles scale differences via ratios

        # Pre-compute RV for this horizon (for HAR baseline)
        rv_h = compute_rv(close, h=h)

        # HAR baseline per-stock (for comparison)
        har_coeffs = {}
        for ticker in VN30_TICKERS:
            if ticker in rv_h.columns:
                try:
                    har_coeffs[ticker] = fit_har(rv_h[ticker].dropna(), train_end_ts)
                except Exception:
                    pass

        # Per-model loop
        for model_name in MODEL_NAMES:
            print(f"\n[{model_name}] Training ensemble ({NUM_MODELS} models)...")

            trainer = EnsembleTrainer(
                model_name=model_name,
                n_hid=N_HID,
                n_epochs=N_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                batch_size=128,
                patience=PATIENCE,
            )

            trainer.train(X_train, y_train, X_val, y_val, adj,
                         num_models=NUM_MODELS, verbose=True)

            # Screen ensemble
            selected = trainer.screen_ensemble(percentile=50)
            print(f"  Selected {len(selected)}/{NUM_MODELS} models after screening")

            # Save ensemble
            model_dir = MODELS_DIR / f'h{h}' / model_name
            trainer.save(model_dir)

            # Save learning curves to timestamped folder
            for i, hist in enumerate(trainer.train_histories):
                curve_path = CURVES_DIR / f'curve_h{h}_{model_name}_m{i}.png'
                save_learning_curve(hist, h, model_name, curve_path)
            print(f"  Curves -> {CURVES_DIR.name}/")

            # Test evaluation
            if n_test == 0:
                print(f"  [WARN] No test data for h={h}")
                continue

            print(f"\n  Test evaluation (n={n_test}):")
            print(f"   {'Ticker':<8} {'R2':>8} {'MAE':>9} {'RMSE':>10} {'QLIKE':>10} {'HMSE':>9} {'HMAE':>9}")
            print(f"   {'-'*70}")

            # Predict (predictions are in raw RV scale, same as targets)
            pred = trainer.predict(X_test, adj, selected=selected)

            # No inverse transform needed - QLIKE trained on raw RV

            # Per-stock metrics
            for si, ticker in enumerate(VN30_TICKERS):
                row = {
                    'horizon': h,
                    'model': model_name,
                    'ticker': ticker,
                    'n_test': n_test,
                    'n_ensemble': len(selected),
                }

                # Ground truth
                y_true = rv_h[ticker].reindex(test_dates).values if ticker in rv_h.columns else None
                if y_true is not None:
                    valid = ~np.isnan(y_true)
                    if valid.sum() >= 2:
                        metrics = compute_metrics(y_true[valid], pred[valid, si])
                        row.update({f'gnn_{k}': v for k, v in metrics.items()})
                        print(f"   {ticker:<8} {metrics['r2']:>8.4f} {metrics['mae']:>9.5f} {metrics['rmse']:>10.5f} {metrics.get('qlike', 0):>10.6f} {metrics.get('hmse', 0):>9.6f} {metrics.get('hmae', 0):>9.6f}")

                # HAR baseline
                try:
                    if ticker in har_coeffs:
                        har_pred = predict_har(rv_h[ticker].dropna(), har_coeffs[ticker], test_ts)
                        common = test_dates.intersection(har_pred.index)
                        if len(common) >= 2:
                            yt = rv_h[ticker].reindex(common).values
                            yp = har_pred.reindex(common).values
                            vm = ~(np.isnan(yt) | np.isnan(yp))
                            if vm.sum() >= 2:
                                har_metrics = compute_metrics(yt[vm], yp[vm])
                                row.update({f'har_{k}': v for k, v in har_metrics.items()})
                except Exception:
                    pass

                row['delta_r2'] = row.get('gnn_r2', np.nan) - row.get('har_r2', np.nan)
                all_results.append(row)

    # Save combined results
    results_df = pd.DataFrame(all_results)
    csv_out = RESULTS_DIR / 'gnnhar_paper_results.csv'
    results_df.to_csv(csv_out, index=False)
    print(f"\n[Results] Saved to {csv_out}")

    # Print summary
    print(f"\n{'='*62}")
    print(f"  SUMMARY")
    print(f"{'='*62}")
    for h in HORIZONS:
        print(f"\n  Horizon h={h}:")
        h_df = results_df[results_df.horizon == h]
        for model_name in MODEL_NAMES:
            m_df = h_df[h_df.model == model_name]
            if m_df.empty:
                continue
            avg_r2 = m_df['gnn_r2'].mean() if 'gnn_r2' in m_df.columns else np.nan
            avg_har_r2 = m_df['har_r2'].mean() if 'har_r2' in m_df.columns else np.nan
            n_beat = (m_df['delta_r2'] > 0).sum() if 'delta_r2' in m_df.columns else 0
            print(f"    {model_name:<10} R2={avg_r2:>+7.4f} | HAR={avg_har_r2:>+7.4f} | beat HAR: {n_beat:2d}/{N_STOCKS}")

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN-HAR model')
    parser.add_argument('--horizon', type=int, choices=[1, 5, 10, 20],
                        help='Single horizon to train (default: all horizons)')
    args = parser.parse_args()
    main(horizon=args.horizon)
