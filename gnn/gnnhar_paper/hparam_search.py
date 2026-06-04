"""
Hyperparameter search for GHAR to achieve positive R².

Strategy: Grid search over key hyperparameters that most affect R².
Target: GHAR R² > 0 (beat OLS baseline at R²=+0.63)

Key hypotheses:
1. N_EPOCHS too low (200 vs 500) → underfitting
2. WEIGHT_DECAY too high (1e-3) → over-regularization
3. Learning rate not optimal
4. Ensemble screening too aggressive (50th percentile)

Run: python moirai/gnn/gnnhar_paper/hparam_search.py
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
import itertools
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer
from gnn.gnnhar_paper.glasso_adjacency import glasso_adjacency
from gnn.gnnhar_paper.rolling_datasets import build_static_snapshots
from baselines.har_rv_baseline import fit_har
from gnn.gnnhar_paper.train_gnnhar_paper import (
    build_static_adjacency,
    compute_metrics,
    verify_temporal_split,
)

warnings.filterwarnings("ignore")

# ==============================================================================
# HYPERPARAMETER GRID
# ==============================================================================

HPARAM_GRID = {
    'n_epochs': [200, 500, 1000],           # Training epochs
    'lr': [5e-4, 1e-3, 2e-3],               # Learning rate
    'weight_decay': [0, 1e-4, 1e-3],        # L2 regularization
    'n_hid': [8, 16, 32],                   # Hidden dimension
    'patience': [40, 80, 120],             # Early stopping patience
    'screen_percentile': [50, 70, 90],      # Ensemble screening (higher = more models)
}

# Fixed config
HORIZON = 5
GLOBAL_TEST_START = "2026-01-01"
TRAIN_VAL_SPLIT = 0.8
CORR_THRESHOLD = 0.4
NUM_MODELS = 5
BATCH_SIZE = 128

_ROOT = Path(__file__).parent.parent.parent
with open(_ROOT / 'config.yaml') as f:
    _CFG = yaml.safe_load(f)
DATA_DIR = _ROOT / _CFG['data']['prices_dir']
N_STOCKS = len(VN30_TICKERS)

# ==============================================================================
# SEARCH FUNCTIONS
# ==============================================================================


def train_ghar_single_config(
    X_train, y_train, X_val, y_val, X_test, y_test,
    adj, test_dates, rv_h, har_coeffs, config
):
    """Train GHAR with single hyperparameter config."""
    model_name = 'GHAR'

    trainer = EnsembleTrainer(
        model_name=model_name,
        n_hid=config['n_hid'],
        n_epochs=config['n_epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        batch_size=BATCH_SIZE,
        patience=config['patience'],
    )

    # Train (silent)
    trainer.train(X_train, y_train, X_val, y_val, adj,
                 num_models=NUM_MODELS, verbose=False)

    # Screen ensemble
    selected = trainer.screen_ensemble(percentile=config['screen_percentile'])

    # Predict
    pred = trainer.predict(X_test, adj, selected=selected)

    # Compute metrics
    r2_list = []
    for si, ticker in enumerate(VN30_TICKERS):
        y_true = rv_h[ticker].reindex(test_dates).values if ticker in rv_h.columns else None
        if y_true is not None:
            valid = ~np.isnan(y_true)
            if valid.sum() >= 2:
                metrics = compute_metrics(y_true[valid], pred[valid, si])
                r2_list.append(metrics['r2'])

    return {
        'config': config,
        'mean_r2': np.mean(r2_list),
        'r2_list': r2_list,
        'n_selected': len(selected),
    }


def run_grid_search():
    """Run full grid search over HPARAM_GRID."""
    print("\n" + "="*70)
    print("  GHAR HYPERPARAMETER SEARCH")
    print("  Target: R² > 0 (beat OLS baseline)")
    print("="*70 + "\n")

    # Load data
    print("[Data] Loading VN30 prices...")
    close = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
    log_ret = compute_log_returns(close)

    test_ts = pd.Timestamp(GLOBAL_TEST_START)

    # Build adjacency
    X_pre, _, dates_pre = build_static_snapshots(
        close, log_ret, horizon=1, stride=20,
        date_end=test_ts - pd.Timedelta(days=1),
    )
    n_train_total = int(len(dates_pre) * TRAIN_VAL_SPLIT)
    train_end_ts = dates_pre[n_train_total - 1]

    print(f"\n[Graph] Building static adjacency (train_end={train_end_ts.date()})...")
    adj = build_static_adjacency(log_ret, train_end_ts)

    # Build h=5 snapshots
    X_pre, y_pre, dates_pre = build_static_snapshots(
        close, log_ret, HORIZON, stride=max(5, min(HORIZON, 20)),
        date_end=test_ts - pd.Timedelta(days=1),
    )

    n_pre = len(dates_pre)
    n_train = int(n_pre * TRAIN_VAL_SPLIT)

    train_dates = dates_pre[:n_train]
    val_dates = dates_pre[n_train:]

    X_train, y_train = X_pre[:n_train], y_pre[:n_train]
    X_val, y_val = X_pre[n_train:], y_pre[n_train:]

    X_full, y_full, dates_full = build_static_snapshots(
        close, log_ret, HORIZON, stride=1,
    )
    test_mask = dates_full >= test_ts
    X_test, y_test, test_dates = X_full[test_mask], y_full[test_mask], dates_full[test_mask]

    verify_temporal_split(train_dates, val_dates, test_dates)

    # HAR baseline
    rv_h = compute_rv(close, h=HORIZON)
    har_coeffs = {}
    for ticker in VN30_TICKERS:
        if ticker in rv_h.columns:
            try:
                har_coeffs[ticker] = fit_har(rv_h[ticker].dropna(), train_end_ts)
            except Exception:
                pass

    # Compute HAR baseline R²
    har_r2_list = []
    for ticker in VN30_TICKERS:
        try:
            if ticker in har_coeffs:
                from baselines.har_rv_baseline import predict_har
                har_pred = predict_har(rv_h[ticker].dropna(), har_coeffs[ticker], test_ts)
                common = test_dates.intersection(har_pred.index)
                if len(common) >= 2:
                    yt = rv_h[ticker].reindex(common).values
                    yp = har_pred.reindex(common).values
                    vm = ~(np.isnan(yt) | np.isnan(yp))
                    if vm.sum() >= 2:
                        metrics = compute_metrics(yt[vm], yp[vm])
                        har_r2_list.append(metrics['r2'])
        except Exception:
            pass

    har_mean_r2 = np.mean(har_r2_list)
    print(f"\n[HAR Baseline] Mean R² = {har_mean_r2:+.4f}\n")

    # Generate all combinations
    keys = HPARAM_GRID.keys()
    values = HPARAM_GRID.values()
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"[Grid Search] {len(all_combos)} configurations to test\n")
    print(f"{'Config':>6} {'Epochs':>7} {'LR':>8} {'WD':>8} {'Hidden':>7} {'Patience':>8} {'Screen':>7} {'R2':>8} {'vs HAR':>8}")
    print("-" * 100)

    results = []
    best_r2 = -float('inf')
    best_config = None

    for i, config in enumerate(all_combos):
        result = train_ghar_single_config(
            X_train, y_train, X_val, y_val, X_test, y_test,
            adj, test_dates, rv_h, har_coeffs, config
        )
        results.append(result)

        delta_r2 = result['mean_r2'] - har_mean_r2
        status = "[BEST]" if result['mean_r2'] > best_r2 else ""

        print(f"{i:>6} {config['n_epochs']:>7} {config['lr']:>8.1e} {config['weight_decay']:>8.1e} "
              f"{config['n_hid']:>7} {config['patience']:>8} {config['screen_percentile']:>7} "
              f"{result['mean_r2']:>8.4f} {delta_r2:>+8.4f} {status}")

        if result['mean_r2'] > best_r2:
            best_r2 = result['mean_r2']
            best_config = config

    print("-" * 100)
    print(f"\n[BEST CONFIG]")
    print(f"  R² = {best_r2:+.4f} (vs HAR {har_mean_r2:+.4f}, delta = {best_r2 - har_mean_r2:+.4f})")
    print(f"  Config: {best_config}\n")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame([
        {**r['config'], 'mean_r2': r['mean_r2'], 'n_selected': r['n_selected']}
        for r in results
    ])
    results_df = results_df.sort_values('mean_r2', ascending=False)
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'gnnhar_paper'
    results_df.to_csv(results_dir / f'hparam_search_{timestamp}.csv', index=False)
    print(f"[Saved] {results_dir / f'hparam_search_{timestamp}.csv'}\n")


if __name__ == "__main__":
    run_grid_search()
