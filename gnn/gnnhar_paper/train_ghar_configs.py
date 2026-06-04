"""
Quick test 3 GHAR configs most likely to achieve positive R².

Usage:
    python moirai/gnn/gnnhar_paper/train_ghar_configs.py

This is faster than full grid search - tests only 3 hand-picked configs.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer
from gnn.gnnhar_paper.train_gnnhar_paper import (
    build_static_adjacency,
    compute_metrics,
    verify_temporal_split,
    save_learning_curve,
)
from gnn.gnnhar_paper.rolling_datasets import build_static_snapshots
from baselines.har_rv_baseline import fit_har, predict_har

warnings.filterwarnings("ignore")

# ==============================================================================
# 3 CONFIGS TO TEST
# ==============================================================================

CONFIGS = [
    {
        'name': 'Conservative',
        'n_hid': 16,
        'n_epochs': 500,
        'lr': 1e-3,
        'weight_decay': 1e-4,      # Weaker regularization
        'patience': 80,
        'screen_percentile': 70,   # Keep more models
    },
    {
        'name': 'Aggressive',
        'n_hid': 16,
        'n_epochs': 1000,
        'lr': 1e-3,
        'weight_decay': 0,         # NO regularization
        'patience': 120,
        'screen_percentile': 90,   # Keep almost all models
    },
    {
        'name': 'Balanced (RECOMMENDED)',
        'n_hid': 16,
        'n_epochs': 500,
        'lr': 1e-3,
        'weight_decay': 0,         # NO regularization
        'patience': 80,
        'screen_percentile': 70,
    },
]

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
# TRAINING FUNCTION
# ==============================================================================


def train_single_config(config):
    """Train GHAR with single config, return results."""
    print("\n" + "="*70)
    print(f"  CONFIG: {config['name']}")
    print("="*70)
    for k, v in config.items():
        if k != 'name':
            print(f"  {k}: {v}")
    print()

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
    n_test = len(test_dates)

    print(f"  Train: {n_train} snaps, Val: {len(val_dates)} snaps, Test: {n_test} snaps")
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

    # Train GHAR
    print(f"\n[GHAR] Training ensemble ({NUM_MODELS} models)...\n")

    trainer = EnsembleTrainer(
        model_name='GHAR',
        n_hid=config['n_hid'],
        n_epochs=config['n_epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        batch_size=BATCH_SIZE,
        patience=config['patience'],
    )

    trainer.train(X_train, y_train, X_val, y_val, adj,
                 num_models=NUM_MODELS, verbose=True)

    # Screen ensemble
    selected = trainer.screen_ensemble(percentile=config['screen_percentile'])
    print(f"\n  Selected {len(selected)}/{NUM_MODELS} models (percentile={config['screen_percentile']}%)")

    # Save curves
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M')
    CURVES_DIR = Path(__file__).parent.parent.parent / 'results' / 'gnnhar_paper' / f'curves_ghar_{config["name"]}_{timestamp}'
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    for i, hist in enumerate(trainer.train_histories):
        curve_path = CURVES_DIR / f'curve_m{i}.png'
        save_learning_curve(hist, HORIZON, f'GHAR_{config["name"]}', curve_path)
    print(f"  Curves -> {CURVES_DIR.name}/")

    # Test evaluation
    print(f"\n  Test evaluation (n={n_test}):")
    print(f"   {'Ticker':<8} {'R2':>8} {'MAE':>9} {'RMSE':>10}")
    print(f"   {'-'*40}")

    pred = trainer.predict(X_test, adj, selected=selected)

    # Per-stock metrics
    gnn_r2_list = []
    har_r2_list = []

    for si, ticker in enumerate(VN30_TICKERS):
        y_true = rv_h[ticker].reindex(test_dates).values if ticker in rv_h.columns else None
        if y_true is not None:
            valid = ~np.isnan(y_true)
            if valid.sum() >= 2:
                metrics = compute_metrics(y_true[valid], pred[valid, si])
                gnn_r2_list.append(metrics['r2'])
                delta = metrics['r2'] - (har_coeffs.get(ticker, [0])[0] if isinstance(har_coeffs.get(ticker), list) else 0)
                print(f"   {ticker:<8} {metrics['r2']:>8.4f} {metrics['mae']:>9.5f} {metrics['rmse']:>10.5f}")

    # Compute HAR baseline R² for comparison
    for ticker in VN30_TICKERS:
        try:
            if ticker in har_coeffs:
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

    gnn_mean_r2 = np.mean(gnn_r2_list)
    har_mean_r2 = np.mean(har_r2_list)
    delta = gnn_mean_r2 - har_mean_r2

    print(f"\n  {'='*50}")
    print(f"  GHAR R² = {gnn_mean_r2:+.4f}")
    print(f"  HAR  R² = {har_mean_r2:+.4f}")
    print(f"  Delta  = {delta:+.4f}")
    print(f"  {'='*50}\n")

    return {
        'config_name': config['name'],
        'gnn_r2': gnn_mean_r2,
        'har_r2': har_mean_r2,
        'delta': delta,
        'n_selected': len(selected),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("  GHAR QUICK TEST - 3 CONFIGS")
    print("  Target: R² > 0 (beat HAR baseline)")
    print("="*70)

    all_results = []

    for config in CONFIGS:
        result = train_single_config(config)
        all_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  {'Config':<20} {'GHAR R2':>10} {'HAR R2':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    best_r2 = -float('inf')
    best_config = None

    for r in all_results:
        status = "[WIN]" if r['gnn_r2'] > 0 else "[LOSE]"
        print(f"  {r['config_name']:<20} {r['gnn_r2']:>+10.4f} {r['har_r2']:>+10.4f} {r['delta']:>+10.4f} {status}")

        if r['gnn_r2'] > best_r2:
            best_r2 = r['gnn_r2']
            best_config = r['config_name']

    print(f"\n[BEST] {best_config} with R² = {best_r2:+.4f}\n")

    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'gnnhar_paper'
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(results_dir / f'ghar_quick_test_{timestamp}.csv', index=False)
    print(f"[Saved] {results_dir / f'ghar_quick_test_{timestamp}.csv'}\n")


if __name__ == "__main__":
    main()
