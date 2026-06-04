"""
HAR-RV baseline: Corsi (2009) Heterogeneous Autoregressive model for Realized Volatility.

Model: RV_t = α + β_d·RV_{t-1} + β_w·RV_{t-5}^(5) + β_m·RV_{t-22}^(22) + ε
where RV^(k) = mean(RV_{t-1},...,RV_{t-k}) (weekly and monthly averages)

Fitted via OLS per stock on training data.  No re-fit on test (true out-of-sample).

Features:
- Multi-horizon support: h=1,5,10,20
- Validation split: 80/20 from pre-2026 data
- Comprehensive metrics: R2, MAE, RMSE per stock and aggregate
- ESS reporting for data sufficiency
- Per-horizon evaluation with CSV output

Usage:
    python baselines/har_rv_baseline.py

Output:
    - results/baselines/har_baseline_metrics_[timestamp].csv
    - results/baselines/har_baseline_summary_[timestamp].txt
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import json

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_rv
from gnn.build_graph import VN30_TICKERS

# Multi-horizon configuration (CONSTRAINTS.md R1)
HORIZONS = [1, 5, 10, 20]


def build_har_features(rv: pd.Series) -> pd.DataFrame:
    """
    Build HAR feature matrix: [RV_{t-1}, RV^(5)_{t}, RV^(22)_{t}].

    Uses 22-day monthly average to match GNNHAR1L implementation.

    Args:
        rv: RV series (indexed by date)

    Returns:
        DataFrame with columns [const, RV_d, RV_w, RV_m]
    """
    rv_d  = rv.shift(1)                                  # daily lag: RV_{t-1}
    rv_w  = rv.shift(1).rolling(5,  min_periods=5).mean()  # weekly avg: mean(RV_{t-5:t-1})
    rv_m  = rv.shift(1).rolling(22, min_periods=22).mean() # monthly avg: mean(RV_{t-22:t-1})

    features = pd.DataFrame({
        "const": 1.0,
        "RV_d":  rv_d,
        "RV_w":  rv_w,
        "RV_m":  rv_m,
    }, index=rv.index)
    return features


def fit_har(
    rv: pd.Series,
    train_end: pd.Timestamp,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, Dict]:
    """
    OLS fit on training data with validation split.

    Splits pre-test data into 80/20 train/val (CONSTRAINTS.md R6).

    Args:
        rv: RV series
        train_end: End date for training period (exclusive)
        val_ratio: Validation split ratio (default 0.2)

    Returns:
        coeffs: Coefficient vector [α, β_d, β_w, β_m]
        split_info: Dict with train/val date ranges and sizes
    """
    features = build_har_features(rv)
    df = pd.concat([features, rv.rename("target")], axis=1).dropna()

    # Split: pre-test data -> train 80%, val 20%
    pre_test = df[df.index < train_end]
    n_val = int(len(pre_test) * val_ratio)

    val = pre_test.iloc[-n_val:]
    train = pre_test.iloc[:-n_val]

    X_train = train[["const", "RV_d", "RV_w", "RV_m"]].values
    y_train = train["target"].values

    # OLS: β = (X'X)^{-1} X'y
    coeffs, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    split_info = {
        'train_start': str(train.index[0].date()),
        'train_end': str(train.index[-1].date()),
        'train_samples': len(train),
        'val_start': str(val.index[0].date()) if len(val) > 0 else 'N/A',
        'val_end': str(val.index[-1].date()) if len(val) > 0 else 'N/A',
        'val_samples': len(val),
    }

    return coeffs, split_info


def predict_har(
    rv: pd.Series,
    coeffs: np.ndarray,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[pd.Series, Dict]:
    """
    Out-of-sample predictions on test set with metrics computation.

    Args:
        rv: RV series
        coeffs: HAR coefficients from training
        test_start: Test period start date
        test_end: Test period end date

    Returns:
        pred: Predictions Series
        metrics: Dict with R2, MAE, RMSE
    """
    features = build_har_features(rv)
    df = pd.concat([features, rv.rename("target")], axis=1)

    # Filter to test period
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    test = df[test_mask].dropna()

    if len(test) == 0:
        empty_pred = pd.Series([], index=pd.DatetimeIndex([]), name=rv.name)
        empty_metrics = {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'n_samples': 0}
        return empty_pred, empty_metrics

    X = test[["const", "RV_d", "RV_w", "RV_m"]].values
    y_true = test["target"].values

    # Predictions
    pred = X @ coeffs
    pred = np.maximum(pred, 0.0)  # RV cannot be negative

    # Compute metrics
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mae = np.mean(np.abs(y_true - pred))
    rmse = np.sqrt(np.mean((y_true - pred) ** 2))

    metrics = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'n_samples': len(test),
        'test_start': str(test.index[0].date()),
        'test_end': str(test.index[-1].date()),
    }

    pred_series = pd.Series(pred, index=test.index, name=rv.name)

    return pred_series, metrics


def compute_ess(rv: pd.Series, horizon: int) -> int:
    """
    Compute Effective Sample Size (ESS) for RV data.

    ESS = N_raw / horizon (Lopez de Prado 2018)

    Args:
        rv: RV series
        horizon: Forecast horizon

    Returns:
        ESS value
    """
    n_raw = len(rv.dropna())
    ess = n_raw // horizon
    return ess


def run_har_baseline(
    prices_dir: str,
    train_end: str = "2024-12-31",
    test_start: str = "2026-01-01",
    test_end: str = "2026-05-31",
    horizons: list[int] | None = None,
    tickers: list[str] | None = None,
    val_ratio: float = 0.2,
) -> Dict[int, Dict[str, Dict]]:
    """
    Run HAR-RV baseline for all VN30 stocks across multiple horizons.

    Args:
        prices_dir: Directory containing price CSV files
        train_end: Training end date (exclusive, val split applied)
        test_start: Test start date
        test_end: Test end date
        horizons: List of horizons to evaluate (default [1,5,10,20])
        tickers: List of stock tickers (default VN30_TICKERS)
        val_ratio: Validation split ratio (default 0.2)

    Returns:
        Nested dict: horizon -> ticker -> {predictions, metrics, coeffs}
    """
    if tickers is None:
        tickers = VN30_TICKERS

    if horizons is None:
        horizons = HORIZONS

    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Load prices once
    close = load_close_prices(prices_dir, tickers=tickers)

    print(f"\n{'='*70}")
    print(f"  HAR-RV Baseline Evaluation")
    print(f"  Horizons: {horizons}")
    print(f"  Train end: {train_end}")
    print(f"  Test period: {test_start} to {test_end}")
    print(f"  Val ratio: {val_ratio*100:.0f}%")
    print(f"{'='*70}\n")

    results = {}

    for h in horizons:
        print(f"\n[Horizon h={h}]")
        print(f"  Computing RV (h={h})...")
        rv_all = compute_rv(close, h=h)

        horizon_results = {}
        horizon_metrics = []

        for ticker in tickers:
            if ticker not in rv_all.columns:
                print(f"  [{ticker}] SKIP - not in RV data")
                continue

            try:
                # Fit HAR model with train/val split
                coeffs, split_info = fit_har(
                    rv_all[ticker],
                    train_end_ts,
                    val_ratio=val_ratio
                )

                # Compute ESS
                train_rv = rv_all[ticker][rv_all[ticker].index < train_end_ts]
                ess = compute_ess(train_rv, h)

                # Predict on test set
                preds, metrics = predict_har(
                    rv_all[ticker],
                    coeffs,
                    test_start_ts,
                    test_end_ts
                )

                metrics.update({
                    'coeffs': coeffs.tolist(),
                    'split_info': split_info,
                    'ESS': ess,
                    'ticker': ticker,
                })

                horizon_results[ticker] = {
                    'predictions': preds,
                    'metrics': metrics,
                    'coeffs': coeffs,
                }

                horizon_metrics.append(metrics)

                print(f"  [{ticker}] R2={metrics['R2']:+.4f}, "
                      f"MAE={metrics['MAE']:.5f}, RMSE={metrics['RMSE']:.5f}, "
                      f"ESS={ess}, n={metrics['n_samples']}")

            except Exception as e:
                print(f"  [{ticker}] FAILED: {e}")
                continue

        results[h] = horizon_results

        # Print horizon summary
        if len(horizon_metrics) > 0:
            mean_r2 = np.mean([m['R2'] for m in horizon_metrics])
            mean_mae = np.mean([m['MAE'] for m in horizon_metrics])
            mean_rmse = np.mean([m['RMSE'] for m in horizon_metrics])
            print(f"\n  Horizon h={h} Aggregate ({len(horizon_metrics)} stocks):")
            print(f"    Mean R2:   {mean_r2:+.4f}")
            print(f"    Mean MAE:  {mean_mae:.5f}")
            print(f"    Mean RMSE: {mean_rmse:.5f}")

    return results


def save_results(
    results: Dict[int, Dict[str, Dict]],
    output_dir: str,
):
    """
    Save HAR baseline results to CSV and summary files.

    Args:
        results: Nested dict from run_har_baseline()
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Save per-stock metrics CSV
    all_metrics = []
    for h, horizon_data in results.items():
        for ticker, stock_data in horizon_data.items():
            metrics = stock_data['metrics'].copy()
            metrics['horizon'] = h

            # Flatten split_info dict into separate columns
            split_info = metrics.pop('split_info', {})
            for key, val in split_info.items():
                metrics[key] = val

            all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Reorder columns for readability
    cols = ['horizon', 'ticker', 'R2', 'MAE', 'RMSE', 'n_samples', 'ESS',
            'train_start', 'train_end', 'train_samples',
            'val_start', 'val_end', 'val_samples',
            'test_start', 'test_end']
    # Only include columns that exist
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]

    csv_path = output_path / f'har_baseline_metrics_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n[Save] Metrics CSV: {csv_path}")

    # 2. Save aggregate summary by horizon
    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("  HAR-RV Baseline - Aggregate Summary")
    summary_lines.append("="*70 + "\n")

    for h in sorted(results.keys()):
        horizon_data = results[h]
        horizon_metrics = [stock_data['metrics'] for stock_data in horizon_data.values()]

        if len(horizon_metrics) == 0:
            continue

        mean_r2 = np.mean([m['R2'] for m in horizon_metrics])
        std_r2 = np.std([m['R2'] for m in horizon_metrics])
        mean_mae = np.mean([m['MAE'] for m in horizon_metrics])
        mean_rmse = np.mean([m['RMSE'] for m in horizon_metrics])

        summary_lines.append(f"Horizon h={h} ({len(horizon_metrics)} stocks):")
        summary_lines.append(f"  R2:   {mean_r2:+.4f} +/- {std_r2:.4f}")
        summary_lines.append(f"  MAE:  {mean_mae:.5f}")
        summary_lines.append(f"  RMSE: {mean_rmse:.5f}")

        # Best and worst stocks
        sorted_by_r2 = sorted(horizon_metrics, key=lambda x: x['R2'], reverse=True)
        best = sorted_by_r2[0]
        worst = sorted_by_r2[-1]

        summary_lines.append(f"  Best:  {best['ticker']} (R2={best['R2']:+.4f})")
        summary_lines.append(f"  Worst: {worst['ticker']} (R2={worst['R2']:+.4f})")
        summary_lines.append("")

    # 3. Add comparison notes
    summary_lines.append("="*70)
    summary_lines.append("  Comparison with GNNHAR1L")
    summary_lines.append("="*70 + "\n")

    # GNNHAR1L results (from per_stock_test_results_20260603_182928.csv)
    # Aggregate across 30 stocks for h=5
    gnnhar_results = {
        5: {'R2': 0.6284, 'MAE': 0.00439, 'RMSE': 0.00635},
    }

    for h, har_metrics_list in [(h, [s['metrics'] for s in results[h].values()])
                                for h in results.keys() if h in gnnhar_results]:
        if len(har_metrics_list) == 0:
            continue

        har_r2 = np.mean([m['R2'] for m in har_metrics_list])
        har_mae = np.mean([m['MAE'] for m in har_metrics_list])
        har_rmse = np.mean([m['RMSE'] for m in har_metrics_list])

        gnn_r2 = gnnhar_results[h]['R2']
        gnn_mae = gnnhar_results[h]['MAE']
        gnn_rmse = gnnhar_results[h]['RMSE']

        r2_diff = (har_r2 - gnn_r2) * 100
        mae_diff = ((har_mae - gnn_mae) / gnn_mae) * 100
        rmse_diff = ((har_rmse - gnn_rmse) / gnn_rmse) * 100

        summary_lines.append(f"Horizon h={h}:")
        summary_lines.append(f"  HAR:   R2={har_r2:+.4f}, MAE={har_mae:.5f}, RMSE={har_rmse:.5f}")
        summary_lines.append(f"  GNNHAR1L: R2={gnn_r2:+.4f}, MAE={gnn_mae:.5f}, RMSE={gnn_rmse:.5f}")
        summary_lines.append(f"  Diff:  R2={r2_diff:+.2f}%, MAE={mae_diff:+.2f}%, RMSE={rmse_diff:+.2f}%")
        summary_lines.append("")

    summary_path = output_path / f'har_baseline_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"[Save] Summary TXT: {summary_path}")
    print(f"\n[Summary]")
    print('\n'.join(summary_lines))

    return csv_path, summary_path


# ── Main execution ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    # Load config from project root
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    prices_dir = cfg["data"]["prices_dir"]
    results_dir = Path(__file__).parent.parent / "results" / "baselines"

    # Run HAR baseline with multi-horizon support
    results = run_har_baseline(
        prices_dir=prices_dir,
        train_end="2025-12-31",  # FIXED: Changed from "2024-12-31" to match GNNHAR1L
        test_start="2026-01-01",
        test_end="2026-05-31",
        horizons=HORIZONS,
        val_ratio=0.2,
    )

    # Save results
    csv_path, summary_path = save_results(results, results_dir)

    print("\n" + "="*70)
    print("  HAR-RV baseline evaluation complete!")
    print("="*70)
    print(f"  Metrics: {csv_path.name}")
    print(f"  Summary: {summary_path.name}")
    print("="*70)
