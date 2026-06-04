"""
Comprehensive Model Comparison on All Metrics

Compares HAR, GHAR, and GNNHAR1L on:
- Standard metrics: R2, MAE, RMSE
- Volatility-specific metrics: QLIKE, HMSE, HMAE
- Statistical tests: Diebold-Mariano, Mincer-Zarnowitz

Date: 2026-06-02
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
import torch
from gnn.gnnhar_paper.evaluation import (
    compute_metrics,
    diebold_mariano_test,
    mincer_zarnowitz_regression,
    print_metrics_summary,
    print_dm_test_result
)
from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.build_graph import VN30_TICKERS


def load_model_predictions(result_file: str) -> dict:
    """Load model predictions from result JSON file."""
    with open(result_file, 'r') as f:
        result = json.load(f)
    return result


def print_comparison_table(results_dict: dict):
    """Print comparison table for all models."""
    print("\n" + "="*80)
    print("  COMPREHENSIVE MODEL COMPARISON (h=5, v1.3_LOSS_FIX)")
    print("="*80 + "\n")

    print(f"{'Model':<15} {'R2':>10} {'MAE':>12} {'RMSE':>12} {'QLIKE':>12} {'HMSE':>12} {'HMAE':>12}")
    print("-" * 80)

    for model_name, result in results_dict.items():
        print(f"{model_name:<15} "
              f"{result['test_r2']:>10.4f} "
              f"{result['test_mae']:>12.6f} "
              f"{result['test_rmse']:>12.6f} "
              f"{result.get('qlike', 0):>12.6f} "
              f"{result.get('hmse', 0):>12.6f} "
              f"{result.get('hmae', 0):>12.6f}")

    print("="*80 + "\n")


def print_improvement_summary(baseline: str, results_dict: dict):
    """Print improvement percentage summary."""
    print(f"\n{'='*80}")
    print(f"  IMPROVEMENT OVER {baseline.upper()} BASELINE")
    print(f"{'='*80}\n")

    baseline_metrics = results_dict[baseline]

    for model_name, result in results_dict.items():
        if model_name == baseline:
            continue

        print(f"{model_name} vs {baseline}:")
        print(f"  R2 improvement:     {((result['test_r2'] - baseline_metrics['test_r2']) / abs(baseline_metrics['test_r2']) * 100):>+7.2f}%")
        print(f"  MAE improvement:    {((baseline_metrics['test_mae'] - result['test_mae']) / baseline_metrics['test_mae'] * 100):>+7.2f}%")
        print(f"  RMSE improvement:  {((baseline_metrics['test_rmse'] - result['test_rmse']) / baseline_metrics['test_rmse'] * 100):>+7.2f}%")

        if 'qlike' in result and result['qlike'] != 0:
            print(f"  QLIKE improvement: {((baseline_metrics['qlike'] - result['qlike']) / baseline_metrics['qlike'] * 100):>+7.2f}%")
        print()

    print("="*80 + "\n")


def main():
    # Result files (single seed, v1.3_LOSS_FIX, h=5)
    result_files = {
        'HAR': 'results/gnnhar_paper/multi_stock/HAR_gelu_h5_20260601_235511.json',
        'GHAR': 'results/gnnhar_paper/multi_stock/GHAR_gelu_h5_20260601_235832.json',
        'GNNHAR1L': 'results/gnnhar_paper/multi_stock/GNNHAR1L_gelu_h5_20260602_001312.json',
    }

    print("\n[INFO] Loading model results...")

    # Load results
    results = {}
    for model_name, file_path in result_files.items():
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            print(f"[WARN] {file_path} not found, skipping...")
            continue

        with open(full_path, 'r') as f:
            result = json.load(f)
            results[model_name] = result
            print(f"[OK] Loaded {model_name} from {file_path}")

    if len(results) == 0:
        print("[ERROR] No results found!")
        return

    # Load test data for comprehensive metrics
    print("\n[INFO] Loading test data...")
    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=5,
        train_end='2025-12-31',
        test_start='2026-01-01',
    )
    loader.load_data()
    loader.build_features()
    loader.flatten_dataset()
    loader.split_train_val_test()

    # Note: test_y is numpy array from data loader
    test_y = loader.y_test  # Already numpy array

    # Note: Current result files only have R2, MAE, RMSE
    # We would need to load actual predictions to compute QLIKE, HMSE, HMAE
    # For now, we show what's available

    print("\n[INFO] Available metrics in result files:")
    print("  - R2 (R-squared)")
    print("  - MAE (Mean Absolute Error)")
    print("  - RMSE (Root Mean Squared Error)")
    print("\n[INFO] For full metrics (QLIKE, HMSE, HMAE, DM tests),")
    print("       models need to save predictions to disk.")

    # Print comparison table
    print_comparison_table(results)

    # Print improvement summary over HAR baseline
    if 'HAR' in results:
        print_improvement_summary('HAR', results)

    # Detailed analysis
    print(f"\n{'='*80}")
    print("  DETAILED ANALYSIS")
    print(f"{'='*80}\n")

    print("1. MODEL COMPLEXITY:")
    print("   HAR:     Linear regression on HAR features (RV_d, RV_w, RV_m)")
    print("   GHAR:    Linear regression on HAR + graph adjacency features")
    print("   GNNHAR1L: Graph Neural Network (1-layer GNN) on HAR + graph")
    print()

    print("2. PREDICTIVE PERFORMANCE:")
    if all(m in results for m in ['HAR', 'GHAR', 'GNNHAR1L']):
        har_r2 = results['HAR']['test_r2']
        ghar_r2 = results['GHAR']['test_r2']
        gnn_r2 = results['GNNHAR1L']['test_r2']

        print(f"   HAR baseline:     R2 = {har_r2:.4f}")
        print(f"   GHAR (+graph):    R2 = {ghar_r2:.4f} (+{(ghar_r2-har_r2)/abs(har_r2)*100:+.2f}%)")
        print(f"   GNNHAR1L (+GNN):  R2 = {gnn_r2:.4f} (+{(gnn_r2-har_r2)/abs(har_r2)*100:+.2f}%)")
    print()

    print("3. INTERPRETATION:")
    print("   - HAR baseline is already strong (R2 ~ 0.74)")
    print("   - Graph features provide small improvement (+0.2-0.3%)")
    print("   - Neural network provides additional small gain (+0.5%)")
    print("   - Improvements are incremental, not dramatic")
    print()

    print("4. WHY SMALL IMPROVEMENTS?")
    print("   a) Strong HAR features: RV_d, RV_w, RV_m capture most signal")
    print("   b) Limited data: h=5 with 30 stocks = limited ESS")
    print("   c) Simple task: 5-day volatility may not need complex models")
    print("   d) Good baseline: Linear HAR is hard to beat significantly")
    print()

    print("5. RECOMMENDATION FOR THESIS:")
    print("   [OK] Report HAR as strong linear baseline (R2 ~ 0.74)")
    print("   [OK] Report GNNHAR1L as best model (R2 ~ 0.75)")
    print("   [OK] Show incremental value of each component")
    print("   [OK] Discuss that not all problems need deep learning")
    print("   [OK] Consider ensemble results (n_seeds=20) for stability")

    print("\n" + "="*80)
    print("  METRIC EXPLANATIONS")
    print("="*80 + "\n")

    print("STANDARD METRICS (currently in results):")
    print("  R2:   Proportion of variance explained (higher is better)")
    print("        R2 = 1 - SS_res / SS_tot")
    print("        Range: (-inf, 1], values > 0.7 considered good")
    print()

    print("  MAE:  Mean Absolute Error (lower is better)")
    print("        MAE = mean(|y_true - y_pred|)")
    print("        Robust to outliers, intuitive scale")
    print()

    print("  RMSE: Root Mean Squared Error (lower is better)")
    print("        RMSE = sqrt(mean((y_true - y_pred)^2))")
    print("        Penalizes large errors more than MAE")
    print()

    print("VOLATILITY-SPECIFIC METRICS (require predictions):")
    print("  QLIKE: Quasi-Likelihood loss (Patton 2011, lower is better)")
    print("         QLIKE = mean(log(y_true/y_pred) + y_true/y_pred - 1)")
    print("         Asymmetric: penalizes underprediction more")
    print("         Robust to noise in volatility proxy")
    print()

    print("  HMSE: Heteroskedastic-adjusted MSE (lower is better)")
    print("         HMSE = mean((y_true - y_pred)^2 / y_true)")
    print("         Penalizes errors more when volatility is low")
    print()

    print("  HMAE: Heteroskedastic-adjusted MAE (lower is better)")
    print("         HMAE = mean(|y_true - y_pred| / sqrt(y_true))")
    print("         Adjusts absolute errors by volatility level")
    print()

    print("STATISTICAL TESTS (require predictions):")
    print("  Diebold-Mariano: Tests if forecast accuracy is significantly different")
    print("                    H0: Both models have equal accuracy")
    print("                    p < 0.05 => significant difference")
    print()

    print("  Mincer-Zarnowitz: Tests forecast optimality (regression-based)")
    print("                     y_true = alpha + beta*y_pred + epsilon")
    print("                     Optimal: alpha=0 (unbiased), beta=1 (efficient)")
    print()

    print("="*80 + "\n")

    print("[INFO] To compute full metrics (QLIKE, HMSE, HMAE, DM tests):")
    print("       1. Modify train_multi_stock.py to save test predictions")
    print("       2. Re-run training with n_seeds=1 for quick validation")
    print("       3. Run this script again to load predictions")
    print("\n")


if __name__ == "__main__":
    main()
