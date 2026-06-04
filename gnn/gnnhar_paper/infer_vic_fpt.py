"""
Inference script for VIC and FPT stocks using trained GNNHAR1L models.

This script demonstrates how to:
1. Load trained GNNHAR1L ensemble models
2. Prepare input data for VIC and FPT stocks
3. Generate predictions for multiple horizons
4. Compare predictions with actual volatility

Usage:
    python gnn/gnnhar_paper/infer_vic_fpt.py --horizon 5 --start_date 2026-01-01 --end_date 2026-05-31
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import yaml
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS, ALL_NODES, NODE_IDX
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
from gnn.gnnhar_paper.ensemble_trainer import EnsembleTrainer
from gnn.gnnhar_paper.rolling_datasets import compute_har_features

# Configuration
ROOT = Path(__file__).parent.parent.parent
with open(ROOT / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

DATA_DIR = ROOT / CFG['data']['prices_dir']
MODELS_DIR = ROOT / 'models' / 'gnnhar_paper'
RESULTS_DIR = ROOT / 'results' / 'gnnhar_paper'

# Target stocks for inference
TARGET_STOCKS = ['VIC', 'FPT']

def load_trained_model(model_name: str, horizon: int, device: torch.device):
    """
    Load trained ensemble model for specific horizon.

    Args:
        model_name: Model name (e.g., 'GNNHAR1L', 'HAR', 'GHAR')
        horizon: Forecast horizon (1, 5, 10, 20)
        device: torch device

    Returns:
        List of trained model state dicts
    """
    model_dir = MODELS_DIR / f'h{horizon}' / model_name

    if not model_dir.exists():
        raise ValueError(f"No trained models found for {model_name} h{horizon}")

    models = []
    metadata_path = model_dir / 'metadata.npz'

    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        metadata = dict(np.load(metadata_path, allow_pickle=True))
        print(f"  Loaded metadata: {list(metadata.keys())}")

    # Load all model files
    model_files = sorted(model_dir.glob('model_*.pt'))
    print(f"  Found {len(model_files)} trained models")

    for model_file in model_files:
        state_dict = torch.load(model_file, map_location=device)
        models.append(state_dict)

    return models, metadata

def prepare_inference_data(
    close_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    target_stocks: list,
    start_date: str,
    end_date: str,
    horizon: int
):
    """
    Prepare data for inference on specific stocks.

    Args:
        close_prices: Close prices for all stocks
        log_returns: Log returns for all stocks
        target_stocks: List of stock tickers to predict
        start_date: Inference start date
        end_date: Inference end date
        horizon: Forecast horizon

    Returns:
        X_features, y_true, dates, stock_indices
    """
    # Filter date range
    date_mask = (log_returns.index >= pd.Timestamp(start_date)) & \
                (log_returns.index <= pd.Timestamp(end_date))
    inference_dates = log_returns.index[date_mask]

    print(f"  Inference period: {len(inference_dates)} dates ({start_date} to {end_date})")

    # Compute HAR features for all stocks
    rv_d, rv_w, rv_m = compute_har_features(log_returns)

    # Compute target RV for all stocks
    rv_target = compute_rv(close_prices, h=horizon)

    # Get stock indices
    stock_indices = {}
    valid_stocks = []

    for stock in target_stocks:
        if stock in VN30_TICKERS and stock in log_returns.columns:
            stock_indices[stock] = NODE_IDX[stock]
            valid_stocks.append(stock)
            print(f"  {stock}: index {NODE_IDX[stock]}")
        else:
            print(f"  [WARN] {stock} not found in data")

    # Prepare features and targets for valid stocks
    n_dates = len(inference_dates)
    X_features = np.zeros((n_dates, len(ALL_NODES), 3), dtype=np.float32)
    y_true = np.zeros((n_dates, len(ALL_NODES)), dtype=np.float32)

    for date_idx, date in enumerate(inference_dates):
        for stock in valid_stocks:
            node_idx = stock_indices[stock]

            # HAR features: [rv_d, rv_w, rv_m]
            X_features[date_idx, node_idx, 0] = rv_d.loc[date, stock]
            X_features[date_idx, node_idx, 1] = rv_w.loc[date, stock]
            X_features[date_idx, node_idx, 2] = rv_m.loc[date, stock]

            # Target RV
            if stock in rv_target.columns:
                y_true[date_idx, node_idx] = rv_target.loc[date, stock]

    return X_features, y_true, inference_dates, stock_indices

def build_adjacency_matrix(log_returns: pd.DataFrame, train_end_date: str, threshold: float = 0.3):
    """
    Build static adjacency matrix from correlation matrix.

    Args:
        log_returns: Log returns for all stocks
        train_end_date: Use data up to this date for correlation
        threshold: Correlation threshold for edges

    Returns:
        Adjacency matrix (N, N)
    """
    # Use pre-2026 data for adjacency (no lookahead)
    train_returns = log_returns[log_returns.index < pd.Timestamp(train_end_date)]

    # Compute correlation matrix
    correlation = train_returns.corr()

    # Create adjacency matrix
    n_nodes = len(ALL_NODES)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i, stock_i in enumerate(ALL_NODES):
        for j, stock_j in enumerate(ALL_NODES):
            if i != j and stock_i in correlation.columns and stock_j in correlation.columns:
                corr_val = correlation.loc[stock_i, stock_j]
                if not pd.isna(corr_val) and abs(corr_val) >= threshold:
                    adj[i, j] = corr_val

    print(f"  Adjacency matrix: {n_nodes}x{n_nodes}, density {(adj > 0).sum() / (n_nodes * n_nodes):.3f}")

    return adj

def run_ensemble_inference(
    model_name: str,
    horizon: int,
    X_features: np.ndarray,
    adj: np.ndarray,
    device: torch.device
):
    """
    Run inference using trained ensemble models.

    Args:
        model_name: Model name (e.g., 'GNNHAR1L')
        horizon: Forecast horizon
        X_features: (n_dates, n_nodes, 3) input features
        adj: (n_nodes, n_nodes) adjacency matrix
        device: torch device

    Returns:
        Predictions (n_dates, n_nodes)
    """
    # Load trained models
    print(f"\n[2] Loading {model_name} models for h{horizon}...")
    models, metadata = load_trained_model(model_name, horizon, device)

    # Get model architecture
    model_class = MODEL_REGISTRY[model_name]
    n_hid = metadata.get('n_hid', 16) if metadata else 16

    print(f"\n[3] Running inference with {len(models)} models...")

    # Ensemble predictions
    predictions = []

    for i, state_dict in enumerate(models):
        # Initialize model
        if model_name == 'HAR':
            model = model_class()
        else:
            model = model_class(n_hid)

        # Load trained weights
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Predict
        X_t = torch.from_numpy(X_features).float().to(device)
        adj_t = torch.from_numpy(adj).float().to(device)

        with torch.no_grad():
            pred = model(X_t, adj_t)
            predictions.append(pred.cpu().numpy())

    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    print(f"  Ensemble prediction shape: {ensemble_pred.shape}")

    return ensemble_pred

def evaluate_predictions(predictions: np.ndarray, y_true: np.ndarray,
                        stock_indices: dict, inference_dates: pd.DatetimeIndex):
    """
    Evaluate predictions vs actual values.

    Args:
        predictions: (n_dates, n_nodes) predicted values
        y_true: (n_dates, n_nodes) actual values
        stock_indices: Map stock name to node index
        inference_dates: Dates for inference

    Returns:
        Results dictionary with metrics per stock
    """
    print(f"\n[4] Evaluating predictions...")

    results = {}

    for stock, node_idx in stock_indices.items():
        # Extract predictions and true values for this stock
        y_pred_stock = predictions[:, node_idx]
        y_true_stock = y_true[:, node_idx]

        # Remove NaN values
        valid_mask = ~np.isnan(y_true_stock) & ~np.isnan(y_pred_stock)
        y_pred_valid = y_pred_stock[valid_mask]
        y_true_valid = y_true_stock[valid_mask]

        if len(y_true_valid) < 2:
            print(f"  {stock}: Insufficient valid data ({len(y_true_valid)} samples)")
            continue

        # Compute metrics
        r2 = 1 - np.sum((y_true_valid - y_pred_valid)**2) / np.sum((y_true_valid - y_true_valid.mean())**2)
        mae = np.mean(np.abs(y_true_valid - y_pred_valid))
        rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid)**2))

        results[stock] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(y_true_valid),
            'predictions': y_pred_valid,
            'true_values': y_true_valid,
            'dates': inference_dates[valid_mask]
        }

        print(f"  {stock}: R²={r2:.4f}, MAE={mae:.6f}, RMSE={rmse:.6f}, n={len(y_true_valid)}")

    return results

def plot_predictions(results: dict, model_name: str, horizon: int, output_dir: Path):
    """
    Plot predictions vs actual values for target stocks.

    Args:
        results: Results dictionary from evaluate_predictions
        model_name: Name of model used
        horizon: Forecast horizon
        output_dir: Directory to save plots
    """
    print(f"\n[5] Generating plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for stock, data in results.items():
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        dates = data['dates']
        y_true = data['true_values']
        y_pred = data['predictions']

        # Plot 1: Time series comparison
        axes[0].plot(dates, y_true, label='Actual RV', marker='o', markersize=3, alpha=0.7)
        axes[0].plot(dates, y_pred, label='Predicted RV', marker='s', markersize=3, alpha=0.7)
        axes[0].set_title(f'{stock} - RV Forecasting (h={horizon}) - {model_name}')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Realized Volatility')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.6)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
        axes[1].set_title(f'{stock} - Prediction vs Actual (R²={data["r2"]:.4f})')
        axes[1].set_xlabel('Actual RV')
        axes[1].set_ylabel('Predicted RV')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_file = output_dir / f'{stock}_h{horizon}_{model_name}_{timestamp}.png'
        plt.savefig(plot_file, dpi=100)
        plt.close()

        print(f"  Saved: {plot_file}")

def save_results(results: dict, model_name: str, horizon: int, output_dir: Path):
    """
    Save inference results to CSV file.

    Args:
        results: Results dictionary from evaluate_predictions
        model_name: Name of model used
        horizon: Forecast horizon
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare summary data
    summary_data = []
    detailed_data = []

    for stock, data in results.items():
        # Summary row
        summary_data.append({
            'model': model_name,
            'horizon': horizon,
            'stock': stock,
            'r2': data['r2'],
            'mae': data['mae'],
            'rmse': data['rmse'],
            'n_samples': data['n_samples']
        })

        # Detailed rows
        for i, (date, y_t, y_p) in enumerate(zip(data['dates'], data['true_values'], data['predictions'])):
            detailed_data.append({
                'model': model_name,
                'horizon': horizon,
                'stock': stock,
                'date': date.strftime('%Y-%m-%d'),
                'actual_rv': y_t,
                'predicted_rv': y_p,
                'error': y_t - y_p,
                'abs_error': abs(y_t - y_p)
            })

    # Save summary
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_file = output_dir / f'inference_summary_h{horizon}_{model_name}_{timestamp}.csv'
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"\n  Saved summary: {summary_file}")

    # Save detailed data
    detail_file = output_dir / f'inference_detailed_h{horizon}_{model_name}_{timestamp}.csv'
    pd.DataFrame(detailed_data).to_csv(detail_file, index=False)
    print(f"  Saved details: {detail_file}")

def main():
    parser = argparse.ArgumentParser(description='Inference for VIC and FPT stocks')
    parser.add_argument('--model', type=str, default='GNNHAR1L',
                       choices=['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L'],
                       help='Model to use for inference')
    parser.add_argument('--horizon', type=int, default=5,
                       choices=[1, 5, 10, 20],
                       help='Forecast horizon')
    parser.add_argument('--start_date', type=str, default='2026-01-01',
                       help='Inference start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2026-05-31',
                       help='Inference end date (YYYY-MM-DD)')
    parser.add_argument('--corr_threshold', type=float, default=0.3,
                       help='Correlation threshold for adjacency matrix')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"{'='*70}")
    print(f"  GNNHAR Inference - VIC & FPT Stocks")
    print(f"  Model: {args.model}, Horizon: h{args.horizon}")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    # Load data
    print("[1] Loading data...")
    tickers_all = VN30_TICKERS + ["VNINDEX"]
    close_prices = load_close_prices(DATA_DIR, tickers=tickers_all)
    log_returns = compute_log_returns(close_prices)

    print(f"  Loaded {close_prices.shape[0]} dates x {close_prices.shape[1]} stocks")

    # Prepare inference data
    print(f"\n[1] Preparing inference data...")
    X_features, y_true, inference_dates, stock_indices = prepare_inference_data(
        close_prices, log_returns, TARGET_STOCKS,
        args.start_date, args.end_date, args.horizon
    )

    # Build adjacency matrix
    print(f"\n[1] Building adjacency matrix...")
    adj = build_adjacency_matrix(log_returns, args.start_date, args.corr_threshold)

    # Run inference
    predictions = run_ensemble_inference(
        args.model, args.horizon, X_features, adj, device
    )

    # Evaluate predictions
    results = evaluate_predictions(predictions, y_true, stock_indices, inference_dates)

    # Generate plots
    plot_predictions(results, args.model, args.horizon, RESULTS_DIR)

    # Save results
    save_results(results, args.model, args.horizon, RESULTS_DIR)

    print(f"\n{'='*70}")
    print(f"  Inference complete!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()