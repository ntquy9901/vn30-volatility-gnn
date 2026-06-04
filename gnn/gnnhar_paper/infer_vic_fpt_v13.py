"""
Inference script for VIC and FPT stocks using v1.3_LOSS_FIX models.

This script loads models from the new multi_stock training directory
and runs inference on VIC and FPT stocks for comparison with old results.

Key differences from infer_vic_fpt.py:
1. Loads models from models/gnnhar_paper_multi_stock/ (v1.3_LOSS_FIX)
2. Uses model metadata to filter by version
3. Applies screening based on validation loss
4. Compares with old buggy model results

Usage:
    python gnn/gnnhar_paper/infer_vic_fpt_v13.py --horizon 5

Expected results:
    - VIC R² should improve from -2.54 (buggy) to positive values
    - FPT R² should improve from -0.85 (buggy) to positive values
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
from glob import glob

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY, create_model
from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.graph_builder import GraphBuilder
from gnn.gnnhar_paper.train_multi_stock import forward_pass_with_mask
from src.volatility_labels import compute_log_returns


def load_latest_models(horizon: int, model_name: str = 'GNNHAR1L', top_k: int = 3):
    """
    Load latest models from multi_stock training directory.

    Args:
        horizon: Forecast horizon (1, 5, 10, 20)
        model_name: Model to load (HAR, GHAR, GNNHAR1L, etc.)
        top_k: Number of top models to load (screened by val loss)

    Returns:
        models: List of (model, metadata) tuples
        timestamp: Training timestamp
    """
    models_dir = Path('models/gnnhar_paper_multi_stock') / f'h{horizon}' / model_name

    if not models_dir.exists():
        raise FileNotFoundError(f"No models found at {models_dir}")

    # Find all model files
    model_files = sorted(models_dir.glob('*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)

    if len(model_files) == 0:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    # Get timestamp from most recent file
    timestamp = model_files[0].stem.split('_')[-1]

    # Filter models by this timestamp
    timestamp_models = [f for f in model_files if timestamp in f.stem]

    print(f"[Load Models] Found {len(timestamp_models)} models from timestamp {timestamp}")

    # Load models and metadata
    models_data = []
    for model_file in timestamp_models:
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            metadata = {
                'seed': checkpoint.get('seed', 'unknown'),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'version': checkpoint.get('version', 'unknown'),
                'n_hid': checkpoint.get('n_hid', 16),
                'activation': checkpoint.get('activation', 'relu'),
            }

            # Only load v1.3_LOSS_FIX models
            if metadata['version'] != 'v1.3_LOSS_FIX':
                print(f"  [SKIP] seed{metadata['seed']} - version {metadata['version']} (not v1.3_LOSS_FIX)")
                continue

            # Create model and load weights
            model = create_model(
                model_name,
                n_hid=metadata['n_hid'],
                activation=metadata['activation'],
                dropout=0.0,
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            models_data.append((model, metadata))
            print(f"  [LOAD] seed{metadata['seed']} - val_loss={metadata['val_loss']:.6f}")

        except Exception as e:
            print(f"  [ERROR] Failed to load {model_file.name}: {e}")
            continue

    if len(models_data) == 0:
        raise ValueError("No valid v1.3_LOSS_FIX models found")

    # Screen by validation loss (keep top_k)
    models_data.sort(key=lambda x: x[1]['val_loss'])
    screened = models_data[:top_k]

    print(f"\n[Screen] Selected {len(screened)}/{len(models_data)} models (lowest val loss)")

    return screened, timestamp


def run_inference_vic_fpt(horizon: int, model_name: str = 'GNNHAR1L',
                         start_date: str = '2026-01-01', end_date: str = '2026-03-27',
                         top_k: int = 3):
    """
    Run inference on VIC and FPT stocks using v1.3_LOSS_FIX models.

    Args:
        horizon: Forecast horizon
        model_name: Model to use
        start_date: Inference start date
        end_date: Inference end date
        top_k: Number of top models to use
    """
    print("\n" + "="*70)
    print("  GNNHAR Inference (v1.3_LOSS_FIX) - VIC & FPT")
    print(f"  Model: {model_name}, Horizon: h{horizon}")
    print(f"  Period: {start_date} to {end_date}")
    print("="*70 + "\n")

    # Step 1: Load models
    print("[Step 1] Loading v1.3_LOSS_FIX models...")
    models, timestamp = load_latest_models(horizon, model_name, top_k=top_k)

    # Step 2: Load data
    print("\n[Step 2] Loading data...")
    data_root = Path(__file__).parent.parent.parent
    import yaml
    with open(data_root / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    data_dir = data_root / cfg['data']['prices_dir']

    close = load_close_prices(data_dir, tickers=VN30_TICKERS)
    log_ret = compute_log_returns(close)

    print(f"  Loaded {close.shape[0]} dates x {close.shape[1]} stocks")

    # Step 3: Build adjacency
    print("\n[Step 3] Building adjacency matrix...")
    train_end = '2024-12-31'  # From training configuration
    graph_builder = GraphBuilder(method='pearson', threshold=0.3)
    adj = graph_builder.build_adjacency(log_ret, pd.Timestamp(train_end))
    adj_tensor = torch.from_numpy(adj).float()

    print(f"  Adjacency: {adj.shape}, density={(adj > 0).sum() / adj.size:.2%}")

    # Step 4: Prepare inference data
    print("\n[Step 4] Preparing inference data...")

    # Get stock indices
    vic_idx = VN30_TICKERS.index('VIC')
    fpt_idx = VN30_TICKERS.index('FPT')

    print(f"  VIC: index {vic_idx}")
    print(f"  FPT: index {fpt_idx}")

    # Filter dates
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    # Use MultiStockDataLoader to prepare data
    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=horizon,
        train_end=train_end,
        test_start='2026-01-01',
    )
    loader.load_data()
    loader.build_features()
    loader.flatten_dataset()
    loader.split_train_val_test()

    # Get test data
    _, _, _, _, X_val, y_val, stocks_val, dates_val, X_test, y_test, stocks_test, dates_test = loader.prepare_pytorch_data(val_split=0.2)

    # Filter to inference period
    dates_pd = pd.to_datetime(dates_test)
    period_mask = (dates_pd >= start_dt) & (dates_pd <= end_dt)

    X_inf = X_test[period_mask]
    y_inf = y_test[period_mask]
    stocks_inf = stocks_test[period_mask]
    dates_inf = dates_test[period_mask]

    # Filter to VIC and FPT
    vic_mask = stocks_inf == vic_idx
    fpt_mask = stocks_inf == fpt_idx

    X_vic = X_inf[vic_mask]
    y_vic = y_inf[vic_mask]
    dates_vic = dates_inf[vic_mask]

    X_fpt = X_inf[fpt_mask]
    y_fpt = y_inf[fpt_mask]
    dates_fpt = dates_inf[fpt_mask]

    print(f"  VIC: {len(dates_vic)} samples")
    print(f"  FPT: {len(dates_fpt)} samples")

    # Step 5: Run inference
    print("\n[Step 5] Running ensemble inference...")

    results = {}

    for stock_name, stock_idx, X_stock, y_stock, dates_stock in [
        ('VIC', vic_idx, X_vic, y_vic, dates_vic),
        ('FPT', fpt_idx, X_fpt, y_fpt, dates_fpt),
    ]:
        print(f"\n  [{stock_name}] Predicting...")

        # Create stock index tensor
        stock_indices = torch.full((len(X_stock),), stock_idx, dtype=torch.long)

        # Ensemble predictions
        predictions = []
        for model, metadata in models:
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_stock).float()
                pred = forward_pass_with_mask(model, X_tensor, adj_tensor, stock_indices)
                pred = torch.clamp(pred, min=1e-4)
                predictions.append(pred.numpy())

        # Average predictions
        y_pred = np.mean(predictions, axis=0).flatten()

        # Compute metrics
        ss_res = np.sum((y_stock - y_pred) ** 2)
        ss_tot = np.sum((y_stock - y_stock.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        mae = np.mean(np.abs(y_stock - y_pred))
        rmse = np.sqrt(np.mean((y_stock - y_pred) ** 2))

        print(f"    R2={r2:+.4f}, MAE={mae:.5f}, RMSE={rmse:.5f}")

        results[stock_name] = {
            'y_true': y_stock,
            'y_pred': y_pred,
            'dates': dates_stock,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
        }

    # Step 6: Save results
    print("\n[Step 6] Saving results...")
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    for stock_name, data in results.items():
        # Create detailed CSV
        rows = []
        for i, date in enumerate(data['dates']):
            rows.append({
                'model': f'{model_name}_v1.3_LOSS_FIX',
                'horizon': horizon,
                'stock': stock_name,
                'date': str(date),
                'actual_rv': data['y_true'][i],
                'predicted_rv': data['y_pred'][i],
                'error': data['y_true'][i] - data['y_pred'][i],
                'abs_error': np.abs(data['y_true'][i] - data['y_pred'][i]),
            })

        df = pd.DataFrame(rows)
        csv_path = data_root / 'results' / 'gnnhar_paper' / f'inference_v13_{stock_name}_h{horizon}_{timestamp_str}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    print("\n" + "="*70)
    print("  SUMMARY (v1.3_LOSS_FIX)")
    print("="*70)
    for stock_name, data in results.items():
        print(f"  {stock_name}: R2={data['r2']:+.4f}, MAE={data['mae']:.5f}, RMSE={data['rmse']:.5f}")

    print("\n  Comparison with old buggy results:")
    print("  VIC:  R2 = -2.54 (buggy) -> {:.2f} (v1.3_LOSS_FIX)".format(results['VIC']['r2']))
    print("  FPT:  R2 = -0.85 (buggy) -> {:.2f} (v1.3_LOSS_FIX)".format(results['FPT']['r2']))
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with v1.3_LOSS_FIX models')
    parser.add_argument('--model', type=str, default='GNNHAR1L',
                        choices=['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L'])
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--start_date', type=str, default='2026-01-01')
    parser.add_argument('--end_date', type=str, default='2026-03-27')
    parser.add_argument('--top_k', type=int, default=3)

    args = parser.parse_args()

    try:
        run_inference_vic_fpt(
            horizon=args.horizon,
            model_name=args.model,
            start_date=args.start_date,
            end_date=args.end_date,
            top_k=args.top_k,
        )
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        raise
