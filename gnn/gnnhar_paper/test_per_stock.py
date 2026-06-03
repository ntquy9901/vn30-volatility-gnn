"""
Per-Stock GNNHAR Performance Testing

Tests v1.3_LOSS_FIX models on all 30 VN30 stocks individually.
Analyzes which stocks perform well/poor and correlates with stock characteristics.

Usage:
    python gnn/gnnhar_paper/test_per_stock.py

Output:
    - CSV: results/gnnhar_paper/per_stock_test_results.csv
    - Chart: results/gnnhar_paper/per_stock_performance.png
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
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS, SECTOR_MAP
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY, create_model
from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.graph_builder import GraphBuilder
from gnn.gnnhar_paper.train_multi_stock import forward_pass_with_mask


def load_latest_models(horizon: int = 5, model_name: str = 'GNNHAR1L', top_k: int = 3):
    """
    Load latest v1.3_LOSS_FIX models from multi_stock training.

    Args:
        horizon: Forecast horizon
        model_name: Model to load
        top_k: Number of top models to use (screened by val loss)

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


def get_stock_characteristics(close: pd.DataFrame, stock: str, test_dates: pd.DatetimeIndex):
    """
    Extract stock characteristics for analysis.

    Args:
        close: Close prices (T, N) DataFrame
        stock: Stock ticker
        test_dates: Test period dates

    Returns:
        dict with sector, market_cap, mean_rv
    """
    # Sector
    sector = SECTOR_MAP.get(stock, 'Unknown')

    # Market cap classification (approximate based on price)
    # This is a rough classification - real market cap would need market data
    avg_price = close.loc[test_dates, stock].mean()
    if avg_price > 50000:  # Very high price = large cap
        market_cap = 'Large'
    elif avg_price > 20000:
        market_cap = 'Mid'
    else:
        market_cap = 'Small'

    # Compute RV for test period
    close_stock = close[stock].loc[test_dates]
    rv = compute_rv(close_stock.to_frame(), h=5)
    mean_rv = rv.mean().mean()

    return {
        'sector': sector,
        'market_cap': market_cap,
        'mean_rv': mean_rv,
    }


def main():
    """Main testing function."""
    print("\n" + "="*70)
    print("  Per-Stock GNNHAR Performance Testing")
    print("  Testing v1.3_LOSS_FIX models on all 30 VN30 stocks")
    print("="*70 + "\n")

    # Configuration
    horizon = 5
    model_name = 'GNNHAR1L'
    top_k = 3
    train_end = '2024-12-31'
    test_start = '2026-01-01'
    test_end = '2026-05-31'

    # Step 1: Load models
    print("[Step 1] Loading v1.3_LOSS_FIX models...")
    models, timestamp = load_latest_models(horizon, model_name, top_k)

    # Step 2: Load data
    print("\n[Step 2] Loading data...")
    data_root = Path(__file__).parent.parent.parent
    import yaml
    with open(data_root / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    data_dir = data_root / cfg['data']['prices_dir']

    close = load_close_prices(data_dir, tickers=VN30_TICKERS)
    print(f"  Loaded {close.shape[0]} dates x {close.shape[1]} stocks")

    # Step 3: Build adjacency
    print("\n[Step 3] Building adjacency matrix...")
    log_ret = compute_log_returns(close)
    graph_builder = GraphBuilder(method='pearson', threshold=0.3)
    adj = graph_builder.build_adjacency(log_ret, pd.Timestamp(train_end))
    adj_tensor = torch.from_numpy(adj).float()
    print(f"  Adjacency: {adj.shape}, density={(adj > 0).sum() / adj.size:.2%}")

    # Step 4: Prepare test data
    print("\n[Step 4] Preparing test data...")
    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=horizon,
        train_end=train_end,
        test_start=test_start,
    )
    loader.load_data()
    loader.build_features()
    loader.flatten_dataset()
    loader.split_train_val_test()

    # Get test data
    _, _, _, _, X_val, y_val, stocks_val, dates_val, \
    X_test, y_test, stocks_test, dates_test = loader.prepare_pytorch_data(val_split=0.2)

    # Filter to test period
    dates_pd = pd.to_datetime(dates_test)
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)
    period_mask = (dates_pd >= test_start_dt) & (dates_pd <= test_end_dt)

    X_inf = X_test[period_mask]
    y_inf = y_test[period_mask]
    stocks_inf = stocks_test[period_mask]
    dates_inf = dates_test[period_mask]

    print(f"  Test period: {dates_inf[0].date()} to {dates_inf[-1].date()} ({len(dates_inf)} days)")

    # Step 5: Run per-stock inference
    print("\n[Step 5] Running per-stock inference...")

    results = []

    for stock_idx, stock in enumerate(VN30_TICKERS):
        print(f"  [{stock}] Testing...")

        # Filter data for this stock
        stock_mask = stocks_inf == stock_idx
        X_stock = X_inf[stock_mask]
        y_stock = y_inf[stock_mask]

        if len(X_stock) < 2:
            print(f"    [SKIP] Insufficient data ({len(X_stock)} samples)")
            continue

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
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = np.mean(np.abs(y_stock - y_pred))
        rmse = np.sqrt(np.mean((y_stock - y_pred) ** 2))

        # Get stock characteristics
        chars = get_stock_characteristics(close, stock, pd.DatetimeIndex(dates_inf))

        # Performance group
        if r2 > 0.5:
            group = 'good'
        elif r2 > 0.0:
            group = 'moderate'
        else:
            group = 'poor'

        print(f"    R2={r2:+.4f}, MAE={mae:.5f}, Group={group}")

        results.append({
            'stock': stock,
            'sector': chars['sector'],
            'market_cap': chars['market_cap'],
            'mean_rv': chars['mean_rv'],
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(X_stock),
            'group': group,
        })

    # Step 6: Save results
    print("\n[Step 6] Saving results...")
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'gnnhar_paper'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / f'per_stock_test_results_{timestamp_str}.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Step 7: Generate visualization
    print("\n[Step 7] Generating visualization...")

    # Sort by R2 for better visualization
    df_sorted = df.sort_values('r2', ascending=False)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    # Color by group
    colors = df_sorted['group'].map({'good': 'green', 'moderate': 'orange', 'poor': 'red'})

    bars = ax.barh(df_sorted['stock'], df_sorted['r2'], color=colors, alpha=0.7)

    # Add sector labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['r2'] + 0.02, i, row['sector'],
                va='center', fontsize=8, color='black')

    # Vertical line at R2 = 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('R2 Score', fontsize=12)
    ax.set_ylabel('Stock', fontsize=12)
    ax.set_title(f'Per-Stock GNNHAR Performance (h={horizon}, {model_name}, v1.3_LOSS_FIX)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 1.0)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good (R2 > 0.5)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate (0.0 < R2 <= 0.5)'),
        Patch(facecolor='red', alpha=0.7, label='Poor (R2 <= 0.0)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    chart_path = results_dir / f'per_stock_performance_{timestamp_str}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {chart_path}")

    # Step 8: Summary statistics
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    group_counts = df['group'].value_counts()
    print(f"\n  Performance Groups:")
    print(f"    Good (R2 > 0.5):    {group_counts.get('good', 0)} stocks")
    print(f"    Moderate (0 < R2 <= 0.5): {group_counts.get('moderate', 0)} stocks")
    print(f"    Poor (R2 <= 0):    {group_counts.get('poor', 0)} stocks")

    print(f"\n  Aggregate Statistics:")
    print(f"    Mean R2:   {df['r2'].mean():+.4f}")
    print(f"    Median R2: {df['r2'].median():+.4f}")
    print(f"    Std R2:    {df['r2'].std():+.4f}")
    print(f"    Best stock: {df.loc[df['r2'].idxmax(), 'stock']} (R2={df['r2'].max():+.4f})")
    print(f"    Worst stock: {df.loc[df['r2'].idxmin(), 'stock']} (R2={df['r2'].min():+.4f})")

    print(f"\n  By Sector:")
    sector_stats = df.groupby('sector')['r2'].agg(['mean', 'count'])
    sector_stats = sector_stats.sort_values('mean', ascending=False)
    for sector, row in sector_stats.iterrows():
        print(f"    {sector:12s} Mean R2={row['mean']:+.4f} (n={int(row['count'])})")

    print(f"\n  By Market Cap:")
    cap_stats = df.groupby('market_cap')['r2'].agg(['mean', 'count'])
    cap_stats = cap_stats.sort_values('mean', ascending=False)
    for cap, row in cap_stats.iterrows():
        print(f"    {cap:8s} Mean R2={row['mean']:+.4f} (n={int(row['count'])})")

    print("\n" + "="*70)
    print(f"  Results saved to: {csv_path.name}")
    print(f"  Chart saved to:   {chart_path.name}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
