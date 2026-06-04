"""
VIC Ensemble Inference - Load trained ensemble models and make predictions

This script demonstrates how end-users can use the trained ensemble models
for volatility forecasting on new data.

Usage:
    python vic_ensemble_inference.py --model GHAR --ticker VIC
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_rv
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY

print("\n" + "="*70)
print("  VIC ENSEMBLE INFERENCE")
print("="*70 + "\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZON = 5  # Must match training horizon

# Ensemble model directory
ENSEMBLE_DIR = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'vic_ensemble_models'

# Default settings
DEFAULT_MODEL = 'GHAR'  # Best performing model
DEFAULT_TICKER = 'VIC'

# =============================================================================
# LOAD ENSEMBLE MODELS
# =============================================================================

def load_ensemble(model_name, ensemble_dir):
    """Load all trained models for an ensemble."""
    model_dir = ensemble_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained ensemble found for {model_name} at {model_dir}")

    # Load ensemble metadata
    metadata_file = model_dir / 'ensemble_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"[Loading] {model_name} ensemble from {model_dir}")
    print(f"  Models: {metadata['num_models']} trained")
    print(f"  Screening: {metadata['num_screened']} models selected")
    print(f"  Seeds: {metadata['seeds']}")

    # Load screened models
    models = []
    for idx in metadata['screened_indices']:
        model_file = model_dir / f'model_{idx}.pt'
        if model_name == 'HAR':
            model = MODEL_REGISTRY[model_name]()
        else:
            n_hid = metadata['n_hid']
            model = MODEL_REGISTRY[model_name](n_hid=n_hid)

        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        models.append(model)

    print(f"  Loaded {len(models)} models for inference")

    return models, metadata

# =============================================================================
# PREPARE INPUT FEATURES
# =============================================================================

def prepare_features(rv_series, lookback_days=22):
    """
    Prepare HAR features for a single prediction.

    Args:
        rv_series: pd.Series of RV values (must have at least 22+horizon days)
        lookback_days: days to look back for HAR features (default 22)

    Returns:
        np.array of shape (1, 3) with [rv_d, rv_w, rv_m] features
    """
    if len(rv_series) < lookback_days + HORIZON:
        raise ValueError(f"Need at least {lookback_days + HORIZON} days of RV data")

    # Take most recent data
    recent_rv = rv_series.iloc[-lookback_days:]

    # HAR features
    rv_d = recent_rv.iloc[-1:].mean()  # Most recent day
    rv_w = recent_rv.iloc[-5:].mean()   # Last 5 days
    rv_m = recent_rv.iloc[-22:].mean()  # Last 22 days

    features = np.array([[rv_d, rv_w, rv_m]])
    return features

# =============================================================================
# MAKE PREDICTION
# =============================================================================

def predict_ensemble(models, features, adj=None):
    """
    Make prediction using ensemble of models.

    Args:
        models: list of trained PyTorch models
        features: np.array of shape (1, 3) with HAR features
        adj: adjacency matrix (default: identity for single stock)

    Returns:
        float: ensemble prediction (average of all models)
    """
    if adj is None:
        adj = np.array([[1.0]])  # Single node for VIC

    X_t = torch.from_numpy(features).float().unsqueeze(1)  # (1, 1, 3)

    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X_t, torch.from_numpy(adj).float())
            predictions.append(pred.squeeze().item())

    # Average predictions
    ensemble_pred = np.mean(predictions)

    # Also provide std (uncertainty estimate)
    pred_std = np.std(predictions)

    return ensemble_pred, pred_std

# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def forecast_volatility(model_name, ticker, forecast_date=None):
    """
    Main function: Forecast volatility for a given ticker and date.

    Args:
        model_name: Model to use (HAR, GHAR, GNNHAR1L, GNNHAR2L, GNNHAR3L)
        ticker: Stock ticker (default: VIC)
        forecast_date: Date to forecast (default: most recent)

    Returns:
        dict with prediction, uncertainty, and metadata
    """
    print(f"\n{'='*70}")
    print(f"  VOLATILITY FORECAST: {ticker} using {model_name}")
    print(f"{'='*70}\n")

    # Load ensemble models
    models, metadata = load_ensemble(model_name, ENSEMBLE_DIR)

    # Load latest data
    print(f"\n[Data] Loading {ticker} price data...")
    close = load_close_prices(
        PROJECT_ROOT / 'data' / 'raw' / 'prices',
        tickers=[ticker]
    )
    rv = compute_rv(close, h=HORIZON)[ticker].dropna()

    # Use most recent data
    if forecast_date is None:
        forecast_date = rv.index[-1]
        print(f"  Forecasting for: {forecast_date.date()}")
    else:
        print(f"  Forecasting for: {forecast_date}")
        # Filter data up to forecast date
        rv = rv[rv.index <= pd.Timestamp(forecast_date)]

    print(f"  Available data: {len(rv)} days")
    print(f"  Date range: {rv.index[0].date()} to {rv.index[-1].date()}")

    # Prepare features
    print(f"\n[Features] Computing HAR features...")
    features = prepare_features(rv)
    print(f"  RV_d (1-day):   {features[0, 0]:.6f}")
    print(f"  RV_w (5-day):   {features[0, 1]:.6f}")
    print(f"  RV_m (22-day):  {features[0, 2]:.6f}")

    # Make prediction
    print(f"\n[Prediction] Running ensemble inference...")
    pred, pred_std = predict_ensemble(models, features)

    # Format results
    result = {
        'ticker': ticker,
        'model': model_name,
        'forecast_date': str(forecast_date),
        'horizon_days': HORIZON,
        'prediction': float(pred),
        'uncertainty': float(pred_std),
        'features': {
            'rv_d': float(features[0, 0]),
            'rv_w': float(features[0, 1]),
            'rv_m': float(features[0, 2])
        },
        'num_models': len(models),
        'ensemble_seeds': metadata['seeds']
    }

    print(f"\n{'='*70}")
    print(f"  FORECAST RESULTS")
    print(f"{'='*70}\n")
    print(f"  Ticker:         {ticker}")
    print(f"  Model:          {model_name} ({len(models)} models)")
    print(f"  Forecast Date:  {forecast_date}")
    print(f"  Horizon:        {HORIZON}-day ahead")
    print(f"  ───────────────────────────────────────")
    print(f"  Prediction:     {pred:.6f}")
    print(f"  Uncertainty:    ±{pred_std:.6f} (1 std)")
    print(f"  ───────────────────────────────────────")
    print(f"  Input Features:")
    print(f"    RV_d (1-day):   {features[0, 0]:.6f}")
    print(f"    RV_w (5-day):   {features[0, 1]:.6f}")
    print(f"    RV_m (22-day):  {features[0, 2]:.6f}")
    print(f"\n{'='*70}\n")

    return result

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VIC Ensemble Volatility Forecasting')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L'],
                        help='Model to use for forecasting')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER,
                        help='Stock ticker to forecast')
    parser.add_argument('--date', type=str, default=None,
                        help='Forecast date (YYYY-MM-DD), default=most recent')

    args = parser.parse_args()

    # Run forecast
    result = forecast_volatility(args.model, args.ticker, args.date)

    # Save result
    output_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'vic_forecasts'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{args.ticker}_{args.model}_forecast.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[Saved] Forecast saved to {output_file}\n")
