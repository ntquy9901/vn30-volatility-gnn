"""
Train all GNNHAR models for VIC stock with BREAKTHROUGH CONFIG.

Breakthrough from train_vic_improved.py:
- Walk-Forward validation (train on last 1000 days)
- 1000 epochs training
- Patience = 200
- Stride = 1 (all snapshots)

Result: Walk-Forward HAR achieves R² = -0.20, competitive with HAR OLS (-0.80)

Usage:
    python moirai/gnn/gnnhar_paper/train_vic_all_models.py

Models:
    1. HAR OLS (baseline)
    2. HAR (nn.Module) - Walk-Forward
    3. GHAR - Walk-Forward
    4. GNNHAR1L - Walk-Forward
    5. GNNHAR2L - Walk-Forward
    6. GNNHAR3L - Walk-Forward

Config:
    Horizon: h=5
    Stride: 1 (all snapshots)
    Test from: 2026-01-01
    Walk-Forward window: 1000 days
    Epochs: 1000
    Patience: 200
"""
import warnings
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# =============================================================================
# PATH SETUP
# =============================================================================
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from baselines.har_rv_baseline import fit_har, predict_har
from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY

print("\n" + "="*70)
print("  VIC STOCK: ALL MODELS - BREAKTHROUGH CONFIG")
print("  Walk-Forward + 1000 epochs (R² = -0.20 for HAR)")
print("="*70 + "\n")

# =============================================================================
# CONFIG - BREAKTHROUGH SETTINGS
# =============================================================================
HORIZON = 5
TICKER = 'VIC'
GLOBAL_TEST_START = "2026-01-01"

# BREAKTHROUGH CONFIG from train_vic_improved.py
N_EPOCHS = 1000
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 200  # Increased for 1000 epochs
BATCH_SIZE = 32
N_HID = 16  # Hidden dimension for GCN layers
WALK_FORWARD_WINDOW = 1000  # Train on last 1000 days

# =============================================================================
# LOAD DATA
# =============================================================================
print("[Data] Loading data...")
import yaml
with open(_ROOT / 'config.yaml') as f:
    cfg = yaml.safe_load(f)
DATA_DIR = _ROOT / cfg['data']['prices_dir']

# Load all VN30 prices for adjacency matrix
from gnn.build_graph import VN30_TICKERS
close_all = load_close_prices(DATA_DIR, tickers=VN30_TICKERS)
print(f"  All stocks shape: {close_all.shape}")

# Load VIC specific
close_vic = load_close_prices(DATA_DIR, tickers=[TICKER])
print(f"  {TICKER} shape: {close_vic.shape}")

# =============================================================================
# BUILD SNAPSHOTS WITH STRIDE = 1 (EXACT SAME as train_vic_improved.py)
# =============================================================================
print(f"\n[Data] Building HAR features with STRIDE = 1...")

# Compute log_returns for adjacency matrix
log_returns = compute_log_returns(close_all)

# Compute RV from close prices (EXACT same as train_vic_improved.py)
rv_vic = compute_rv(close_vic, h=HORIZON)[TICKER].dropna()
print(f"  RV (h={HORIZON}): {len(rv_vic)} samples")

def build_snapshots_stride1(rv_series, horizon=5):
    """
    Build HAR snapshots with STRIDE = 1.
    Returns: X (n, 3), y (n,), dates
    """
    # Need at least 22 days for monthly window + horizon days ahead
    min_history = 22 + horizon

    X_list = []
    y_list = []
    date_list = []

    for i in range(min_history, len(rv_series) - horizon):
        # Get current date
        current_date = rv_series.index[i]

        # Target: RV from i to i+horizon
        target = rv_series.iloc[i:i+horizon].mean()

        # Features: look back from day i
        # RV_d: 1 day back (i-1 to i)
        rv_d = rv_series.iloc[i-1:i].mean()

        # RV_w: 5 days back (i-5 to i)
        rv_w = rv_series.iloc[i-5:i].mean()

        # RV_m: 22 days back (i-22 to i)
        rv_m = rv_series.iloc[i-22:i].mean()

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)
        date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

X_vic, y_vic, dates = build_snapshots_stride1(rv_vic, HORIZON)

print(f"  Total snapshots: {len(X_vic)} (stride=1)")
print(f"  X shape: {X_vic.shape}")
print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")

# =============================================================================
# SPLIT DATA - WALK-FORWARD VALIDATION
# =============================================================================
test_ts = pd.Timestamp(GLOBAL_TEST_START)
pre_test_mask = dates < test_ts

X_pre = X_vic[pre_test_mask]
y_pre = y_vic[pre_test_mask]
dates_pre = dates[pre_test_mask]

# =============================================================================
# WALK-FORWARD: Train on last 1000 days
# =============================================================================
X_train_wf = X_pre[-WALK_FORWARD_WINDOW:]
y_train_wf = y_pre[-WALK_FORWARD_WINDOW:]
dates_train_wf = dates_pre[-WALK_FORWARD_WINDOW:]

# Use last 20% of walk-forward window for validation
n_train_wf = int(len(X_train_wf) * 0.8)

X_train = X_train_wf[:n_train_wf]
y_train = y_train_wf[:n_train_wf]
dates_train = dates_train_wf[:n_train_wf]

X_val = X_train_wf[n_train_wf:]
y_val = y_train_wf[n_train_wf:]
dates_val = dates_train_wf[n_train_wf:]

# Test set
test_mask = dates >= test_ts
X_test = X_vic[test_mask]
y_test = y_vic[test_mask]
dates_test = dates[test_mask]

print(f"\n[Split] WALK-FORWARD VALIDATION:")
print(f"  Train: {len(X_train)} samples ({dates_train[0].date()} to {dates_train[-1].date()})")
print(f"  Val:   {len(X_val)} samples ({dates_val[0].date()} to {dates_val[-1].date()})")
print(f"  Test:  {len(X_test)} samples ({dates_test[0].date()})")
print(f"\n[Statistics]")
print(f"  Train mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")
print(f"  Val   mean: {y_val.mean():.6f}, std: {y_val.std():.6f}")
print(f"  Test  mean: {y_test.mean():.6f}, std: {y_test.std():.6f}")

# =============================================================================
# BUILD ADJACENCY MATRIX (for GNN models)
# =============================================================================
print(f"\n[Graph] Building adjacency matrix...")

def build_correlation_adjacency(log_returns, window=252):
    """
    Build adjacency matrix from rolling correlation.

    For each stock pair, compute correlation over rolling window.
    Use mean correlation across time as edge weight.
    """
    n_stocks = len(VN30_TICKERS)
    adj = np.zeros((n_stocks, n_stocks), dtype=np.float32)

    # Compute correlation matrix
    corr_matrix = log_returns[VN30_TICKERS].corr()
    adj = corr_matrix.values.astype(np.float32)

    # Normalize: keep only positive correlations
    adj = np.maximum(adj, 0)

    # Add self-loops
    np.fill_diagonal(adj, 1.0)

    return adj

adj = build_correlation_adjacency(log_returns)
print(f"  Adjacency shape: {adj.shape}")
print(f"  Sparsity: {(adj > 0).sum() / adj.size * 100:.1f}% positive")

# =============================================================================
# METRICS FUNCTION
# =============================================================================
def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def compute_all_metrics(y_true, y_pred, model_name):
    """Compute comprehensive metrics."""
    r2 = compute_r2(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    return {
        'model': model_name,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'pred_mean': y_pred.mean(),
        'pred_std': y_pred.std(),
    }

# =============================================================================
# 1. HAR BASELINE OLS
# =============================================================================
print(f"\n{'='*70}")
print(f"  1. HAR OLS (Baseline)")
print(f"{'='*70}")

# Compute RV for HAR OLS
rv_vic = compute_rv(close_vic, h=HORIZON)[TICKER].dropna()

# Fit HAR OLS
har_coeffs = fit_har(rv_vic, dates_train[-1])
print(f"  Coefficients: {har_coeffs}")

# Predict
har_pred_ols = predict_har(rv_vic, har_coeffs, test_ts)
har_pred_aligned = har_pred_ols.reindex(dates_test).dropna()

if len(har_pred_aligned) > 0:
    y_test_aligned = y_test[:len(har_pred_aligned)]
    metrics_har_ols = compute_all_metrics(y_test_aligned, har_pred_aligned.values, 'HAR_OLS')
    print(f"\n  Results:")
    print(f"    R² = {metrics_har_ols['r2']:+.4f}")
    print(f"    MAE = {metrics_har_ols['mae']:.6f}")
    print(f"    RMSE = {metrics_har_ols['rmse']:.6f}")
else:
    metrics_har_ols = {'model': 'HAR_OLS', 'r2': np.nan, 'mae': np.nan}
    print("  [WARN] No valid predictions")

# =============================================================================
# 2. HAR (nn.Module) - WALK-FORWARD
# =============================================================================
print(f"\n{'='*70}")
print(f"  2. HAR (nn.Module) - WALK-FORWARD")
print(f"{'='*70}")

# Create model (EXACT same as train_vic_improved.py)
class HAR_WF(nn.Module):
    """HAR model with Walk-Forward training."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.linear(x)
        h = self.relu(h)
        return h.squeeze(-1)

def train_model_wf(model, X_train, y_train, X_val, y_val, model_name):
    """Train with Walk-Forward config (1000 epochs, patience=200)."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        X_t = torch.from_numpy(X_train).float()
        y_t = torch.from_numpy(y_train).float()

        pred = model(X_t)
        loss = criterion(pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        # No gradient clipping - like train_vic_improved.py
        optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            X_v = torch.from_numpy(X_val).float()
            y_v = torch.from_numpy(y_val).float()
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Load best
    model.load_state_dict(best_state)
    return model

# Train HAR with Walk-Forward
model_har_wf = HAR_WF()
model_har_wf = train_model_wf(model_har_wf, X_train, y_train, X_val, y_val, 'HAR_WF')

# Evaluate
model_har_wf.eval()
with torch.no_grad():
    X_ts = torch.from_numpy(X_test).float()
    har_pred = model_har_wf(X_ts).numpy()

metrics_har_nn = compute_all_metrics(y_test, har_pred, 'HAR_WF')
print(f"\n  Results:")
print(f"    R² = {metrics_har_nn['r2']:+.4f}")
print(f"    MAE = {metrics_har_nn['mae']:.6f}")
print(f"    RMSE = {metrics_har_nn['rmse']:.6f}")
print(f"    Pred mean: {har_pred.mean():.6f}")

# Check if ReLU killing output
zero_pct = (har_pred == 0).sum() / len(har_pred) * 100
print(f"    Pred % zero: {zero_pct:.1f}%")

# =============================================================================
# BUILD MULTI-STOCK SNAPSHOTS FOR GNN MODELS
# =============================================================================
print(f"\n{'='*70}")
print(f"  Building multi-stock snapshots for GNN models...")
print(f"{'='*70}")

# Build RV for all stocks
print("\n  Building RV for all stocks...")
rv_all_stocks = {}
for ticker in VN30_TICKERS:
    rv_ticker = compute_rv(close_all[[ticker]], h=HORIZON)[ticker].dropna()
    rv_all_stocks[ticker] = rv_ticker

# Build multi-stock snapshots (same method as single-stock)
def build_multi_stock_snapshots(rv_dict, horizon):
    """
    Build snapshots for GNN models using compute_rv.
    All stocks use same date range (intersection).
    """
    # Find common date range
    common_dates = None
    for ticker in VN30_TICKERS:
        if ticker in rv_dict:
            if common_dates is None:
                common_dates = rv_dict[ticker].index
            else:
                common_dates = common_dates.intersection(rv_dict[ticker].index)

    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No common dates found")

    min_history = 22 + horizon
    X_list, y_list, date_list = [], [], []

    for i in range(min_history, len(common_dates) - horizon):
        current_date = common_dates[i]

        # Build feature matrix (30, 3)
        feat_matrix = np.zeros((30, 3), dtype=np.float32)
        target_vector = np.zeros(30, dtype=np.float32)

        valid = True
        for j, ticker in enumerate(VN30_TICKERS):
            rv_series = rv_dict.get(ticker)
            if rv_series is None:
                valid = False
                break

            # Find index of current_date in this stock's RV series
            if current_date not in rv_series.index:
                valid = False
                break

            idx = rv_series.index.get_loc(current_date)

            # Target: RV from i to i+horizon
            if idx + horizon >= len(rv_series):
                valid = False
                break
            target = rv_series.iloc[idx:idx+horizon].mean()

            # Features: look back from day i
            # RV_d: 1 day back (idx-1 to idx)
            if idx < 1:
                valid = False
                break
            feat_d = rv_series.iloc[idx-1:idx].mean()

            # RV_w: 5 days back (idx-5 to idx)
            if idx < 5:
                valid = False
                break
            feat_w = rv_series.iloc[idx-5:idx].mean()

            # RV_m: 22 days back (idx-22 to idx)
            if idx < 22:
                valid = False
                break
            feat_m = rv_series.iloc[idx-22:idx].mean()

            if np.isnan([feat_d, feat_w, feat_m, target]).any():
                valid = False
                break

            feat_matrix[j, :] = [feat_d, feat_w, feat_m]
            target_vector[j] = target

        if valid:
            X_list.append(feat_matrix)
            y_list.append(target_vector)
            date_list.append(current_date)

    return np.array(X_list), np.array(y_list), pd.Index(date_list)

# Build multi-stock snapshots
X_multi, y_multi, dates_multi = build_multi_stock_snapshots(rv_all_stocks, HORIZON)
print(f"  Multi-stock snapshots: {len(X_multi)}")

# =============================================================================
# WALK-FORWARD SPLIT FOR MULTI-STOCK DATA
# =============================================================================
pre_test_mask_multi = dates_multi < test_ts
X_pre_multi = X_multi[pre_test_mask_multi]
y_pre_multi = y_multi[pre_test_mask_multi]
dates_pre_multi = dates_multi[pre_test_mask_multi]

# Walk-Forward: Last 1000 days
X_train_multi_wf = X_pre_multi[-WALK_FORWARD_WINDOW:]
y_train_multi_wf = y_pre_multi[-WALK_FORWARD_WINDOW:]

n_train_multi_wf = int(len(X_train_multi_wf) * 0.8)

X_train_multi = X_train_multi_wf[:n_train_multi_wf]
y_train_multi = y_train_multi_wf[:n_train_multi_wf]

X_val_multi = X_train_multi_wf[n_train_multi_wf:]
y_val_multi = y_train_multi_wf[n_train_multi_wf:]

test_mask_multi = dates_multi >= test_ts
X_test_multi = X_multi[test_mask_multi]
y_test_multi = y_multi[test_mask_multi]
dates_test_multi = dates_multi[test_mask_multi]

# Get VIC index
vic_idx = VN30_TICKERS.index(TICKER)
print(f"  VIC index: {vic_idx}")

# Convert to tensors for GNN training
X_train_t = torch.from_numpy(X_train_multi).float()
y_train_t = torch.from_numpy(y_train_multi).float()
X_val_t = torch.from_numpy(X_val_multi).float()
y_val_t = torch.from_numpy(y_val_multi).float()
X_test_t = torch.from_numpy(X_test_multi).float()
adj_t = torch.from_numpy(adj).float()

# =============================================================================
# 3. GHAR - WALK-FORWARD
# =============================================================================
print(f"\n{'='*70}")
print(f"  3. GHAR - WALK-FORWARD")
print(f"{'='*70}")

ghar_model = MODEL_REGISTRY['GHAR'](n_hid=N_HID)

def train_gnn_wf(model, X_train_t, y_train_t, X_val_t, y_val_t, adj_t, model_name):
    """Train GNN with Walk-Forward config."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        pred = model(X_train_t, adj_t)
        loss = criterion(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t, adj_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model

ghar_model = train_gnn_wf(ghar_model, X_train_t, y_train_t, X_val_t, y_val_t, adj_t, 'GHAR')

ghar_model.eval()
with torch.no_grad():
    ghar_pred = ghar_model(X_test_t, adj_t).numpy()

ghar_pred_vic = ghar_pred[:, vic_idx]
y_test_vic_multi = y_test_multi[:, vic_idx]

metrics_ghar = compute_all_metrics(y_test_vic_multi, ghar_pred_vic, 'GHAR_WF')
print(f"\n  VIC Results:")
print(f"    R² = {metrics_ghar['r2']:+.4f}")
print(f"    MAE = {metrics_ghar['mae']:.6f}")
print(f"    RMSE = {metrics_ghar['rmse']:.6f}")

# =============================================================================
# 4. GNNHAR1L - WALK-FORWARD
# =============================================================================
print(f"\n{'='*70}")
print(f"  4. GNNHAR1L - WALK-FORWARD")
print(f"{'='*70}")

gnn1l_model = MODEL_REGISTRY['GNNHAR1L'](n_hid=N_HID)
gnn1l_model = train_gnn_wf(gnn1l_model, X_train_t, y_train_t, X_val_t, y_val_t, adj_t, 'GNNHAR1L')

gnn1l_model.eval()
with torch.no_grad():
    gnn1l_pred = gnn1l_model(X_test_t, adj_t).numpy()

gnn1l_pred_vic = gnn1l_pred[:, vic_idx]
metrics_gnn1l = compute_all_metrics(y_test_vic_multi, gnn1l_pred_vic, 'GNNHAR1L_WF')
print(f"\n  VIC Results:")
print(f"    R² = {metrics_gnn1l['r2']:+.4f}")
print(f"    MAE = {metrics_gnn1l['mae']:.6f}")
print(f"    RMSE = {metrics_gnn1l['rmse']:.6f}")

# =============================================================================
# 5. GNNHAR2L - WALK-FORWARD
# =============================================================================
print(f"\n{'='*70}")
print(f"  5. GNNHAR2L - WALK-FORWARD")
print(f"{'='*70}")

gnn2l_model = MODEL_REGISTRY['GNNHAR2L'](n_hid=N_HID)
gnn2l_model = train_gnn_wf(gnn2l_model, X_train_t, y_train_t, X_val_t, y_val_t, adj_t, 'GNNHAR2L')

gnn2l_model.eval()
with torch.no_grad():
    gnn2l_pred = gnn2l_model(X_test_t, adj_t).numpy()

gnn2l_pred_vic = gnn2l_pred[:, vic_idx]
metrics_gnn2l = compute_all_metrics(y_test_vic_multi, gnn2l_pred_vic, 'GNNHAR2L_WF')
print(f"\n  VIC Results:")
print(f"    R² = {metrics_gnn2l['r2']:+.4f}")
print(f"    MAE = {metrics_gnn2l['mae']:.6f}")
print(f"    RMSE = {metrics_gnn2l['rmse']:.6f}")

# =============================================================================
# 6. GNNHAR3L - WALK-FORWARD
# =============================================================================
print(f"\n{'='*70}")
print(f"  6. GNNHAR3L - WALK-FORWARD")
print(f"{'='*70}")

gnn3l_model = MODEL_REGISTRY['GNNHAR3L'](n_hid=N_HID)
gnn3l_model = train_gnn_wf(gnn3l_model, X_train_t, y_train_t, X_val_t, y_val_t, adj_t, 'GNNHAR3L')

gnn3l_model.eval()
with torch.no_grad():
    gnn3l_pred = gnn3l_model(X_test_t, adj_t).numpy()

gnn3l_pred_vic = gnn3l_pred[:, vic_idx]
metrics_gnn3l = compute_all_metrics(y_test_vic_multi, gnn3l_pred_vic, 'GNNHAR3L_WF')
print(f"\n  VIC Results:")
print(f"    R² = {metrics_gnn3l['r2']:+.4f}")
print(f"    MAE = {metrics_gnn3l['mae']:.6f}")
print(f"    RMSE = {metrics_gnn3l['rmse']:.6f}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print(f"\n{'='*70}")
print(f"  SUMMARY: ALL MODELS FOR VIC (h={HORIZON})")
print(f"  Config: Walk-Forward + 1000 epochs")
print(f"{'='*70}")
print(f"{'Model':<20} {'R2':>10} {'MAE':>12} {'Beat OLS?'}")
print(f"{'-'*50}")

all_metrics = [
    metrics_har_ols,
    metrics_har_nn,
    metrics_ghar,
    metrics_gnn1l,
    metrics_gnn2l,
    metrics_gnn3l,
]

ols_r2 = metrics_har_ols.get('r2', 0)

for m in all_metrics:
    model = m['model']
    r2 = m.get('r2', np.nan)
    mae = m.get('mae', np.nan)
    beat = 'YES' if not np.isnan(r2) and r2 > ols_r2 else 'NO'
    print(f"{model:<20} {r2:>+10.4f} {mae:12.6f} {beat}")

print(f"{'-'*50}")

# Best model by R²
best_model = max(all_metrics, key=lambda x: x.get('r2', -np.inf))
print(f"\n  Best model: {best_model['model']} (R² = {best_model['r2']:+.4f})")

# Comparison with OLS
diff_from_ols = best_model['r2'] - ols_r2
print(f"  Improvement vs OLS: {diff_from_ols:+.4f}")

if best_model['r2'] > ols_r2:
    print(f"  *** SUCCESS: GNN/HAR beats HAR OLS! ***")
else:
    print(f"  INFO: HAR OLS remains baseline")

print(f"\n{'='*70}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = _ROOT / 'results' / 'gnnhar_paper' / 'vic_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'ticker': TICKER,
    'horizon': HORIZON,
    'stride': 1,
    'test_start': GLOBAL_TEST_START,
    'walk_forward_window': WALK_FORWARD_WINDOW,
    'n_epochs': N_EPOCHS,
    'patience': PATIENCE,
    'n_train': int(len(X_train)),
    'n_val': int(len(X_val)),
    'n_test': int(len(X_test)),
    'train_mean': float(y_train.mean()),
    'test_mean': float(y_test.mean()),
    'models': {m['model']: {'r2': float(m.get('r2', np.nan)), 'mae': float(m.get('mae', np.nan)), 'rmse': float(m.get('rmse', np.nan))} for m in all_metrics},
    'best_model': best_model['model'],
    'improvement_vs_ols': float(diff_from_ols),
}

with open(output_dir / f'vic_h{HORIZON}_walkforward_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] Results saved to {output_dir / f'vic_h{HORIZON}_walkforward_results.json'}")
print("\n" + "="*70)
