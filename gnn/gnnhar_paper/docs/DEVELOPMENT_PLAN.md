# Multi-Stock GNNHAR Development Plan

**Date:** 2026-05-31
**Status:** Ready for implementation
**Duration:** 8 days (3 phases)

---

## Executive Summary

This plan implements the complete GNNHAR paper architecture for VN30 volatility forecasting using multi-stock training (30 stocks). Based on analysis of the original code at https://github.com/chaozhang-ox/GNNHAR, this plan addresses the critical failure of single-stock training (R² = -0.14 to -12.92) by implementing proper multi-stock batching and graph construction.

**Key Clarification from Original Code:**
- **GHAR does NOT use sklearn LinearRegression** - it's a PyTorch neural network with GCN layer
- **GHAR vs GNNHAR:** GHAR has linear graph branch (GCN only), GNNHAR has nonlinear branch (GCN + ReLU + MLP)

---

## Phase 1: Data Pipeline (Days 1-2)

### 1.1 Multi-Stock Data Loader Module

**File:** `gnn/gnnhar_paper/data_loader.py`

**Requirements:**
```python
class MultiStockDataLoader:
    """Load and prepare multi-stock HAR dataset for GNNHAR training"""
    
    def __init__(self, tickers, horizon=5, train_end="2025-12-31"):
        """
        Args:
            tickers: List of 30 VN30 stock symbols
            horizon: Forecasting horizon (default 5 days)
            train_end: Training data end date (temporal split boundary)
        """
        self.tickers = tickers
        self.horizon = horizon
        self.train_end = pd.Timestamp(train_end)
    
    def load_close_prices(self, prices_dir):
        """Load OHLCV data and extract close prices
        
        Returns:
            pd.DataFrame: (2500 days, 30 stocks) close prices
        """
        from src.volatility_labels import load_close_prices
        return load_close_prices(prices_dir, tickers=self.tickers)
    
    def compute_rv(self, close):
        """Compute realized volatility
        
        Formula: RV_t(h) = std(log_return[t:t+h])
        
        Args:
            close: (2500, 30) close prices
        
        Returns:
            pd.DataFrame: (2500, 30) RV values per stock per day
        """
        from src.volatility_labels import compute_rv
        return compute_rv(close, h=self.horizon)
    
    def build_har_features(self, rv):
        """Build HAR features per stock
        
        Features: [RV_d, RV_w, RV_m] where:
        - RV_d = lag1 (daily)
        - RV_w = average of past 5 days
        - RV_m = average of past 22 days
        
        Args:
            rv: (2500, 30) RV values
        
        Returns:
            dict: {ticker: (2500, 3) HAR features DataFrame}
        """
        features_dict = {}
        for ticker in self.tickers:
            rv_series = rv[ticker]
            rv_d = rv_series.shift(1)
            rv_w = rv_series.shift(1).rolling(5, min_periods=5).mean()
            rv_m = rv_series.shift(1).rolling(22, min_periods=22).mean()
            
            features_dict[ticker] = pd.DataFrame({
                'RV_d': rv_d,
                'RV_w': rv_w,
                'RV_m': rv_m
            })
        return features_dict
    
    def flatten_dataset(self, features_dict, rv):
        """Flatten multi-stock data to (N_stocks × N_dates, 3) format
        
        Creates flattened dataset where each row is a stock-date sample.
        Format: [stock_0_day_0, stock_0_day_1, ..., stock_0_day_T,
                 stock_1_day_0, ..., stock_29_day_T]
        
        Args:
            features_dict: {ticker: (2500, 3) features}
            rv: (2500, 30) RV values
        
        Returns:
            X: (60000, 3) flattened HAR features
            y: (60000,) flattened RV targets
            stock_indices: (60000,) stock ID for each sample
            date_indices: (60000,) date ID for each sample
        """
        all_samples = []
        all_targets = []
        stock_indices = []
        date_indices = []
        
        for stock_id, ticker in enumerate(self.tickers):
            feats = features_dict[ticker]
            targets = rv[ticker]
            
            # Align and drop NaN
            valid_idx = feats.dropna().index.intersection(targets.dropna().index)
            
            for date_id, date in enumerate(valid_idx):
                all_samples.append(feats.loc[date].values)
                all_targets.append(targets.loc[date])
                stock_indices.append(stock_id)
                date_indices.append(date_id)
        
        X = np.array(all_samples)
        y = np.array(all_targets)
        stock_indices = np.array(stock_indices)
        date_indices = np.array(date_indices)
        
        return X, y, stock_indices, date_indices
    
    def split_train_val_test(self, X, y, stock_indices, date_indices):
        """Temporal split with validation set
        
        Split: 80% train, 20% validation (from pre-2026 data)
               Test: all 2026 data
        
        Args:
            X, y, stock_indices, date_indices: Flattened arrays
        
        Returns:
            train_X, train_y, train_stocks, train_dates
            val_X, val_y, val_stocks, val_dates
            test_X, test_y, test_stocks, test_dates
        """
        # Find dates before and after train_end
        train_mask = date_indices < self.get_train_end_index()
        test_mask = date_indices >= self.get_train_end_index()
        
        # Split train into train/val (80/20)
        train_indices = np.where(train_mask)[0]
        split_point = int(len(train_indices) * 0.8)
        
        train_idx = train_indices[:split_point]
        val_idx = train_indices[split_point:]
        test_idx = np.where(test_mask)[0]
        
        return (X[train_idx], y[train_idx], stock_indices[train_idx], date_indices[train_idx]), \
               (X[val_idx], y[val_idx], stock_indices[val_idx], date_indices[val_idx]), \
               (X[test_idx], y[test_idx], stock_indices[test_idx], date_indices[test_idx])
    
    def get_train_end_index(self):
        """Helper to get date index for train_end boundary"""
        # Implementation depends on date indexing
        return 1900  # Placeholder, needs actual date mapping
```

**Success criteria:**
- Data loader loads all 30 stocks correctly
- Flattened dataset has ~60000 samples
- Train/val/test splits are temporally consistent

---

### 1.2 Graph Construction Module

**File:** `gnn/gnnhar_paper/graph_builder.py`

**Requirements:**
```python
class GraphBuilder:
    """Construct adjacency matrix for VN30 volatility spillover network"""
    
    def __init__(self, method='pearson', threshold=0.3):
        """
        Args:
            method: 'pearson' (simple) or 'glasso' (paper's method)
            threshold: Correlation threshold for edges (pearson only)
        """
        self.method = method
        self.threshold = threshold
    
    def compute_returns(self, close):
        """Compute daily returns from close prices
        
        Args:
            close: (2500, 30) close prices
        
        Returns:
            returns: (2499, 30) daily returns
        """
        return close.pct_change().dropna()
    
    def build_pearson_adjacency(self, returns):
        """Build adjacency from Pearson correlation
        
        Process:
        1. Compute correlation matrix
        2. Apply threshold (keep only strong correlations)
        3. Normalize row-wise (sum to 1)
        
        Args:
            returns: (2499, 30) daily returns
        
        Returns:
            adj: (30, 30) normalized adjacency matrix
        """
        # Step 1: Correlation matrix
        corr = returns.corr()
        
        # Step 2: Threshold
        adj = (corr.abs() >= self.threshold).astype(float)
        np.fill_diagonal(adj.values, 0)  # No self-loops
        
        # Step 3: Normalize row-wise
        adj = adj.div(adj.sum(axis=1), axis=0)
        
        return adj.values
    
    def build_glasso_adjacency(self, returns):
        """Build adjacency using GLASSO (paper's method)
        
        From original code (lines 223-234):
        - Use GraphicalLassoCV to estimate precision matrix
        - Create adjacency from non-zero partial correlations
        - Normalize with symmetric scaling
        
        Args:
            returns: (2499, 30) daily returns
        
        Returns:
            adj: (30, 30) normalized adjacency matrix
        """
        from sklearn.covariance import GraphicalLassoCV
        
        # Step 1: GLASSO
        model = GraphicalLassoCV()
        model.fit(returns)
        
        # Step 2: Precision matrix -> adjacency
        prec = model.precision_
        corr = (prec != 0)
        corr_adj = corr - np.identity(30)  # Remove diagonal
        
        # Step 3: Symmetric normalization (from paper line 230)
        d_sqrt_inv = np.diag(np.sqrt(1 / (corr_adj.sum(1) + 1e-8)))
        adj = np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv)
        
        return adj
    
    def build_adjacency(self, close):
        """Main method to build adjacency matrix
        
        Args:
            close: (2500, 30) close prices
        
        Returns:
            adj: (30, 30) adjacency matrix
        """
        returns = self.compute_returns(close)
        
        if self.method == 'pearson':
            return self.build_pearson_adjacency(returns)
        elif self.method == 'glasso':
            return self.build_glasso_adjacency(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def visualize_graph(self, adj, ticker_list, save_path):
        """Visualize adjacency matrix (optional, for debugging)
        
        Args:
            adj: (30, 30) adjacency
            ticker_list: List of 30 stock symbols
            save_path: Where to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(adj, cmap='viridis', aspect='auto')
        plt.colorbar(label='Edge weight')
        plt.xticks(range(30), ticker_list, rotation=90)
        plt.yticks(range(30), ticker_list)
        plt.title('Volatility Spillover Network Adjacency Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
```

**Success criteria:**
- Pearson adjacency produces sparse graph (threshold=0.3)
- GLASSO adjacency produces paper-style graph
- Adjacency is normalized (rows sum to 1)
- Visualization shows sensible stock clusters

---

## Phase 2: Training Infrastructure (Days 3-4)

### 2.1 Forward Pass with Stock Masking

**Critical component from original code (lines 102-105):**

```python
def forward_pass_with_mask(model, batch_X, adj, batch_stocks, n_stocks=30):
    """Forward pass with stock masking for multi-stock batches
    
    Key insight from original code: Each batch contains subset of stocks,
    but model expects (batch, N, features) where N=30 (all stocks).
    Solution: Create masked node_feat matrix with zeros for non-batch stocks.
    
    Args:
        model: GNNHAR model (HAR, GHAR, GNNHAR1L, etc.)
        batch_X: (batch_size, 3) - HAR features for samples in batch
        adj: (30, 30) - full adjacency matrix
        batch_stocks: (batch_size,) - stock ID for each sample
        n_stocks: Total number of stocks (default 30)
    
    Returns:
        predictions: (batch_size,) - predictions for actual samples
    """
    batch_size = batch_X.shape[0]
    
    # Step 1: Create node_feat matrix (batch, N, 3)
    # Initialize with zeros
    node_feat = torch.zeros(batch_size, n_stocks, 3)
    
    # Place actual features in correct stock positions
    for i in range(batch_size):
        stock_id = batch_stocks[i].item()
        node_feat[i, stock_id, :] = batch_X[i, :]
    
    # Step 2: Forward through model
    # Model expects: node_feat (batch, N, 3), adj (N, N)
    # Returns: predictions (batch, N) - one per stock
    predictions = model(node_feat, adj)
    
    # Step 3: Extract predictions for actual stocks in batch
    # Each sample in batch corresponds to one stock
    batch_predictions = predictions[torch.arange(batch_size), batch_stocks]
    
    return batch_predictions
```

**Why this is critical:**
- GCN requires (batch, N, features) where N = all stocks
- But batch only contains subset of stocks (random 128 samples)
- Masking ensures model sees all stocks but only gradients for batch stocks

### 2.2 Training Loop Module

**File:** `gnn/gnnhar_paper/train_multi_stock.py`

```python
def train_single_model(model, train_loader, val_loader, adj, n_stocks=30,
                       epochs=1500, lr=1e-3, weight_decay=1e-5, seed=42):
    """Train a single model with early stopping
    
    Based on original code (lines 347-410)
    
    Args:
        model: GNNHAR model instance
        train_loader: DataLoader with shuffled batches
        val_loader: DataLoader with validation data
        adj: (30, 30) adjacency matrix
        n_stocks: Number of stocks (30)
        epochs: Max training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        seed: Random seed for reproducibility
    
    Returns:
        trained_model: Model with best validation weights
        val_loss: Best validation loss achieved
        train_history: Training curve data
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    adj = torch.from_numpy(adj).float().to(device)
    
    # Optimizer (AdamW from paper)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss function
    criterion = quasi_likelihood_loss
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 150
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = []
        
        for batch_X, batch_y, batch_stocks, _ in train_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_stocks = batch_stocks.to(device)
            
            # Forward pass with masking
            pred = forward_pass_with_mask(model, batch_X, adj, batch_stocks, n_stocks)
            
            # Compute loss
            loss = criterion(pred.unsqueeze(1), batch_y.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss.append(loss.item())
        
        train_loss = np.mean(epoch_train_loss)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_stocks, _ in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_stocks = batch_stocks.to(device)
                
                pred = forward_pass_with_mask(model, batch_X, adj, batch_stocks, n_stocks)
                loss = criterion(pred.unsqueeze(1), batch_y.unsqueeze(1))
                
                epoch_val_loss.append(loss.item())
        
        val_loss = np.mean(epoch_val_loss)
        val_losses.append(val_loss)
        
        # Progress logging (every 10% of epochs)
        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    train_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_epochs': len(train_losses)
    }
    
    return model, best_val_loss, train_history


def train_ensemble(models_to_train, train_loader, val_loader, test_loader, adj,
                  stock_indices_dict, n_seeds=20, n_hid=16):
    """Train ensemble of models with different seeds
    
    Based on original code (lines 412-440, ensemble training loop)
    
    Args:
        models_to_train: List of model names ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']
        train_loader, val_loader, test_loader: DataLoaders
        adj: (30, 30) adjacency matrix
        stock_indices_dict: Dict with train/test stock indices
        n_seeds: Number of models per model variant (default 20)
        n_hid: Hidden dimension for GCN layers
    
    Returns:
        results: Dict with ensemble predictions and metrics
    """
    from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n[{model_name}] Training ensemble of {n_seeds} models...")
        
        # Generate random seeds
        seeds = [42, 123, 456, 789, 321, 111, 222, 333, 444, 555,
                 666, 777, 888, 999, 101, 202, 303, 404, 505, 606][:n_seeds]
        
        model_predictions = []
        model_val_losses = []
        train_histories = []
        
        # Train n_seeds models
        for seed_idx, seed in enumerate(seeds):
            # Create model
            if model_name == 'HAR':
                model = MODEL_REGISTRY[model_name]()
            else:
                model = MODEL_REGISTRY[model_name](n_hid=n_hid)
            
            # Train
            trained_model, val_loss, train_history = train_single_model(
                model, train_loader, val_loader, adj, seed=seed
            )
            
            # Test predictions
            trained_model.eval()
            test_preds = []
            
            with torch.no_grad():
                for batch_X, batch_y, batch_stocks, _ in test_loader:
                    batch_X = batch_X.to(trained_model.parameters().__next__().device)
                    batch_stocks = batch_stocks.to(trained_model.parameters().__next__().device)
                    
                    pred = forward_pass_with_mask(trained_model, batch_X, adj, batch_stocks)
                    test_preds.append(pred.cpu().numpy())
            
            model_predictions.append(np.concatenate(test_preds))
            model_val_losses.append(val_loss)
            train_histories.append(train_history)
            
            print(f"    Seed {seed}: val_loss={val_loss:.6f}")
        
        # Screen by validation loss (keep top 50%)
        median_val_loss = np.median(model_val_losses)
        screened_indices = [i for i, vl in enumerate(model_val_losses) if vl <= median_val_loss]
        screened_preds = [model_predictions[i] for i in screened_indices]
        
        print(f"  Screening: kept {len(screened_preds)}/{len(model_predictions)} models")
        
        # Ensemble average
        ensemble_pred = np.mean(screened_preds, axis=0)
        
        # Compute metrics
        y_test = get_test_targets(test_loader)
        r2 = 1 - np.sum((y_test - ensemble_pred)**2) / np.sum((y_test - y_test.mean())**2)
        mae = np.mean(np.abs(y_test - ensemble_pred))
        rmse = np.sqrt(np.mean((y_test - ensemble_pred)**2))
        
        print(f"  Ensemble ({len(screened_preds)} models): R² = {r2:+.4f}, MAE = {mae:.6f}")
        
        results[model_name] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_models': len(screened_preds),
            'val_losses': model_val_losses,
            'predictions': ensemble_pred
        }
    
    return results


def get_test_targets(test_loader):
    """Extract test targets from DataLoader"""
    targets = []
    for _, batch_y, _, _ in test_loader:
        targets.append(batch_y.numpy())
    return np.concatenate(targets)
```

**Success criteria:**
- Training loop completes without crashes
- Early stopping triggers appropriately
- Ensemble screening works (keeps top 50%)
- Predictions have reasonable R² values

---

## Phase 3: Full Training and Evaluation (Days 5-8)

### 3.1 Main Training Script

**File:** `gnn/gnnhar_paper/run_multi_stock_ensemble.py`

```python
"""
Multi-Stock GNNHAR Ensemble Training for VN30 Volatility Forecasting

This script replicates the GNNHAR paper architecture on VN30 data using
multi-stock training (30 stocks) with proper graph construction.

Expected results (based on paper):
- HAR OLS: R² ≈ 0.60-0.70 (baseline)
- GHAR: R² ≈ 0.65-0.75 (linear spillover)
- GNNHAR1L: R² ≈ 0.70-0.80 (nonlinear spillover, best model)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
from gnn.gnnhar_paper.graph_builder import GraphBuilder
from gnn.gnnhar_paper.train_multi_stock import train_ensemble, get_test_targets
from gnn.build_graph import VN30_TICKERS
from baselines.har_rv_baseline import run_har_baseline

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS = VN30_TICKERS  # 30 stocks
HORIZON = 5
TRAIN_END_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-05-31"

MODELS_TO_TRAIN = ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']
N_SEEDS = 20
N_HID = 16
N_EPOCHS = 1500
LR = 1e-3
WEIGHT_DECAY = 1e-5  # Paper's value (not 1e-3!)
BATCH_SIZE = 128

GRAPH_METHOD = 'glasso'  # 'glasso' (paper) or 'pearson' (simpler)
GRAPH_THRESHOLD = 0.3  # For pearson method

# =============================================================================
# MAIN PIPELINE
# =============================================================================

print("\n" + "="*70)
print("  MULTI-STOCK GNNHAR ENSEMBLE TRAINING (30 STOCKS)")
print("="*70 + "\n")

# Step 1: Load data
print("[Step 1] Loading multi-stock data...")
data_loader = MultiStockDataLoader(
    tickers=TICKERS,
    horizon=HORIZON,
    train_end=TRAIN_END_DATE
)

close = data_loader.load_close_prices(PROJECT_ROOT / 'data' / 'raw' / 'prices')
rv = data_loader.compute_rv(close)
print(f"  Loaded {len(TICKERS)} stocks × {len(close)} days")
print(f"  RV shape: {rv.shape}")

# Step 2: Build HAR features
print("\n[Step 2] Building HAR features...")
features_dict = data_loader.build_har_features(rv)
print(f"  Features per stock: {list(features_dict.values())[0].shape}")

# Step 3: Flatten dataset
print("\n[Step 3] Flattening multi-stock dataset...")
X, y, stock_indices, date_indices = data_loader.flatten_dataset(features_dict, rv)
print(f"  Flattened dataset: {X.shape[0]} samples ({len(TICKERS)} stocks × ~{X.shape[0]//len(TICKERS)} dates)")

# Step 4: Split data
print("\n[Step 4] Splitting train/val/test...")
train_data, val_data, test_data = data_loader.split_train_val_test(
    X, y, stock_indices, date_indices
)
print(f"  Train: {len(train_data[0])} samples")
print(f"  Val:   {len(val_data[0])} samples")
print(f"  Test:  {len(test_data[0])} samples")

# Step 5: Build adjacency matrix
print(f"\n[Step 5] Building adjacency matrix ({GRAPH_METHOD} method)...")
graph_builder = GraphBuilder(method=GRAPH_METHOD, threshold=GRAPH_THRESHOLD)
adj = graph_builder.build_adjacency(close)
print(f"  Adjacency shape: {adj.shape}")
print(f"  Sparsity: {(adj == 0).sum() / adj.size:.2%}")

# Step 6: Create DataLoaders
print("\n[Step 6] creating DataLoaders...")
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    torch.from_numpy(train_data[0]).float(),
    torch.from_numpy(train_data[1]).float(),
    torch.from_numpy(train_data[2]).long(),
    torch.from_numpy(train_data[3]).long()
)

val_dataset = TensorDataset(
    torch.from_numpy(val_data[0]).float(),
    torch.from_numpy(val_data[1]).float(),
    torch.from_numpy(val_data[2]).long(),
    torch.from_numpy(val_data[3]).long()
)

test_dataset = TensorDataset(
    torch.from_numpy(test_data[0]).float(),
    torch.from_numpy(test_data[1]).float(),
    torch.from_numpy(test_data[2]).long(),
    torch.from_numpy(test_data[3]).long()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Step 7: Train ensemble
print("\n" + "="*70)
print("  ENSEMBLE TRAINING")
print("="*70 + "\n")

stock_indices_dict = {
    'train': train_data[2],
    'test': test_data[2]
}

results = train_ensemble(
    models_to_train=MODELS_TO_TRAIN,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    adj=adj,
    stock_indices_dict=stock_indices_dict,
    n_seeds=N_SEEDS,
    n_hid=N_HID
)

# Step 8: Compute HAR OLS baseline
print("\n[HAR OLS Baseline] Computing per-stock HAR OLS...")
har_ols_results = run_har_baseline(
    prices_dir=str(PROJECT_ROOT / 'data' / 'raw' / 'prices'),
    train_end=TRAIN_END_DATE,
    test_start=TEST_START_DATE,
    horizon=HORIZON,
    tickers=TICKERS
)

# Aggregate HAR OLS predictions
har_ols_preds_all = []
y_test_all = []

for ticker in TICKERS:
    if ticker in har_ols_results:
        har_ols_preds_all.extend(har_ols_results[ticker].values)
        y_test_all.extend(rv[ticker].loc[TEST_START_DATE:TEST_END_DATE].values)

har_ols_preds = np.array(har_ols_preds_all)
y_test_har = np.array(y_test_all)

# Scale HAR predictions by horizon (to match neural network targets)
har_ols_preds = har_ols_preds / HORIZON
y_test_har = y_test_har / HORIZON

# Compute HAR OLS metrics
ss_res = np.sum((y_test_har - har_ols_preds)**2)
ss_tot = np.sum((y_test_har - y_test_har.mean())**2)
har_ols_r2 = 1 - (ss_res / (ss_tot + 1e-8))
har_ols_mae = np.mean(np.abs(y_test_har - har_ols_preds))

print(f"  HAR OLS: R² = {har_ols_r2:+.4f}, MAE = {har_ols_mae:.6f}")

results['HAR_OLS'] = {
    'r2': float(har_ols_r2),
    'mae': float(har_ols_mae),
    'n_models': 1
}

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("  SUMMARY: MULTI-STOCK ENSEMBLE RESULTS")
print("="*70 + "\n")

print(f"Architecture: {len(TICKERS)} stocks with {GRAPH_METHOD} graph")
print(f"Training: {len(train_data[0])} samples ({TRAIN_END_DATE} end)")
print(f"Test: {len(test_data[0])} samples ({TEST_START_DATE} to {TEST_END_DATE})")
print(f"Ensemble: {N_SEEDS} models per variant, screened by val_loss")
print(f"\nModel Performance:")
print(f"{'Model':<15} {'R2':>10} {'MAE':>12} {'Improvement':>15}")
print("-"*60)

baseline_r2 = results['HAR_OLS']['r2']

for model_name in MODELS_TO_TRAIN + ['HAR_OLS']:
    if model_name in results:
        r2 = results[model_name]['r2']
        mae = results[model_name]['mae']
        improvement = r2 - baseline_r2
        print(f"{model_name:<15} {r2:>+10.4f} {mae:>12.6f} {improvement:>+15.4f}")

print("\n" + "="*70 + "\n")

# Save results
output_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'multi_stock_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'multi_stock_ensemble_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'architecture': 'multi_stock',
        'n_stocks': len(TICKERS),
        'graph_method': GRAPH_METHOD,
        'train_period': f"~{TRAIN_END_DATE}",
        'test_period': f"{TEST_START_DATE} to {TEST_END_DATE}",
        'n_train_samples': len(train_data[0]),
        'n_test_samples': len(test_data[0]),
        'ensemble_size': N_SEEDS,
        'results': results
    }, f, indent=2)

print(f"[Saved] Results saved to {output_file}\n")
```

---

## Implementation Checklist

### Day 1
- [ ] Create `data_loader.py` module
- [ ] Test data loading with 30 stocks
- [ ] Verify flattened dataset shape (~60000, 3)

### Day 2
- [ ] Create `graph_builder.py` module
- [ ] Implement Pearson adjacency
- [ ] Implement GLASSO adjacency
- [ ] Visualize adjacency matrices

### Day 3
- [ ] Implement `forward_pass_with_mask()`
- [ ] Test masking with synthetic data
- [ ] Verify dimensions match paper

### Day 4
- [ ] Implement `train_single_model()`
- [ ] Implement ensemble training loop
- [ ] Test with 2-3 seeds (sanity check)

### Day 5
- [ ] Create main training script
- [ ] Run full training (5 models × 20 seeds)
- [ ] Monitor for stability issues

### Day 6
- [ ] Compute HAR OLS baseline
- [ ] Aggregate results
- [ ] Save predictions and models

### Day 7
- [ ] Generate evaluation metrics
- [ ] Create prediction plots
- [ ] Analyze graph spillover effects

### Day 8
- [ ] Write results summary
- [ ] Compare GHAR vs GNNHAR
- [ ] Update thesis chapter

---

## Success Criteria

### Minimum Success
- All 5 models train without crashes
- HAR OLS achieves R² > 0.60 (baseline verification)
- At least 2 neural models beat HAR OLS (R² > HAR_OLS)

### Expected Success
- HAR OLS achieves R² ≈ 0.65-0.70
- GHAR achieves R² ≈ 0.68-0.72
- GNNHAR1L achieves R² ≈ 0.72-0.78
- Clear hierarchy: HAR < GHAR < GNNHAR1L

### Exceptional Success
- GNNHAR1L achieves R² > 0.80
- All models beat HAR OLS
- Training stable (≤30% seed failures)
- Graph shows clear sector clusters

---

## Comparison: GHAR vs GNNHAR

Based on original code analysis (lines 149-197):

### GHAR (Linear Graph Spillover)
```python
class GHAR(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1)       # H1: Local HAR
        self.gcn1 = GraphConvLayer(3, n_hid) # H2: GCN only
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)  # (batch, N, 1)
        H2 = self.gcn1(node_feat, adj) # (batch, N, n_hid) - NO NONLINEARITY
        res = H1 + H2                   # DIMENSION MISMATCH BUG IN ORIGINAL
        res = self.relu(res)
        return res.squeeze(-1)
```

**Key characteristics:**
- GCN layer but NO ReLU/MLP after it
- Linear spillover from neighbors
- Tests if **any** graph information helps

### GNNHAR1L (Nonlinear Graph Spillover)
```python
class GNNHAR1L(nn.Module):
    def __init__(self, n_hid):
        self.linear1 = nn.Linear(3, 1)       # H1: Local HAR
        self.gcn1 = GraphConvLayer(3, n_hid) # H2: GCN
        self.mlp1 = nn.Linear(n_hid, 1)      # H2: MLP projection
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        H2 = self.gcn1(node_feat, adj)
        H2 = self.relu(H2)                    # NONLINEARITY
        H2 = self.mlp1(H2)                    # PROJECTION
        res = H1 + H2
        res = self.relu(res)
        return res.squeeze(-1)
```

**Key characteristics:**
- ReLU after GCN + MLP projection
- Nonlinear spillover from neighbors
- Paper's main contribution

### Expected Performance Difference

| Model | Spillover Type | Expected R² | Why |
|-------|---------------|-------------|-----|
| HAR | None (baseline) | 0.65-0.70 | No graph information |
| GHAR | Linear | 0.68-0.72 | Linear spillover adds small benefit |
| GNNHAR1L | Nonlinear | 0.72-0.78 | Nonlinear interactions capture complex spillover |
| GNNHAR2L | 2-hop nonlinear | 0.73-0.79 | 2-hop neighbors add marginal benefit |
| GNNHAR3L | 3-hop nonlinear | 0.70-0.78 | May over-smooth, similar to 1L |

**Why GHAR exists:**
- Baseline for graph contribution
- If GHAR ≈ HAR, then graph provides no value
- If GHAR > HAR, then linear spillover exists
- If GNNHAR > GHAR, then nonlinear spillover matters

---

## References

1. **Original code:** https://github.com/chaozhang-ox/GNNHAR/blob/main/GNNHAR.py
2. **Paper:** Zhang et al. (2024) "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks", IJF
3. **Architecture doc:** `MULTI_STOCK_ARCHITECTURE.md`
4. **Failure analysis:** `SINGLE_STOCK_FAILURE_ANALYSIS.md`
