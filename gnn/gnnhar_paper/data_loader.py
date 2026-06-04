"""
Multi-Stock Data Loader for GNNHAR models.

Loads 30 VN30 stocks, builds HAR features, and flattens to (N_stocks × N_dates, 3)
format suitable for sklearn LinearRegression and PyTorch models.

Key difference from single-stock:
- Flattens across stocks AND dates (not just dates)
- Returns stock indices and date indices for reconstruction
- Supports batch-level processing for multi-stock models

Usage:
    loader = MultiStockDataLoader(tickers=VN30_TICKERS, horizon=5)
    X_train, y_train, stocks_train, dates_train = loader.get_train_data()
    X_test, y_test, stocks_test, dates_test = loader.get_test_data()
"""
import numpy as np
import pandas as pd
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_labels import load_close_prices, compute_rv


class MultiStockDataLoader:
    """
    Load and prepare multi-stock HAR dataset for GNNHAR models.

    Data format:
        Input: 30 stocks × ~2500 days OHLCV prices
        Output: Flattened (N_stocks × N_dates, 3) HAR features

    Example:
        30 stocks × 2000 valid dates = 60000 samples
        Each sample: [RV_d, RV_w, RV_m] for one stock on one date
    """

    def __init__(
        self,
        tickers: list[str],
        horizon: int = 5,
        train_end: str = "2025-12-31",
        test_start: str = "2026-01-01",
        prices_dir: str | Path = None,
    ):
        """
        Initialize multi-stock data loader.

        Args:
            tickers: List of stock tickers (e.g., VN30_TICKERS)
            horizon: RV horizon in days (default 5 = 1 week)
            train_end: Training data end date (global temporal split)
            test_start: Test data start date (global temporal split)
            prices_dir: Directory containing *_ohlcv.csv files
        """
        self.tickers = tickers
        self.horizon = horizon
        self.train_end = pd.Timestamp(train_end)
        self.test_start = pd.Timestamp(test_start)
        self.prices_dir = Path(prices_dir) if prices_dir else PROJECT_ROOT / "data/raw/prices"

        # Data containers
        self.close = None
        self.rv = None
        self.features_dict = {}
        self.targets_dict = {}

        # Flattened data
        self.X_train = None
        self.y_train = None
        self.stocks_train = None
        self.dates_train = None

        self.X_val = None
        self.y_val = None
        self.stocks_val = None
        self.dates_val = None

        self.X_test = None
        self.y_test = None
        self.stocks_test = None
        self.dates_test = None

    def load_data(self):
        """Load close prices and compute RV for all stocks."""
        print(f"[Data] Loading close prices for {len(self.tickers)} stocks...")
        self.close = load_close_prices(self.prices_dir, tickers=self.tickers)
        print(f"  Shape: {self.close.shape} | Range: {self.close.index[0].date()} to {self.close.index[-1].date()}")

        print(f"[Data] Computing RV (h={self.horizon})...")
        self.rv = compute_rv(self.close, h=self.horizon)
        print(f"  RV shape: {self.rv.shape}")

    def build_har_features(self, rv_series: pd.Series) -> pd.DataFrame:
        """
        Build HAR features [RV_d, RV_w, RV_m] for a single stock.

        Args:
            rv_series: RV series for one stock (indexed by date)

        Returns:
            DataFrame with columns [RV_d, RV_w, RV_m], same index as rv_series
        """
        rv_d = rv_series.shift(1)                                  # daily lag
        rv_w = rv_series.shift(1).rolling(5, min_periods=5).mean()   # weekly avg
        rv_m = rv_series.shift(1).rolling(22, min_periods=22).mean()  # monthly avg

        features = pd.DataFrame({
            "RV_d": rv_d,
            "RV_w": rv_w,
            "RV_m": rv_m,
        }, index=rv_series.index)
        return features

    def build_features(self):
        """Build HAR features and extract targets for all stocks."""
        print(f"[Features] Building HAR features for {len(self.tickers)} stocks...")

        for ticker in self.tickers:
            if ticker not in self.rv.columns:
                print(f"  WARNING: {ticker} not in RV data, skipping")
                continue

            # Build HAR features
            feats = self.build_har_features(self.rv[ticker])
            self.features_dict[ticker] = feats

            # Extract targets (RV values)
            self.targets_dict[ticker] = self.rv[ticker]

        print(f"  Built features for {len(self.features_dict)} stocks")

    def flatten_dataset(self):
        """
        Flatten multi-stock data to (N_samples, 3) format.

        For each stock-date pair with valid features and target:
            - Append features [RV_d, RV_w, RV_m] to X
            - Append target RV to y
            - Append stock index to stocks
            - Append date to dates

        Result:
            X: (N_stocks × N_dates, 3) HAR features
            y: (N_stocks × N_dates,) RV targets
            stocks: (N_stocks × N_dates,) stock indices (0-29)
            dates: (N_stocks × N_dates,) date timestamps
        """
        print("[Flatten] Flattening multi-stock dataset...")

        all_samples = []
        all_targets = []
        all_stocks = []
        all_dates = []

        stock_to_idx = {ticker: idx for idx, ticker in enumerate(self.tickers)}

        for ticker in self.tickers:
            if ticker not in self.features_dict:
                continue

            feats = self.features_dict[ticker]  # (N_dates, 3)
            targets = self.targets_dict[ticker]  # (N_dates,)

            # Align features and targets, drop NaN rows
            df = pd.concat([feats, targets.rename("target")], axis=1).dropna()

            if len(df) == 0:
                print(f"  WARNING: {ticker} has no valid samples")
                continue

            # Extract arrays
            X_stock = df[["RV_d", "RV_w", "RV_m"]].values  # (N_samples, 3)
            y_stock = df["target"].values                  # (N_samples,)
            dates_stock = df.index                         # (N_samples,)

            # Append to global lists
            all_samples.append(X_stock)
            all_targets.append(y_stock)
            all_stocks.append(np.full(len(X_stock), stock_to_idx[ticker]))
            all_dates.append(dates_stock)

        # Concatenate all stocks
        if len(all_samples) == 0:
            raise ValueError("No valid samples found")

        X_all = np.vstack(all_samples)           # (N_total, 3)
        y_all = np.concatenate(all_targets)      # (N_total,)
        stocks_all = np.concatenate(all_stocks)  # (N_total,)
        dates_all = pd.DatetimeIndex(np.concatenate(all_dates))  # Use DatetimeIndex

        print(f"  Total samples: {len(X_all)} = {len(self.tickers)} stocks × ~{len(X_all)//len(self.tickers)} dates each")

        # Store raw flattened data
        self.X_all = X_all
        self.y_all = y_all
        self.stocks_all = stocks_all
        self.dates_all = dates_all

    def split_train_val_test(self):
        """
        Split flattened dataset into train/val/test with temporal cutoff.

        Split strategy (global temporal split, same for all stocks):
            Train: dates <= train_end
            Val:   train_end < dates <= test_start (if any dates between)
            Test:  dates >= test_start

        For sklearn GHAR:
            Use train set for fitting
            Use test set for evaluation

        For PyTorch GNNHAR:
            Further split train into 80/20 train/val for early stopping
        """
        print(f"\n[Split] Global temporal split:")

        # Train split: dates <= train_end
        train_mask = self.dates_all <= self.train_end
        self.X_train = self.X_all[train_mask]
        self.y_train = self.y_all[train_mask]
        self.stocks_train = self.stocks_all[train_mask]
        self.dates_train = self.dates_all[train_mask]

        # Test split: dates >= test_start
        test_mask = self.dates_all >= self.test_start
        self.X_test = self.X_all[test_mask]
        self.y_test = self.y_all[test_mask]
        self.stocks_test = self.stocks_all[test_mask]
        self.dates_test = self.dates_all[test_mask]

        # Validation split: dates between train_end and test_start
        val_mask = (self.dates_all > self.train_end) & (self.dates_all < self.test_start)
        self.X_val = self.X_all[val_mask]
        self.y_val = self.y_all[val_mask]
        self.stocks_val = self.stocks_all[val_mask]
        self.dates_val = self.dates_all[val_mask]

        # Print split statistics
        print(f"  Train: {len(self.X_train)} samples ({self.dates_train[0].date()} to {self.dates_train[-1].date()})")
        print(f"  Val:   {len(self.X_val)} samples" + (f" ({self.dates_val[0].date()} to {self.dates_val[-1].date()})" if len(self.X_val) > 0 else ""))
        print(f"  Test:  {len(self.X_test)} samples ({self.dates_test[0].date()} to {self.dates_test[-1].date()})")

        # Distribution analysis
        train_mean = self.y_train.mean()
        test_mean = self.y_test.mean()
        shift_pct = (test_mean - train_mean) / train_mean * 100 if train_mean > 0 else 0

        print(f"\n[Distribution] Target RV analysis:")
        print(f"  Train mean: {train_mean:.6f}")
        print(f"  Test mean:  {test_mean:.6f}")
        print(f"  Shift:      {shift_pct:+.1f}%")

        # Effective Sample Size (ESS) for training
        # ESS = N_raw / max_horizon (Lopez de Prado 2018)
        # For multi-stock: ESS = (N_stocks × N_dates) / horizon
        n_stocks_train = len(np.unique(self.stocks_train))
        n_dates_per_stock = len(self.X_train) // n_stocks_train if n_stocks_train > 0 else 0
        ess = len(self.X_train) / self.horizon

        print(f"\n[ESS] Effective Sample Size:")
        print(f"  Raw samples: {len(self.X_train)}")
        print(f"  ESS (N/horizon): {ess:.0f}")
        print(f"  Stocks in train: {n_stocks_train}")
        print(f"  Avg dates per stock: {n_dates_per_stock}")

    def prepare_sklearn_data(self):
        """
        Prepare data for sklearn models (HAR OLS, GHAR-sklearn).

        Returns:
            X_train, y_train, stocks_train, dates_train
            X_test, y_test, stocks_test, dates_test
        """
        return (
            self.X_train, self.y_train, self.stocks_train, self.dates_train,
            self.X_test, self.y_test, self.stocks_test, self.dates_test
        )

    def prepare_pytorch_data(self, val_split=0.2):
        """
        Prepare data for PyTorch models with train/val split.

        Further splits training data into train/val for early stopping.

        Args:
            val_split: Fraction of training data for validation (default 0.2)

        Returns:
            X_train, y_train, stocks_train, dates_train
            X_val, y_val, stocks_val, dates_val
            X_test, y_test, stocks_test, dates_test
        """
        # Split train into train/val (temporal: last 20% of train data)
        if len(self.X_train) > 0:
            split_idx = int(len(self.X_train) * (1 - val_split))

            X_train_final = self.X_train[:split_idx]
            y_train_final = self.y_train[:split_idx]
            stocks_train_final = self.stocks_train[:split_idx]
            dates_train_final = self.dates_train[:split_idx]

            X_val_final = self.X_train[split_idx:]
            y_val_final = self.y_train[split_idx:]
            stocks_val_final = self.stocks_train[split_idx:]
            dates_val_final = self.dates_train[split_idx:]
        else:
            # No validation data available
            X_train_final = self.X_train
            y_train_final = self.y_train
            stocks_train_final = self.stocks_train
            dates_train_final = self.dates_train

            X_val_final = np.array([]).reshape(0, 3)
            y_val_final = np.array([])
            stocks_val_final = np.array([], dtype=int)
            dates_val_final = pd.DatetimeIndex([])

        return (
            X_train_final, y_train_final, stocks_train_final, dates_train_final,
            X_val_final, y_val_final, stocks_val_final, dates_val_final,
            self.X_test, self.y_test, self.stocks_test, self.dates_test
        )

    def get_summary(self):
        """Return summary statistics of loaded data."""
        return {
            "n_tickers": len(self.tickers),
            "horizon": self.horizon,
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "n_train": len(self.X_train) if self.X_train is not None else 0,
            "n_val": len(self.X_val) if self.X_val is not None else 0,
            "n_test": len(self.X_test) if self.X_test is not None else 0,
            "train_mean": float(self.y_train.mean()) if self.y_train is not None and len(self.y_train) > 0 else 0,
            "test_mean": float(self.y_test.mean()) if self.y_test is not None and len(self.y_test) > 0 else 0,
        }


# ── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from gnn.build_graph import VN30_TICKERS

    print("="*70)
    print("Multi-Stock Data Loader Test")
    print("="*70)

    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=5,
        train_end="2025-12-31",
        test_start="2026-01-01",
    )

    # Load and prepare data
    loader.load_data()
    loader.build_features()
    loader.flatten_dataset()
    loader.split_train_val_test()

    # Test sklearn data preparation
    print("\n[sklearn] Preparing data for sklearn models...")
    X_train, y_train, stocks_train, dates_train, X_test, y_test, stocks_test, dates_test = loader.prepare_sklearn_data()
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    # Test PyTorch data preparation
    print("\n[PyTorch] Preparing data for PyTorch models...")
    (X_train_pt, y_train_pt, stocks_train_pt, dates_train_pt,
     X_val_pt, y_val_pt, stocks_val_pt, dates_val_pt,
     X_test_pt, y_test_pt, stocks_test_pt, dates_test_pt) = loader.prepare_pytorch_data(val_split=0.2)
    print(f"  Train: X={X_train_pt.shape}, y={y_train_pt.shape}")
    print(f"  Val:   X={X_val_pt.shape}, y={y_val_pt.shape}")
    print(f"  Test:  X={X_test_pt.shape}, y={y_test_pt.shape}")

    # Print summary
    print("\n[Summary]")
    summary = loader.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\ndata_loader.py OK.")
