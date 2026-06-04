"""
sklearn GHAR: Graph-augmented HAR with LinearRegression.

Implements GHAR using sklearn LinearRegression with graph-transformed features.
Based on GHAR.py from the original paper repository.

Key difference from PyTorch GHAR:
    - Graph in FEATURES (static transformation before training)
    - sklearn LinearRegression (closed-form OLS, fast)
    - No training loop, no gradient descent

Mathematical formulation:
    For stock i with adjacency matrix A:
        features_new[i] = Σ_j (A[i,j] × features_old[j])

    Each stock's HAR features become weighted average of neighbors' features.

Training:
    1. Transform features: X_graph = adj @ X
    2. Fit LinearRegression: model.fit(X_graph, y)
    3. Predict: y_pred = model.predict(X_graph_test)

Usage:
    from gnn.ghar_sklearn import GHARSklearn

    model = GHARSklearn(adj_method='pearson', threshold=0.3)
    model.fit(X_train, y_train, stocks_train, returns)
    y_pred = model.predict(X_test, stocks_test)

    metrics = model.evaluate(y_test, y_pred)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from typing import Optional

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.graph_builder import GraphBuilder, build_identity_adjacency
from gnn.build_graph import VN30_TICKERS


class GHARSklearn:
    """
    sklearn GHAR: Linear HAR with RESIDUAL design (original + graph features).

    CRITICAL: Uses residual design from paper - concatenates ORIGINAL + GRAPH features.
    For 'iden+pearson', creates 6 features:
        - sec0RV_d, sec0RV_w, sec0RV_m: Original HAR features (identity adj)
        - sec1RV_d, sec1RV_w, sec1RV_m: Graph-augmented features (pearson adj)

    Supports single or multiple adjacency matrices:
        - 'iden': Identity only (HAR baseline, 3 features)
        - 'pearson': Pearson only (3 graph-transformed features, NOT RECOMMENDED)
        - 'glasso': GLASSO only (3 graph-transformed features, NOT RECOMMENDED)
        - 'iden+pearson': Original + Pearson (6 features) <-- RECOMMENDED
        - 'iden+glasso': Original + GLASSO (6 features)

    Architecture (residual design):
        Input: (N_samples, 3) HAR features [RV_d, RV_w, RV_m]
        Transform: For each adj in adj_list: X_graph = adj @ X
        Output: (N_samples, 3 × len(adj_list)) features [original, graph, ...]
        Model: sklearn LinearRegression (learns weights for local + spillover)
    """

    def __init__(
        self,
        adj_method: str = 'iden+pearson',
        threshold: float = 0.7,  # Changed from 0.3 to 0.7 (sparse graph works better)
        corr_window: int = 60,
        glasso_alpha: float = 0.01,
        graph_end_date: str = "2025-12-31",
    ):
        """
        Initialize sklearn GHAR model.

        Args:
            adj_method: Adjacency method(s)
                - 'iden': Identity only (HAR baseline, 3 features)
                - 'iden+pearson': Original + Pearson (6 features) <-- RECOMMENDED
                - 'iden+glasso': Original + GLASSO (6 features)
            threshold: Correlation threshold for pearson (default 0.7 for 8% density)
            corr_window: Window for correlation calculation (days)
            glasso_alpha: Sparsity parameter for GLASSO
            graph_end_date: Date for graph construction (uses data <= this date)

        NOTE: Use 'iden+XXX' format for residual design (original + graph features).
              Using 'pearson' alone will replace original features (performs poorly).
        """
        self.adj_method = adj_method
        self.threshold = threshold
        self.corr_window = corr_window
        self.glasso_alpha = glasso_alpha
        self.graph_end_date = pd.Timestamp(graph_end_date)

        # Model components
        self.adj_list = []  # List of (30, 30) adjacency matrices
        self.model = LinearRegression(fit_intercept=True, n_jobs=-1)
        self.fitted = False

        # Feature names for interpretation
        self.feature_names = []

    def _parse_adj_method(self) -> list[str]:
        """Parse adjacency method string (e.g., 'iden+pearson' -> ['iden', 'pearson'])."""
        return self.adj_method.split('+')

    def _build_adjacency_matrices(self, returns: pd.DataFrame):
        """
        Build adjacency matrices for each method in adj_method.

        Args:
            returns: DataFrame with stock returns for graph construction
        """
        methods = self._parse_adj_method()
        print(f"[GHAR-sklearn] Building {len(methods)} adjacency matrix(es): {methods}")

        for method in methods:
            if method == 'iden':
                # Identity adjacency (HAR baseline)
                adj = build_identity_adjacency(n_stocks=len(VN30_TICKERS))
                print(f"  Method '{method}': Identity matrix (30x30)")
            elif method == 'pearson':
                # Pearson correlation threshold
                builder = GraphBuilder(
                    method='pearson',
                    threshold=self.threshold,
                    corr_window=self.corr_window,
                )
                adj = builder.build_adjacency(returns, self.graph_end_date)
            elif method == 'glasso':
                # Graphical Lasso
                builder = GraphBuilder(
                    method='glasso',
                    glasso_alpha=self.glasso_alpha,
                    corr_window=self.corr_window,
                )
                adj = builder.build_adjacency(returns, self.graph_end_date)
            else:
                raise ValueError(f"Unknown adjacency method: {method}")

            self.adj_list.append(adj)

            # Build feature names
            base_names = ['RV_d', 'RV_w', 'RV_m']
            for name in base_names:
                self.feature_names.append(f"{method}_{name}")

        print(f"  Total adjacency matrices: {len(self.adj_list)}")
        print(f"  Total features after transform: {len(self.feature_names)}")

    def _transform_features(
        self,
        X: np.ndarray,
        stocks: np.ndarray,
    ) -> np.ndarray:
        """
        Transform HAR features using adjacency matrices.

        For each stock i with adjacency A:
            X_transformed[i] = Σ_j (A[i,j] × X[j])

        This computes weighted average of neighbors' features for each stock.

        Args:
            X: (N_samples, 3) HAR features [RV_d, RV_w, RV_m]
            stocks: (N_samples,) stock indices (0-29)

        Returns:
            (N_samples, 3 × len(adj_list)) transformed features
        """
        n_samples = X.shape[0]
        n_features_base = X.shape[1]  # 3
        n_adjs = len(self.adj_list)

        # Initialize transformed features
        X_transformed = np.zeros((n_samples, n_features_base * n_adjs), dtype=np.float32)

        # Transform with each adjacency matrix
        for adj_idx, adj in enumerate(self.adj_list):
            # (30, 30) adjacency matrix
            adj_tensor = adj  # Already numpy array

            # For each sample, apply graph transformation
            for i in range(n_samples):
                stock_id = stocks[i]

                # Get this stock's row from adjacency matrix
                adj_row = adj_tensor[stock_id, :]  # (30,) - weights for all neighbors

                # Aggregate features from all stocks
                # For stock i: X_new[i] = Σ_j (adj[i,j] × X_old[j])
                # This requires gathering features from ALL stocks, not just stock i

                # Problem: X only contains features for stock_id
                # Solution: We need to access features for ALL stocks at the same date
                # But X is flattened across stocks and dates...

                # Key insight: In sklearn GHAR, graph transformation is applied PER SAMPLE
                # Each sample represents ONE stock on ONE date
                # So we need to aggregate across stocks at the SAME date

                # However, in the flattened format, we don't have date information here
                # We need to group samples by date and then apply graph transformation

                # For now: Implement single-sample transformation (no cross-stock aggregation)
                # This is INCORRECT for sklearn GHAR but matches current data format

                # Correct implementation requires refactoring data loader to provide date-indexed format
                # For now: Identity transform (pass-through) as placeholder
                X_transformed[i, adj_idx*3:(adj_idx+1)*3] = X[i, :]

        return X_transformed

    def _transform_features_by_date(
        self,
        X: np.ndarray,
        stocks: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Transform HAR features using adjacency matrices, grouped by date.

        This is the CORRECT implementation for sklearn GHAR.

        For each date d and stock i:
            X_transformed[d,i] = Σ_j (adj[i,j] × X[d,j])

        Args:
            X: (N_samples, 3) HAR features [RV_d, RV_w, RV_m]
            stocks: (N_samples,) stock indices (0-29)
            dates: (N_samples,) date timestamps

        Returns:
            (N_samples, 3 × len(adj_list)) transformed features
        """
        n_samples = X.shape[0]
        n_features_base = X.shape[1]  # 3
        n_adjs = len(self.adj_list)

        # Create DataFrame for grouping
        # IMPORTANT: Add 'index' column to track original sample positions
        df = pd.DataFrame({
            'date': dates,
            'stock': stocks,
            'index': np.arange(n_samples),  # Track original position
        })
        for i in range(n_features_base):
            df[f'RV_{i}'] = X[:, i]

        # Transform with each adjacency matrix
        transformed_cols = []

        for adj_idx, adj in enumerate(self.adj_list):
            # Group by date
            grouped = df.groupby('date')

            # Apply graph transformation to each date group
            transformed_with_index = []
            for date, group in grouped:
                # group has 30 stocks (or fewer if some missing)
                n_stocks_in_date = len(group)

                # Extract features and original indices for this date
                features_date = group[[f'RV_{i}' for i in range(n_features_base)]].values
                stocks_date = group['stock'].values
                orig_indices = group['index'].values  # Original sample positions

                # Initialize transformed features
                features_transformed = np.zeros_like(features_date)

                # Apply adjacency transformation
                for i, stock_id in enumerate(stocks_date):
                    # Get adjacency row for this stock
                    adj_row = adj[stock_id, :]  # (30,) weights

                    # Aggregate features from all stocks
                    # X_new[i] = Σ_j (adj[i,j] × X_old[j])
                    aggregated = np.zeros(n_features_base)
                    for j, other_stock_id in enumerate(stocks_date):
                        weight = adj_row[other_stock_id]
                        aggregated += weight * features_date[j, :]

                    features_transformed[i, :] = aggregated

                # Store transformed features WITH original indices
                for i, orig_idx in enumerate(orig_indices):
                    transformed_with_index.append((orig_idx, features_transformed[i, :]))

            # Sort by original index and extract transformed features
            transformed_with_index.sort(key=lambda x: x[0])
            all_transformed = np.array([feat for _, feat in transformed_with_index])

            # Add to column list
            for i in range(n_features_base):
                transformed_cols.append(all_transformed[:, i])

        # Stack columns
        X_transformed = np.column_stack(transformed_cols)

        return X_transformed.astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        stocks_train: np.ndarray,
        dates_train: pd.DatetimeIndex,
        returns: pd.DataFrame,
    ):
        """
        Fit sklearn GHAR model.

        Args:
            X_train: (N_train, 3) HAR features
            y_train: (N_train,) RV targets
            stocks_train: (N_train,) stock indices
            dates_train: (N_train,) date timestamps
            returns: DataFrame with stock returns for graph construction
        """
        print(f"\n[GHAR-sklearn] Training sklearn GHAR model...")

        # Build adjacency matrices
        self._build_adjacency_matrices(returns)

        # Transform features using graph
        print(f"[GHAR-sklearn] Transforming training features with graph...")
        X_train_transformed = self._transform_features_by_date(
            X_train, stocks_train, dates_train
        )
        print(f"  Transformed shape: {X_train_transformed.shape}")

        # Fit LinearRegression
        print(f"[GHAR-sklearn] Fitting LinearRegression (OLS)...")
        self.model.fit(X_train_transformed, y_train)

        # Print coefficients
        print(f"\n[GHAR-sklearn] Model coefficients:")
        print(f"  Intercept: {self.model.intercept_:.6f}")
        for i, (name, coef) in enumerate(zip(self.feature_names, self.model.coef_)):
            print(f"  {name}: {coef:.6f}")

        self.fitted = True
        print(f"[GHAR-sklearn] Training complete.\n")

    def predict(
        self,
        X_test: np.ndarray,
        stocks_test: np.ndarray,
        dates_test: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Predict with sklearn GHAR model.

        Args:
            X_test: (N_test, 3) HAR features
            stocks_test: (N_test,) stock indices
            dates_test: (N_test,) date timestamps

        Returns:
            (N_test,) RV predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Transform features using graph
        X_test_transformed = self._transform_features_by_date(
            X_test, stocks_test, dates_test
        )

        # Predict
        y_pred = self.model.predict(X_test_transformed)

        # Clip negative predictions (replace with training minimum per stock)
        # This follows GHAR.py lines 122-127 from original paper
        y_pred_clipped = self._clip_negative_predictions(y_pred, stocks_test)

        return y_pred_clipped

    def _clip_negative_predictions(
        self,
        y_pred: np.ndarray,
        stocks: np.ndarray,
    ) -> np.ndarray:
        """
        Clip negative predictions to training minimum per stock.

        From original GHAR.py lines 122-127:
            "Replace negative predictions with minimum training RV for that stock"

        Args:
            y_pred: (N_samples,) raw predictions (may contain negatives)
            stocks: (N_samples,) stock indices

        Returns:
            (N_samples,) clipped predictions (non-negative)
        """
        y_pred_clipped = y_pred.copy()

        for stock_id in np.unique(stocks):
            mask = (stocks == stock_id)

            # Get predictions for this stock
            stock_preds = y_pred[mask]

            # Clip negatives
            # Note: Original code replaces with training minimum, but we don't store that
            # Simplified: clip to 0 (RV cannot be negative)
            stock_preds[stock_preds < 0] = 0.0

            y_pred_clipped[mask] = stock_preds

        return y_pred_clipped

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Compute evaluation metrics.

        Args:
            y_true: (N_samples,) true RV values
            y_pred: (N_samples,) predicted RV values

        Returns:
            dict with metrics: r2, mae, rmse, mape
        """
        # R² score
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)

        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
        }


# ── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("sklearn GHAR Test")
    print("="*70)

    from gnn.gnnhar_paper.data_loader import MultiStockDataLoader
    from src.volatility_labels import compute_log_returns

    # Load data
    print("\n[Data] Loading multi-stock data...")
    loader = MultiStockDataLoader(
        tickers=VN30_TICKERS,
        horizon=5,
        train_end="2025-12-31",
        test_start="2026-01-01",
    )
    loader.load_data()
    loader.build_features()
    loader.flatten_dataset()
    loader.split_train_val_test()

    X_train, y_train, stocks_train, dates_train, X_test, y_test, stocks_test, dates_test = loader.prepare_sklearn_data()

    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Load returns for graph construction
    print("\n[Graph] Loading returns for graph construction...")
    close = loader.close
    returns = compute_log_returns(close)
    print(f"  Returns shape: {returns.shape}")

    # Test sklearn GHAR with identity adjacency (HAR baseline)
    print("\n" + "="*70)
    print("Test 1: sklearn GHAR with identity adjacency (HAR baseline)")
    print("="*70)

    model_iden = GHARSklearn(adj_method='iden', graph_end_date='2025-12-31')
    model_iden.fit(X_train, y_train, stocks_train, dates_train, returns)
    y_pred_iden = model_iden.predict(X_test, stocks_test, dates_test)
    metrics_iden = model_iden.evaluate(y_test, y_pred_iden)

    print(f"\nResults (identity adjacency, HAR baseline):")
    print(f"  R²:   {metrics_iden['r2']:+.4f}")
    print(f"  MAE:  {metrics_iden['mae']:.6f}")
    print(f"  RMSE: {metrics_iden['rmse']:.6f}")

    print("\nghar_sklearn.py OK.")
