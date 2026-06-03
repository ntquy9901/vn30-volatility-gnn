"""
sklearn models for train_multi_stock.py compatibility.

Provides wrapper classes that allow sklearn LinearRegression to be used
in the PyTorch-based training pipeline, enabling unified baseline comparison.

Key difference from PyTorch models:
    - sklearn uses closed-form OLS solution (no training loop/epochs)
    - sklearn uses 100% of training data (no validation split needed)
    - sklearn is deterministic (same result every time, no randomness)
"""

from sklearn.linear_model import LinearRegression
import numpy as np


class HAR_OLS:
    """
    sklearn HAR-OLS wrapper for train_multi_stock.py compatibility.

    Mimics PyTorch nn.Module interface while using sklearn LinearRegression internally.
    Uses 100% of training data (no validation split) for maximum data efficiency.

    Architecture:
        - Per-stock LinearRegression: 30 separate models (one per VN30 stock)
        - Features: [RV_d, RV_w, RV_m] HAR features
        - Target: Realized volatility (z-scored residuals)
        - Method: Ordinary Least Squares (closed-form solution)

    Training:
        - Closed-form OLS solution (matrix inversion, no gradient descent)
        - Instant fit (no epochs needed)
        - Uses all training data (no validation split)

    This wrapper allows sklearn to fit into the existing PyTorch training
    infrastructure while maintaining the sklearn advantage of 25% more
    training data (96,390 samples vs 77,112 for PyTorch).

    Expected performance: R² ≈ 0.7532 (upper bound with maximum data efficiency)
    """

    def __init__(self):
        """Initialize sklearn HAR-OLS model (no learnable parameters)."""
        self.stock_models = {}
        self.fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, stocks_train: np.ndarray):
        """
        Fit sklearn HAR-OLS model (per-stock LinearRegression).

        Args:
            X_train: (N_samples, 3) HAR features [RV_d, RV_w, RV_m]
            y_train: (N_samples,) RV targets
            stocks_train: (N_samples,) stock indices (0-29)

        Process:
            - Fit separate LinearRegression for each stock (30 models total)
            - Uses sklearn's closed-form OLS solution (instant)
            - No gradient descent, no epochs, no early stopping needed

        Time complexity: O(N) where N = number of training samples
        """
        print("    [HAR_OLS] Fitting per-stock LinearRegression models...")

        self.stock_models = {}
        for stock_id in np.unique(stocks_train):
            mask = (stocks_train == stock_id)
            X_stock = X_train[mask]
            y_stock = y_train[mask]

            # Fit sklearn LinearRegression (closed-form OLS)
            model = LinearRegression(fit_intercept=True, n_jobs=-1)
            model.fit(X_stock, y_stock)
            self.stock_models[stock_id] = model

        self.fitted = True
        print(f"    [HAR_OLS] Trained {len(self.stock_models)} stock-specific models")

    def predict(self, X: np.ndarray, stocks: np.ndarray) -> np.ndarray:
        """
        Predict using stock-specific sklearn models.

        Args:
            X: (N_samples, 3) HAR features
            stocks: (N_samples,) stock indices (0-29)

        Returns:
            (N_samples,) RV predictions

        Process:
            - For each stock, use that stock's trained LinearRegression
            - Concatenate predictions across all stocks
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = np.zeros(len(X))

        for stock_id, model in self.stock_models.items():
            mask = (stocks == stock_id)
            if mask.any():
                predictions[mask] = model.predict(X[mask])

        return predictions

    def forward(self, X: np.ndarray, stocks: np.ndarray) -> np.ndarray:
        """
        PyTorch-style forward pass for compatibility.

        This method mimics PyTorch's forward() interface so HAR_OLS can
        be used in existing PyTorch code without changes.

        Args:
            X: (N_samples, 3) HAR features
            stocks: (N_samples,) stock indices

        Returns:
            (N_samples,) RV predictions
        """
        return self.predict(X, stocks)

    def parameters(self):
        """
        Return dummy parameters for PyTorch compatibility.

        sklearn LinearRegression has no learnable parameters in the PyTorch
        sense (no .parameters() iterator, no gradients). This method returns
        an empty list to prevent errors in PyTorch code that expects parameters.

        Returns:
            [] (empty list, sklearn has no learnable parameters)
        """
        return []

    def train(self):
        """Dummy method for PyTorch compatibility (sklearn has no train mode)."""
        pass  # sklearn has no train/eval modes

    def eval(self):
        """Dummy method for PyTorch compatibility (sklearn has no train mode)."""
        pass  # sklearn is always in "eval" mode (closed-form solution)
