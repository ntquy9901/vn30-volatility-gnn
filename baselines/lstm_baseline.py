"""
LSTM baseline: per-stock 1-layer LSTM trained on rolling RV windows.

Architecture:
  Input  : sequence of past `seq_len` RV values
  LSTM   : (seq_len, 1) → hidden_dim=32
  Linear : hidden_dim → 1

Trained walk-forward (same regime as GNN): fit on data up to train_end,
predict on 2026 test period.

Usage:
    python baselines/lstm_baseline.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_rv
from gnn.build_graph import VN30_TICKERS


class LSTMModel(nn.Module):
    def __init__(self, seq_len: int = 20, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 1)"""
        out, _ = self.lstm(x)
        last    = out[:, -1, :]          # (batch, hidden)
        return self.head(self.drop(last)) # (batch, 1)


def make_sequences(rv: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window → (X, y) arrays."""
    X, y = [], []
    for i in range(len(rv) - seq_len):
        X.append(rv[i : i + seq_len])
        y.append(rv[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_lstm(
    rv: pd.Series,
    train_end: pd.Timestamp,
    seq_len: int = 20,
    hidden: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> LSTMModel:
    """Train LSTM on training data, return fitted model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_rv = rv[rv.index <= train_end].dropna().values

    # Normalize by training std
    rv_std = train_rv.std() + 1e-8
    train_rv_norm = train_rv / rv_std

    X, y = make_sequences(train_rv_norm, seq_len)
    if len(X) < batch_size:
        batch_size = max(1, len(X))

    X_t = torch.tensor(X).unsqueeze(-1)   # (N, seq_len, 1)
    y_t = torch.tensor(y).unsqueeze(-1)   # (N, 1)

    model     = LSTMModel(seq_len=seq_len, hidden=hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    n = len(X_t)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Store normalization constant on the model
    model._rv_std = rv_std
    return model


def predict_lstm(
    model: LSTMModel,
    rv: pd.Series,
    test_start: pd.Timestamp,
    seq_len: int = 20,
) -> pd.Series:
    """
    Out-of-sample predictions: for each test date, use the previous seq_len RV values.
    """
    rv_std = model._rv_std
    model.eval()

    test_dates = rv.index[rv.index >= test_start]
    preds = {}

    with torch.no_grad():
        for t in test_dates:
            # history strictly before t
            hist = rv[rv.index < t].dropna()
            if len(hist) < seq_len:
                continue
            seq = hist.iloc[-seq_len:].values / rv_std
            x   = torch.tensor(seq, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
            pred = float(model(x).squeeze()) * rv_std
            pred = max(pred, 0.0)
            preds[t] = pred

    return pd.Series(preds, name=rv.name)


def run_lstm_baseline(
    prices_dir: str,
    train_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    horizon: int = 20,
    seq_len: int = 20,
    hidden: int = 32,
    epochs: int = 50,
    tickers: list[str] | None = None,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """
    Run LSTM baseline for all VN30 stocks.

    Returns:
        dict mapping ticker -> pd.Series of predictions for test period.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    train_end_ts  = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    close  = load_close_prices(prices_dir, tickers=tickers)
    rv_all = compute_rv(close, h=horizon)

    results = {}
    for ticker in tickers:
        if ticker not in rv_all.columns:
            continue
        try:
            model = fit_lstm(
                rv_all[ticker], train_end_ts,
                seq_len=seq_len, hidden=hidden, epochs=epochs, seed=seed,
            )
            preds = predict_lstm(model, rv_all[ticker], test_start_ts, seq_len=seq_len)
            results[ticker] = preds
        except Exception as e:
            print(f"  LSTM {ticker} FAILED: {e}")

    return results


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    prices_dir = cfg["data"]["prices_dir"]
    train_end  = cfg["data"]["train_end"]
    test_start = cfg["data"]["test_start"]
    horizon    = cfg["model"]["horizon"]

    print("LSTM baseline smoke test (3 stocks, 10 epochs)")
    results = run_lstm_baseline(
        prices_dir, train_end=train_end, test_start=test_start,
        horizon=horizon, epochs=10, tickers=["VCB", "HPG", "FPT"],
    )

    print(f"\nResults:")
    for t, s in results.items():
        print(f"  {t}: {len(s)} test predictions, mean={s.mean():.4f}, std={s.std():.4f}")
    print("\nlstm_baseline.py smoke test OK.")
