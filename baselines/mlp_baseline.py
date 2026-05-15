"""
MLP + Moirai2 baseline: same embeddings as GNN but no graph structure.

Architecture:
  node features (30, 384) → Linear(384→64) → ReLU → Dropout
                          → Linear(64→32)  → ReLU → Dropout
                          → Linear(32→16)  → ReLU
                          → Linear(16→1)

Key differences from GNN:
  - No SAGEConv / message passing — each stock predicted independently
  - Same Moirai2 embeddings, same walk-forward windows, same MSE loss
  - Shared weights across all 30 stocks (same as GNN)

This baseline answers: does the graph structure add value beyond
Moirai2 embeddings alone?  GNN > MLP → graph contribution confirmed.

Usage:
    python baselines/mlp_baseline.py
"""
import os
import sys
import warnings
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

warnings.filterwarnings("ignore", message="The mean prediction is not stored")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from src.embed_extractor import Moirai2Embedder
from gnn.build_graph import VN30_TICKERS, ALL_NODES
from gnn.train import walk_forward_dates, extract_embeddings


class VolatilityMLP(nn.Module):
    """
    Per-stock MLP on Moirai2 embeddings.  Identical depth/width to VolatilityGNN
    so any difference in results is attributable to the graph structure only.
    """
    def __init__(
        self,
        in_dim: int = 384,
        hidden: list[int] | None = None,
        mlp_hidden: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]
        if mlp_hidden is None:
            mlp_hidden = [16]

        self.drop = nn.Dropout(dropout)
        dims = [in_dim] + hidden
        layers: list[nn.Module] = []
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        mlp_dims = [hidden[-1]] + mlp_hidden + [1]
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (num_nodes, in_dim) → (num_nodes, 1)"""
        return self.net(x)


def train_mlp_walkforward(
    config: dict,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Walk-forward training for MLP+Moirai2 baseline.
    Mirrors gnn/train.py train_walkforward() exactly, minus the graph.

    Returns:
        DataFrame of per-window training metrics.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import random
    random.seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])

    prices_dir  = config["data"]["prices_dir"]
    train_end   = pd.Timestamp(config["data"]["train_end"])
    data_start  = pd.Timestamp(config["data"].get("data_start", "2014-06-27"))
    context_len = config["model"]["context_length"]
    stride      = config["model"]["stride"]
    horizon     = config["model"]["horizon"]

    close   = load_close_prices(prices_dir, tickers=VN30_TICKERS + ["VNINDEX"])
    close   = close[close.index >= data_start]
    log_ret = compute_log_returns(close)
    rv_all  = compute_rv(close[VN30_TICKERS], h=horizon)

    embedder = Moirai2Embedder(
        size="small",
        context_length=context_len,
        patch_size=config["model"].get("patch_size", 32),
    )

    model = VolatilityMLP(
        in_dim=Moirai2Embedder.D_MODEL,
        hidden=config["model"]["gnn_hidden"],
        mlp_hidden=config["model"]["mlp_hidden"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    train_dates = pd.DatetimeIndex(log_ret.index)
    window_ends = walk_forward_dates(train_dates, train_end, context_len, stride)
    epochs      = config["training"]["epochs"]
    patience    = config["training"]["early_stopping"]
    grad_clip   = config["training"].get("grad_clip", 1.0)

    window_metrics: list[dict] = []
    best_loss = float("inf")
    patience_counter = 0

    for w_idx, window_end in enumerate(window_ends):
        node_feats = extract_embeddings(embedder, log_ret, window_end, context_len)

        rv_at_date = rv_all.loc[window_end] if window_end in rv_all.index else None
        if rv_at_date is None or rv_at_date.isna().all():
            continue

        # Labels for VN30 stocks only (nodes 1-30)
        labels = np.zeros(31, dtype=np.float32)
        for i, ticker in enumerate(VN30_TICKERS):
            val = rv_at_date.get(ticker, np.nan)
            labels[i + 1] = float(val) if not pd.isna(val) else 0.0
        labels_t  = torch.tensor(labels, dtype=torch.float).to(device)
        loss_mask = torch.ones(31, dtype=torch.bool)
        loss_mask[0] = False   # exclude VNINDEX (no label)

        for _ in range(epochs):
            model.train()
            x = node_feats.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)            # (31,)
            pred_m   = pred[loss_mask]
            target_m = labels_t[loss_mask]
            loss = torch.mean((pred_m - target_m) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        window_metrics.append({
            "window": w_idx + 1,
            "window_end": window_end.date(),
            "train_loss": loss.item(),
        })

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_mlp.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if not (results_dir / "best_mlp.pt").exists():
        torch.save(model.state_dict(), results_dir / "best_mlp.pt")

    return pd.DataFrame(window_metrics)


@torch.no_grad()
def run_mlp_inference(
    config: dict,
    checkpoint_path: str,
    results_dir: str = "results",
) -> dict[str, pd.Series]:
    """
    Run MLP+Moirai2 inference on the 2026 test set.

    Returns:
        dict mapping ticker -> pd.Series of predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prices_dir  = config["data"]["prices_dir"]
    test_start  = pd.Timestamp(config["data"]["test_start"])
    data_start  = pd.Timestamp(config["data"].get("data_start", "2014-06-27"))
    context_len = config["model"]["context_length"]
    horizon     = config["model"]["horizon"]

    close   = load_close_prices(prices_dir, tickers=VN30_TICKERS + ["VNINDEX"])
    close   = close[close.index >= data_start]
    log_ret = compute_log_returns(close)
    rv_all  = compute_rv(close[VN30_TICKERS], h=horizon)

    embedder = Moirai2Embedder(
        size="small",
        context_length=context_len,
        patch_size=config["model"].get("patch_size", 32),
    )

    model = VolatilityMLP(
        in_dim=Moirai2Embedder.D_MODEL,
        hidden=config["model"]["gnn_hidden"],
        mlp_hidden=config["model"]["mlp_hidden"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_dates = rv_all.index[
        (rv_all.index >= test_start) & (~rv_all.isna().all(axis=1))
    ]

    preds_per_stock: dict[str, dict] = {t: {} for t in VN30_TICKERS}
    for date in test_dates:
        node_feats = extract_embeddings(embedder, log_ret, date, context_len).to(device)
        pred = model(node_feats).cpu().numpy().ravel()   # (31,)
        for i, ticker in enumerate(VN30_TICKERS):
            preds_per_stock[ticker][date] = max(float(pred[i + 1]), 0.0)

    return {t: pd.Series(preds_per_stock[t]) for t in VN30_TICKERS}


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Smoke test: few windows
    cfg["training"]["epochs"]        = 2
    cfg["training"]["early_stopping"] = 999
    cfg["model"]["context_length"]   = 200
    cfg["model"]["stride"]           = 500

    print("MLP+Moirai2 smoke test (2 epochs, few windows)...")
    metrics_df = train_mlp_walkforward(cfg, results_dir="results")
    print(f"Windows processed: {len(metrics_df)}")
    if len(metrics_df) > 0:
        print(metrics_df.tail(3).to_string(index=False))
    print("\nmlp_baseline.py smoke test OK.")
