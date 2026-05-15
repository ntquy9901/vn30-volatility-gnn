"""
Walk-forward training loop for VolatilityGNN.

Pipeline per window (FR20-FR23, FR30-FR32):
  1. Build graph snapshot at window end date
  2. Extract Moirai2 embeddings for each stock
  3. Train GNN head for `epochs` steps
  4. Evaluate on held-out test windows

Usage:
    python gnn/train.py
"""
import os
import sys
import warnings
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", message="The mean prediction is not stored")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_labels import (
    load_close_prices, compute_log_returns, compute_rv
)
from src.embed_extractor import Moirai2Embedder
from gnn.build_graph import build_graph, VN30_TICKERS, ALL_NODES, NODE_IDX
from gnn.model import VolatilityGNN


# ── Logging ─────────────────────────────────────────────────────────────────
def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("gnn_train")


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Walk-forward window generator ────────────────────────────────────────────
def walk_forward_dates(
    dates: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    context_length: int,
    stride: int,
) -> list[pd.Timestamp]:
    """
    Return list of window_end dates for training windows.
    Each window uses the last `context_length` days ending at window_end.
    Windows advance by `stride` trading days.
    """
    train_dates = dates[dates <= train_end]
    if len(train_dates) < context_length:
        return []
    # First valid window ends at index context_length-1
    window_ends = []
    idx = context_length - 1
    while idx < len(train_dates):
        window_ends.append(train_dates[idx])
        idx += stride
    return window_ends


# ── Moirai2 embedding extraction ─────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(
    embedder: Moirai2Embedder,
    log_ret: pd.DataFrame,
    window_end: pd.Timestamp,
    context_length: int,
) -> torch.Tensor:
    """
    Extract Moirai2 embeddings for all 31 nodes at a given window end date.

    Returns:
        Tensor (31, 384) — node_0 = VNINDEX, nodes 1-30 = VN30 stocks.
    """
    # All tickers in node order
    node_features = np.zeros((31, Moirai2Embedder.D_MODEL), dtype=np.float32)

    for node_idx, ticker in enumerate(ALL_NODES):
        if ticker not in log_ret.columns:
            continue
        # Get last context_length returns ending at window_end
        col = log_ret[ticker]
        col_valid = col[col.index <= window_end].dropna()
        if len(col_valid) < context_length:
            continue
        series_window = col_valid.iloc[-context_length:]
        # Re-index to business day frequency for Moirai2
        series_window = series_window.asfreq("B", method="ffill")

        result = embedder.embed_series(
            {ticker: series_window}, prediction_length=1
        )
        node_features[node_idx] = result[ticker]

    return torch.tensor(node_features, dtype=torch.float)


# ── Training ─────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: VolatilityGNN,
    optimizer: optim.Optimizer,
    node_features: torch.Tensor,
    graph_data,
    rv_labels: torch.Tensor,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    node_features = node_features.to(device)
    g             = graph_data.g.to(device)
    loss_mask     = graph_data.loss_mask.to(device)
    rv_labels     = rv_labels.to(device)

    optimizer.zero_grad()
    pred = model(g, node_features)
    loss = model.masked_loss(pred, rv_labels.unsqueeze(-1), loss_mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


# ── Main training loop ───────────────────────────────────────────────────────
def train_walkforward(config: dict, results_dir: str = "results") -> pd.DataFrame:
    """
    Walk-forward training with Moirai2 embeddings + GNN.

    Returns:
        DataFrame of per-window training metrics.
    """
    results_dir = Path(results_dir)
    logger = setup_logging(results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(config["training"]["seed"])

    # ── Load data ────────────────────────────────────────────────────────────
    prices_dir   = config["data"]["prices_dir"]
    train_end    = pd.Timestamp(config["data"]["train_end"])
    data_start   = pd.Timestamp(config["data"].get("data_start", "2014-06-27"))
    context_len  = config["model"]["context_length"]
    stride       = config["model"]["stride"]
    horizon      = config["model"]["horizon"]
    corr_window  = config["model"]["corr_window"]
    corr_thresh  = config["model"]["corr_threshold"]
    use_log_rv   = config["model"].get("use_log_rv", False)

    logger.info("Loading prices...")
    close = load_close_prices(prices_dir, tickers=VN30_TICKERS + ["VNINDEX"])
    close = close[close.index >= data_start]
    log_ret = compute_log_returns(close)

    logger.info("Computing RV labels...")
    rv_all = compute_rv(close[VN30_TICKERS], h=horizon)

    # ── Moirai2 embedder ─────────────────────────────────────────────────────
    logger.info("Loading Moirai2-small...")
    embedder = Moirai2Embedder(
        size="small",
        context_length=context_len,
        patch_size=config["model"].get("patch_size", 32),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = VolatilityGNN(
        in_dim=Moirai2Embedder.D_MODEL,
        hidden=config["model"]["gnn_hidden"],
        mlp_hidden=config["model"]["mlp_hidden"],
        dropout=config["model"]["dropout"],
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    # ── Walk-forward windows ──────────────────────────────────────────────────
    train_dates  = pd.DatetimeIndex(log_ret.index)
    window_ends  = walk_forward_dates(train_dates, train_end, context_len, stride)
    logger.info(f"Walk-forward windows: {len(window_ends)}")

    epochs      = config["training"]["epochs"]
    patience    = config["training"]["early_stopping"]
    grad_clip   = config["training"].get("grad_clip", 1.0)

    window_metrics: list[dict] = []
    best_loss = float("inf")
    patience_counter = 0

    for w_idx, window_end in enumerate(window_ends):
        logger.info(f"Window {w_idx+1}/{len(window_ends)}: end={window_end.date()}")

        # Build graph snapshot
        graph = build_graph(log_ret, end_date=window_end,
                            corr_window=corr_window,
                            corr_threshold=corr_thresh)

        # Extract embeddings (frozen Moirai2 — no gradient)
        node_feats = extract_embeddings(embedder, log_ret, window_end, context_len)

        # RV label at window_end for each VN30 stock
        rv_at_date = rv_all.loc[window_end] if window_end in rv_all.index else None
        if rv_at_date is None or rv_at_date.isna().all():
            logger.info(f"  Skipping window — no RV label at {window_end.date()}")
            continue

        # Build label tensor (31,): node_0=VNINDEX gets 0 (masked out)
        labels = np.zeros(31, dtype=np.float32)
        for i, ticker in enumerate(VN30_TICKERS):
            val = rv_at_date.get(ticker, np.nan)
            labels[i + 1] = float(val) if not pd.isna(val) else 0.0
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        if use_log_rv:
            labels_tensor = torch.log(labels_tensor.clamp(min=1e-8))

        # Train for `epochs` iterations on this window
        for epoch in range(epochs):
            loss = train_one_epoch(
                model, optimizer, node_feats, graph,
                labels_tensor, grad_clip, device
            )

        window_metrics.append({
            "window": w_idx + 1,
            "window_end": window_end.date(),
            "train_loss": loss,
        })

        # Early stopping based on recent window loss
        if loss < best_loss - 1e-6:
            best_loss = loss
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_gnn.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at window {w_idx+1}")
                break

    # Save checkpoint if not yet saved
    if not (results_dir / "best_gnn.pt").exists():
        torch.save(model.state_dict(), results_dir / "best_gnn.pt")

    metrics_df = pd.DataFrame(window_metrics)
    metrics_df.to_csv(results_dir / "train_window_metrics.csv", index=False)
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")
    return metrics_df


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Smoke test: use only first 3 windows to verify pipeline runs
    cfg["training"]["epochs"] = 2
    cfg["training"]["early_stopping"] = 999
    cfg["model"]["context_length"] = 200
    cfg["model"]["stride"] = 500   # large stride → few windows

    print("Running smoke test (2 epochs, few windows)...")
    df = train_walkforward(cfg, results_dir="results")
    print(f"\nSmoke test complete. Windows processed: {len(df)}")
    if len(df) > 0:
        print(df.tail(3).to_string(index=False))
    print("\ntrain.py smoke test OK.")
