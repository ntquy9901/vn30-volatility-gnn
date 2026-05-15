"""
Build the VN30 + VNINDEX stock graph for each monthly snapshot.

Graph layout (FR12-FR17):
  node_0           = VNINDEX (virtual hub)
  node_1 .. node_30 = VN30 stocks (in VN30_TICKERS order)

Edges:
  Stock-stock : Pearson(corr_window days) > corr_threshold  OR  same_sector
  Hub edges   : VNINDEX -> stock_i  for all i in [1..30]

Edge attributes: [pearson_corr, same_sector_flag]  shape (E, 2)

Usage:
    python gnn/build_graph.py
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ── VN30 sector map (hardcoded domain knowledge) ────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "VCB": "bank",  "BID": "bank",  "CTG": "bank",  "MBB": "bank",
    "TCB": "bank",  "HDB": "bank",  "VPB": "bank",  "STB": "bank",
    "ACB": "bank",  "SSB": "bank",  "SHB": "bank",  "TPB": "bank",
    "VIB": "bank",
    "VHM": "realestate",  "NVL": "realestate",  "PDR": "realestate",
    "BCM": "realestate",
    "HPG": "steel",  "HSG": "steel",  "NKG": "steel",
    "GAS": "energy",  "PLX": "energy",  "POW": "energy",
    "FPT": "tech",
    "MWG": "retail",  "MSN": "retail",  "SAB": "retail",
    "VNM": "fmcg",
    "BVH": "insurance",
    "GVR": "rubber",
    "VJC": "aviation",
    "VIC": "conglomerate",
    "SSI": "securities",
}

VN30_TICKERS: list[str] = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SHB", "SSB",
    "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM",
]

# node_0 = VNINDEX, nodes 1-30 = VN30_TICKERS
ALL_NODES: list[str] = ["VNINDEX"] + VN30_TICKERS
NODE_IDX:  dict[str, int] = {t: i for i, t in enumerate(ALL_NODES)}


def _pearson(log_ret: pd.DataFrame, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    """
    Pearson correlation matrix using the `window` trading days ending at end_date.
    Only past data is used (no look-ahead).
    """
    mask = log_ret.index <= end_date
    hist = log_ret.loc[mask].iloc[-window:]
    corr = hist.corr()
    return corr


def build_graph(
    log_returns: pd.DataFrame,
    end_date: str | pd.Timestamp,
    corr_window: int = 60,
    corr_threshold: float = 0.4,
) -> Data:
    """
    Build a PyG Data object for the 31-node VN30+VNINDEX graph.

    Args:
        log_returns : DataFrame with columns = ALL_NODES (VNINDEX first),
                      DatetimeIndex, sorted ascending.
        end_date    : Graph construction date (uses only data <= end_date).
        corr_window : Trailing days for Pearson correlation.
        corr_threshold : Edge exists if corr > threshold OR same sector.

    Returns:
        torch_geometric.data.Data with:
            num_nodes  = 31
            edge_index : (2, E) long tensor
            edge_attr  : (E, 2) float tensor [pearson_corr, same_sector_flag]
            loss_mask  : (31,) bool tensor — False for VNINDEX (node_0)
    """
    end_date = pd.Timestamp(end_date)

    # Pearson on stock-to-stock (nodes 1-30 only; VNINDEX handled separately)
    stock_cols = [c for c in log_returns.columns if c in set(VN30_TICKERS)]
    corr_mat = _pearson(log_returns[stock_cols], end_date, corr_window)

    # VNINDEX correlation with each stock
    vnindex_corr: dict[str, float] = {}
    if "VNINDEX" in log_returns.columns:
        for ticker in stock_cols:
            mask = log_returns.index <= end_date
            hist = log_returns.loc[mask].iloc[-corr_window:]
            pair = hist[["VNINDEX", ticker]].dropna()
            if len(pair) >= 10:
                vnindex_corr[ticker] = float(pair.corr().iloc[0, 1])
            else:
                vnindex_corr[ticker] = 0.0

    src_list, dst_list, attr_list = [], [], []

    # Stock-stock edges
    for i, ti in enumerate(stock_cols):
        for j, tj in enumerate(stock_cols):
            if i >= j:
                continue
            c = float(corr_mat.loc[ti, tj]) if (ti in corr_mat and tj in corr_mat.columns) else 0.0
            same_sect = int(SECTOR_MAP.get(ti) == SECTOR_MAP.get(tj)
                            and SECTOR_MAP.get(ti) is not None)
            if c > corr_threshold or same_sect:
                ni, nj = NODE_IDX[ti], NODE_IDX[tj]
                # Undirected: add both directions
                src_list += [ni, nj]
                dst_list += [nj, ni]
                attr_list += [[c, same_sect], [c, same_sect]]

    # VNINDEX hub edges: node_0 -> node_i (directed)
    for ticker in stock_cols:
        c = vnindex_corr.get(ticker, 0.0)
        ni = NODE_IDX[ticker]
        src_list.append(NODE_IDX["VNINDEX"])
        dst_list.append(ni)
        attr_list.append([c, 0])  # same_sector=0 (VNINDEX has no sector)

    # Self-loops for any isolated nodes (degree-0 guard, FR17)
    node_has_edge = set(src_list) | set(dst_list)
    isolated = []
    for n in range(31):
        if n not in node_has_edge:
            src_list.append(n)
            dst_list.append(n)
            attr_list.append([0.0, 0])
            isolated.append(ALL_NODES[n])
    if isolated:
        import warnings
        warnings.warn(
            f"build_graph: {len(isolated)} isolated nodes got self-loops: {isolated}",
            stacklevel=2,
        )

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float)

    # loss_mask: False for VNINDEX (node_0), True for all VN30 stocks
    loss_mask = torch.ones(31, dtype=torch.bool)
    loss_mask[0] = False

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=31,
        loss_mask=loss_mask,
    )
    return data


# ── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.volatility_labels import load_close_prices, compute_log_returns

    PRICES_DIR = "data/raw/prices"
    END_DATE   = "2025-12-31"

    print("Loading prices...")
    close = load_close_prices(PRICES_DIR, tickers=VN30_TICKERS + ["VNINDEX"])
    log_ret = compute_log_returns(close)

    print(f"Building graph at {END_DATE}...")
    g = build_graph(log_ret, end_date=END_DATE)

    n_edges = g.edge_index.shape[1]
    n_stock_edges = sum(1 for i in range(n_edges)
                        if g.edge_index[0, i] != 0 and g.edge_index[1, i] != 0
                        and g.edge_index[0, i] != g.edge_index[1, i])
    n_hub_edges   = sum(1 for i in range(n_edges) if g.edge_index[0, i] == 0)

    print(f"\nGraph stats at {END_DATE}:")
    print(f"  Nodes      : {g.num_nodes}")
    print(f"  Total edges: {n_edges}")
    print(f"  Hub edges  : {n_hub_edges}  (VNINDEX -> stocks)")
    print(f"  Stock edges: {n_stock_edges}")
    print(f"  edge_attr shape: {g.edge_attr.shape}")
    print(f"  loss_mask  : {g.loss_mask.sum()}/31 stocks in loss")

    # Degree stats (stock nodes only)
    degrees = torch.zeros(31, dtype=torch.long)
    for dst in g.edge_index[1]:
        degrees[dst] += 1
    stock_degrees = degrees[1:]  # exclude VNINDEX
    print(f"\nStock node degree stats (in-degree):")
    print(f"  Min: {stock_degrees.min().item()}")
    print(f"  Mean: {stock_degrees.float().mean().item():.2f}")
    print(f"  Max: {stock_degrees.max().item()}")
    if (stock_degrees == 0).any():
        isolated = [VN30_TICKERS[i] for i in (stock_degrees == 0).nonzero(as_tuple=True)[0]]
        print(f"  WARNING isolated (degree=0): {isolated}")
    else:
        print(f"  No isolated nodes")

    print("\nbuild_graph.py OK.")
