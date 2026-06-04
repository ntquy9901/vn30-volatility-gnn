"""
HAR snapshot builder va static graph constructor cho GNN+HAR.

Cac quyet dinh thiet ke chinh:
  1. STATIC GRAPH: 1 graph duy nhat duoc xay dung tu toan bo training period,
     tai su dung cho tat ca snapshots. Graph dong (thay doi theo thoi gian) se
     gay nhieu (Pearson tren cua so ngan = khong on dinh) va ton kem tinh toan.

  2. HAR FEATURES [rv_d, rv_w, rv_m]: trung khop voi dau vao cua HAR-OLS.
     Dung cung features voi HAR dam bao GNN bat dau tu "bieu dien tuong thich voi HAR"
     thay vi phai tu hoc lai cac khai niem vol co ban tu dau.

  3. KHONG LEAKAGE (no lookahead):
       rv_d[t] = |log_return[t]|  (du lieu qua khu, no leakage)
       rv_w[t] = std(log_ret[t-4:t+1])  (qua khu, no leakage)
       Target  = std(log_ret[t+1:t+H])  (tuong lai, dung cach sau t)

  4. STRIDE: lay mau cach quang de giam overlap giua labels lien tiep.
     Ngay t va t+1 co (H-1)/H returns trung nhau trong target window ->
     labels tuong quan cao. stride=H thi cac targets khong con trung nhau.

Graph: 1 undirected static graph (Pearson >= 0.4 HOAC cung nganh).
Features: rolling std (ket thuc tai t, khong leakage).

Xem them: docs/learning/05_graphsage_gnn_har.md
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.build_graph import build_graph, GraphData, ALL_NODES, VN30_TICKERS
from src.volatility_labels import compute_rv, compute_log_returns, compute_past_rv

import dgl

warnings.filterwarnings("ignore")

N_NODES    = 31   # giu lai de tuong thich (compare_gnnhar_paper.py)
N_FEATURES = 3    # [rv_d, rv_w, rv_m]
N_STOCKS   = len(VN30_TICKERS)   # 30


def build_static_graph(
    log_returns: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    corr_threshold: float = 0.4,
) -> GraphData:
    """
    Xay dung 1 static DGL graph dua tren Pearson toan bo training period.
    Dung full train length lam corr_window de nam bat cau truc tuong quan
    dai han on dinh (60-day Pearson co the nhieu, dac biet voi stocks it giao dich).
    """
    train_lr    = log_returns[log_returns.index <= train_end_ts]
    corr_window = len(train_lr)   # dung TẤT CA ngay training cho Pearson on dinh
    return build_graph(
        log_returns,
        end_date=train_end_ts,
        corr_window=corr_window,
        corr_threshold=corr_threshold,
    )


def _compute_har_features(log_returns: pd.DataFrame) -> tuple:
    """
    Tinh 3 HAR features cho moi stock va moi ngay.
    Tra ve (rv_d, rv_w, rv_m) la DataFrame co cung index voi log_returns.

    rv_d[t] = |r_t|                   -- absolute daily return (proxy vol 1 ngay)
              (khong dung rolling(1).std(ddof=1) vi ket qua luon la NaN khi window=1)
    rv_w[t] = std(r_{t-4:t+1}, ddof=1) -- 5-day realized vol (= "HAR weekly" component)
    rv_m[t] = std(r_{t-19:t+1}, ddof=1) -- 20-day realized vol (= "HAR monthly" component)

    Tat ca features ket thuc tai t, target bat dau tai t+1 -> khong co lookahead.
    """
    rv_d = log_returns.abs()                    # vol proxy ngay
    rv_w = compute_past_rv(log_returns, h=5)    # 5-day rolling std
    rv_m = compute_past_rv(log_returns, h=20)   # 20-day rolling std
    return rv_d, rv_w, rv_m


def build_snapshots(
    close: pd.DataFrame,
    log_returns: pd.DataFrame,
    horizons: list,
    stride: int,
    date_start: pd.Timestamp | None = None,
    date_end:   pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Xay dung GNN snapshots cho MIMO (nhieu horizons cung luc).
    Moi "snapshot" = 1 training example tai ngay t:
      X[t] : (31, 3) -- HAR features cho 31 nodes (VNINDEX + 30 stocks) tai thoi diem t
      y[t] : (30, 4) -- RV targets cho 30 VN30 stocks x 4 horizons [1,5,10,20]

    Returns:
      X     : (n_snaps, 31, 3)  float32
      y     : (n_snaps, 30, 4)  float32
      dates : DatetimeIndex do dai n_snaps
    """
    rv_d, rv_w, rv_m = _compute_har_features(log_returns)
    max_h = max(horizons)

    # Tinh truoc target DataFrames cho tat ca horizons
    rv_targets = {}
    for h in horizons:
        rv_targets[h] = compute_rv(close, h=h)

    # Ngay hop le t can thoa man HAI dieu kien:
    # (a) rv_m khong NaN: can 20 ngay lich su (rolling 20)
    # (b) rv_h_max khong NaN: can H ngay tuong lai cho label
    anchor = VN30_TICKERS[0]
    feat_ok = rv_m[anchor].notna() if anchor in rv_m.columns else pd.Series(True, index=log_returns.index)
    tgt_ok  = rv_targets[max_h][anchor].notna() if anchor in rv_targets[max_h].columns else pd.Series(True, index=close.index)

    valid_mask = feat_ok & tgt_ok
    if date_start is not None:
        valid_mask = valid_mask & (log_returns.index >= date_start)
    if date_end is not None:
        valid_mask = valid_mask & (log_returns.index <= date_end)

    valid_dates_all = log_returns.index[valid_mask]
    # Lay mau cach quang theo stride de giam tuong quan giua labels lien tiep.
    # stride=20 cho h=20: cac target windows khong trung nhau (ideal).
    # stride=5 cho h=1: giam 80% overlap (tu 4/5 returns trung -> 0/5 trung).
    snapshot_dates = valid_dates_all[::stride]

    n_snaps  = len(snapshot_dates)
    n_stocks = len(VN30_TICKERS)
    n_horiz  = len(horizons)

    X = np.zeros((n_snaps, N_NODES, N_FEATURES), dtype=np.float32)
    y = np.zeros((n_snaps, n_stocks, n_horiz),   dtype=np.float32)

    for i, date in enumerate(snapshot_dates):
        # Dien node features: 31 nodes (VNINDEX + 30 VN30 stocks)
        for j, node in enumerate(ALL_NODES):
            if node not in log_returns.columns:
                continue
            try:
                X[i, j, 0] = float(rv_d.loc[date, node]) if not pd.isna(rv_d.loc[date, node]) else 0.0
                X[i, j, 1] = float(rv_w.loc[date, node]) if not pd.isna(rv_w.loc[date, node]) else 0.0
                X[i, j, 2] = float(rv_m.loc[date, node]) if not pd.isna(rv_m.loc[date, node]) else 0.0
            except (KeyError, ValueError):
                pass

        # Dien targets: 30 VN30 stocks x so luong horizons
        for k, h in enumerate(horizons):
            for si, ticker in enumerate(VN30_TICKERS):
                if ticker not in rv_targets[h].columns:
                    continue
                try:
                    val = rv_targets[h].loc[date, ticker]
                    y[i, si, k] = float(val) if not pd.isna(val) else 0.0
                except (KeyError, ValueError):
                    pass

    return X, y, snapshot_dates


# ─────────────────────────── SISO (30-node, khong co VNINDEX) ─────────────────

def build_static_graph_30(
    log_returns: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    corr_threshold: float = 0.4,
) -> object:
    """
    Xay dung 30-node DGL graph tu VN30_TICKERS (bo VNINDEX).

    Tai sao bo VNINDEX?
    VNINDEX la chi so gia quyen von hoa cua chinh cac VN30 stocks, nen tuong quan
    cua no voi moi stock la phan tu nhien (tautological). Giu lai VNINDEX tao
    "hub" thong tri message passing: moi stock chi lay tin hieu tu chi so, bo qua
    thong tin ngang hang (peer-to-peer) giua cac stocks.

    Quy tac canh (edge rules):
      - Pearson(train_lr, full period) > corr_threshold (0.4): co-movement return
      - Cung SECTOR_MAP sector: chia se yeu to rui ro nganh
      - Undirected (ca 2 chieu i->j va j->i): cho phep truyen tin hai chieu
      - Self-loop cho nodes bi co lap: GraphSAGE aggregates over incoming edges;
        node khong co edges se bi zero representation neu khong co self-loop.
    """
    from gnn.build_graph import SECTOR_MAP
    # Chi giu VN30 stocks, cat tai diem ket thuc training
    stock_lr = log_returns[[t for t in VN30_TICKERS if t in log_returns.columns]]
    train_lr = stock_lr[stock_lr.index <= train_end_ts]
    # Pearson tren toan bo training period = tuong quan dai han on dinh
    corr_mat = train_lr.corr()

    n = N_STOCKS
    src_list, dst_list = [], []

    for i, ti in enumerate(VN30_TICKERS):
        for j, tj in enumerate(VN30_TICKERS):
            if i >= j:
                continue   # xu ly moi cap 1 lan; them ca 2 chieu o duoi
            c = (float(corr_mat.loc[ti, tj])
                 if (ti in corr_mat.index and tj in corr_mat.columns) else 0.0)
            same_sect = (SECTOR_MAP.get(ti) == SECTOR_MAP.get(tj)
                         and SECTOR_MAP.get(ti) is not None)
            if c > corr_threshold or same_sect:
                # Them CA HAI CHIEU: node j can edge TU i (va nguoc lai)
                # de GraphSAGE co the tap hop thong tin tu ca hai phia.
                src_list += [i, j]
                dst_list += [j, i]

    # Self-loop cho stocks bi co lap (tranh zero-representation trong GraphSAGE)
    node_has_edge = set(src_list) | set(dst_list)
    for ni in range(n):
        if ni not in node_has_edge:
            src_list.append(ni)
            dst_list.append(ni)

    return dgl.graph((src_list, dst_list), num_nodes=n)


def build_snapshots_siso(
    close: pd.DataFrame,
    log_returns: pd.DataFrame,
    horizon: int,
    stride: int,
    date_start: pd.Timestamp | None = None,
    date_end:   pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Xay dung GNN snapshots cho 1 horizon duy nhat (SISO version).
    Goi 1 lan cho moi h trong [1, 5, 10, 20] trong qua trinh training.

    Snapshot tai ngay t:
      X[t] : (30, 3) -- [rv_d, rv_w, rv_m] cho moi stock tai thoi diem t
      y[t] : (30,)   -- RV target = std(log_ret[t+1 : t+H]) cho moi stock

    Returns:
      X     : (n_snaps, 30, 3)  float32
      y     : (n_snaps, 30)     float32
      dates : DatetimeIndex do dai n_snaps
    """
    rv_d, rv_w, rv_m = _compute_har_features(log_returns)
    rv_target = compute_rv(close, h=horizon)

    # Ngay hop le: ca features lan target deu khong NaN
    anchor  = VN30_TICKERS[0]
    feat_ok = (rv_m[anchor].notna() if anchor in rv_m.columns
               else pd.Series(True, index=log_returns.index))
    tgt_ok  = (rv_target[anchor].notna() if anchor in rv_target.columns
               else pd.Series(True, index=close.index))

    valid_mask = feat_ok & tgt_ok
    if date_start is not None:
        valid_mask = valid_mask & (log_returns.index >= date_start)
    if date_end is not None:
        valid_mask = valid_mask & (log_returns.index <= date_end)

    # stride: giam overlap giua labels lien tiep.
    # Vi du h=5, stride=5: cac target windows lien tiep KHONG CO DAY TRUNG NHAU.
    # h=1, stride=5: moi target la |r_t| tren cac ngay khac nhau (khong overlap).
    snapshot_dates = log_returns.index[valid_mask][::stride]
    n_snaps = len(snapshot_dates)

    # Vectorized fill: reindex ca DataFrame 1 lan (nhanh hon vong lap tung ngay)
    def _fill(df: pd.DataFrame) -> np.ndarray:
        # reindex(snapshot_dates): align theo hang (ngay)
        # reindex(columns=VN30_TICKERS): dam bao thu tu cot khop voi VN30_TICKERS
        # nan_to_num: thay NaN (stock thieu data) bang 0.0 (dau vao trung tinh cho GNN)
        return np.nan_to_num(
            df.reindex(snapshot_dates).reindex(columns=VN30_TICKERS).values,
            nan=0.0,
        ).astype(np.float32)

    # Stack theo axis=2: X[t, stock_i, 0] = rv_d, [stock_i, 1] = rv_w, [stock_i, 2] = rv_m
    X = np.stack([_fill(rv_d), _fill(rv_w), _fill(rv_m)], axis=2)  # (n, 30, 3)
    y = _fill(rv_target)                                              # (n, 30)

    return X, y, snapshot_dates
