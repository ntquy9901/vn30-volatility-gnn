"""
Data Quality Check — VN30 OHLCV
================================
Kiểm tra chất lượng và số lượng dữ liệu để đánh giá khả năng train LSTM/GNN.

Charts được lưu vào: results/data_quality/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# -- CONFIG ------------------------------------------------------------------
DATA_DIR  = Path("data/raw/prices")
OUT_DIR   = Path("results/data_quality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK   = 20     # LSTM lookback window
MAX_H      = 20     # max forecast horizon
# ESS = N_raw / MAX_H (López de Prado 2018 Ch.7 — overlapping labels reduce independence)
# With stride=1, h=20: ESS = raw_samples / 20
# HAR needs ~41 ESS/param (3 params). LSTM(hidden=32) needs ~4000 params -> needs >>41x more ESS.
# No universal paper threshold; we use ESS>=100 as absolute minimum (Branco et al. 2024 implicit).
MIN_ESS    = 100    # minimum EFFECTIVE independent samples (= raw_train / MAX_H)
MIN_TRAIN  = MIN_ESS * MAX_H  # = 2000 raw samples minimum for h=20
CORR_WIN   = 60     # cửa sổ tính Pearson correlation cho GNN graph
MOIRAI_WIN = 200    # cửa sổ Moirai2 feature extraction
STRIDE_GNN = 20     # stride cho GNN batch

VN30_STOCKS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","NVL","PDR","PLX","POW","SAB","SHB","SSB",
    "SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM"
]

SECTORS = {
    "VCB":"Ngân hàng","BID":"Ngân hàng","CTG":"Ngân hàng","MBB":"Ngân hàng",
    "TCB":"Ngân hàng","HDB":"Ngân hàng","VPB":"Ngân hàng","STB":"Ngân hàng",
    "ACB":"Ngân hàng","SSB":"Ngân hàng","SHB":"Ngân hàng","TPB":"Ngân hàng",
    "VIB":"Ngân hàng",
    "VHM":"BĐS","NVL":"BĐS","PDR":"BĐS","BCM":"BĐS",
    "HPG":"Thép/CN","GAS":"Năng lượng","PLX":"Năng lượng","POW":"Năng lượng",
    "GVR":"Nông nghiệp","SAB":"Tiêu dùng","VNM":"Tiêu dùng","MSN":"Tiêu dùng",
    "MWG":"Bán lẻ","FPT":"Công nghệ","VIC":"Đầu tư","BVH":"Bảo hiểm",
    "VJC":"Hàng không","SSI":"Chứng khoán",
}

# -- LOAD DATA ---------------------------------------------------------------

def load_all(stocks=VN30_STOCKS):
    frames = {}
    missing_stocks = []
    for ticker in stocks:
        path = DATA_DIR / f"{ticker}_ohlcv.csv"
        if not path.exists():
            missing_stocks.append(ticker)
            continue
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.sort_index()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        frames[ticker] = df
    if missing_stocks:
        print(f"  [WARN] File không tồn tại: {missing_stocks}")
    return frames

# -- CHART 1: Data Availability Timeline -------------------------------------

def chart_timeline(frames):
    fig, ax = plt.subplots(figsize=(14, 8))
    tickers = sorted(frames.keys())
    for i, t in enumerate(tickers):
        df = frames[t]
        ax.barh(i, (df.index[-1] - df.index[0]).days,
                left=mdates.date2num(df.index[0]),
                height=0.7,
                color=plt.cm.tab20(i % 20), alpha=0.85)
        n = len(df)
        ax.text(mdates.date2num(df.index[-1]) + 20, i,
                f" {n}d", va="center", fontsize=8)

    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.axvline(mdates.date2num(pd.Timestamp("2020-01-01")),
               color="red", ls="--", lw=1.2, label="Val start (2020)")
    ax.axvline(mdates.date2num(pd.Timestamp("2022-01-01")),
               color="orange", ls="--", lw=1.2, label="Test start (2022)")
    ax.set_title("Chart 1 — Data Availability per Stock (trading days)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_timeline.png")
    plt.close()
    print("  [OK] Chart 1: Timeline saved")

# -- CHART 2: Sample Count Assessment (LSTM / GNN) ---------------------------

def chart_sample_counts(frames):
    records = []
    for t, df in frames.items():
        n = len(df.dropna(subset=["log_ret"]))
        lstm_samples = max(0, n - LOOKBACK - MAX_H)
        # train portion ≈ 70%
        train_days  = int(n * 0.70)
        val_days    = int(n * 0.15)
        test_days   = n - train_days - val_days
        lstm_train  = max(0, train_days - LOOKBACK - MAX_H)
        # GNN batch windows: need CORR_WIN + MOIRAI_WIN for first snapshot
        first_snap  = max(CORR_WIN, MOIRAI_WIN)
        gnn_snaps   = max(0, (n - first_snap) // STRIDE_GNN)
        records.append({
            "ticker": t,
            "total_days": n,
            "lstm_samples": lstm_samples,
            "lstm_train": lstm_train,
            "gnn_snapshots": gnn_snaps,
            "sector": SECTORS.get(t, "Khác"),
        })
    df_cnt = pd.DataFrame(records).sort_values("lstm_samples", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    # Total days
    colors = ["#e74c3c" if v < 1500 else "#2ecc71" for v in df_cnt["total_days"]]
    axes[0].barh(df_cnt["ticker"], df_cnt["total_days"], color=colors, alpha=0.85)
    axes[0].axvline(1500, color="red", ls="--", lw=1.2, label="Min 1500d")
    axes[0].set_title("Total Trading Days")
    axes[0].legend(fontsize=8)

    # LSTM train samples
    colors2 = ["#e74c3c" if v < MIN_TRAIN else "#3498db" for v in df_cnt["lstm_train"]]
    axes[1].barh(df_cnt["ticker"], df_cnt["lstm_train"], color=colors2, alpha=0.85)
    axes[1].axvline(MIN_TRAIN, color="red", ls="--", lw=1.2,
                    label=f"Min train={MIN_TRAIN}")
    axes[1].set_title(f"LSTM Train Samples (stride=1, L={LOOKBACK})")
    axes[1].legend(fontsize=8)

    # GNN snapshots
    colors3 = ["#e74c3c" if v < 50 else "#9b59b6" for v in df_cnt["gnn_snapshots"]]
    axes[2].barh(df_cnt["ticker"], df_cnt["gnn_snapshots"], color=colors3, alpha=0.85)
    axes[2].axvline(50, color="red", ls="--", lw=1.2, label="Min 50 snapshots")
    axes[2].set_title(f"GNN Snapshots (stride={STRIDE_GNN}d, first={max(CORR_WIN,MOIRAI_WIN)}d)")
    axes[2].legend(fontsize=8)

    plt.suptitle("Chart 2 — Sample Count Assessment per Stock", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_sample_counts.png")
    plt.close()
    print("  [OK] Chart 2: Sample counts saved")
    return df_cnt

# -- CHART 3: Missing Values & Gaps ------------------------------------------

def chart_missing(frames):
    records = []
    for t, df in frames.items():
        nan_ret  = df["log_ret"].isna().sum()
        nan_cls  = df["close"].isna().sum()
        zero_vol = (df["volume"] == 0).sum()
        records.append({
            "ticker": t,
            "NaN log_ret": nan_ret,
            "NaN close": nan_cls,
            "Zero volume": zero_vol,
        })
    df_m = pd.DataFrame(records).set_index("ticker")

    fig, ax = plt.subplots(figsize=(14, 6))
    df_m.plot(kind="bar", ax=ax, color=["#e74c3c","#e67e22","#3498db"], alpha=0.85)
    ax.set_title("Chart 3 — Missing Values & Zero-Volume Days per Stock")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_missing_values.png")
    plt.close()
    print("  [OK] Chart 3: Missing values saved")

# -- CHART 4: Log-Return Distribution per Stock ------------------------------

def chart_return_dist(frames):
    tickers = sorted(frames.keys())
    n = len(tickers)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.8))
    axes = axes.flatten()

    for i, t in enumerate(tickers):
        ret = frames[t]["log_ret"].dropna()
        axes[i].hist(ret, bins=60, color="#3498db", alpha=0.7, density=True)
        # overlay normal
        mu, sigma = ret.mean(), ret.std()
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        axes[i].plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=1.5)
        sk = ret.skew()
        ku = ret.kurtosis()
        axes[i].set_title(f"{t}\nsk={sk:.2f} ku={ku:.2f}", fontsize=8)
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Chart 4 — Log-Return Distribution (blue=actual, red=normal)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_return_distributions.png")
    plt.close()
    print("  [OK] Chart 4: Return distributions saved")

# -- CHART 5: Rolling Volatility (RV20) per Stock ----------------------------

def chart_rolling_vol(frames):
    tickers = sorted(frames.keys())
    ncols = 5
    nrows = (len(tickers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.5))
    axes = axes.flatten()

    for i, t in enumerate(tickers):
        rv = frames[t]["log_ret"].rolling(20).std() * np.sqrt(252)
        rv.plot(ax=axes[i], lw=0.8, color="#2c3e50", alpha=0.85)
        axes[i].set_title(t, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Ann. Vol", fontsize=7)
        # highlight COVID crash
        axes[i].axvspan(pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-12-31"),
                        alpha=0.12, color="red", label="COVID")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Chart 5 — Rolling 20-day Realized Volatility (Ann.)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_rolling_volatility.png")
    plt.close()
    print("  [OK] Chart 5: Rolling volatility saved")

# -- CHART 6: Correlation Heatmap (full period) ------------------------------

def chart_correlation(frames):
    common_dates = None
    for df in frames.values():
        idx = set(df.dropna(subset=["log_ret"]).index)
        common_dates = idx if common_dates is None else common_dates & idx
    common_dates = sorted(common_dates)

    ret_matrix = pd.DataFrame({
        t: frames[t].loc[common_dates, "log_ret"]
        for t in sorted(frames.keys())
    })

    corr = ret_matrix.corr()

    # sort by sector
    sector_order = sorted(corr.columns,
                          key=lambda x: (SECTORS.get(x, "Z"), x))
    corr = corr.loc[sector_order, sector_order]

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdYlGn",
                vmin=-0.2, vmax=1.0, center=0.4,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                linewidths=0.3)
    ax.set_title("Chart 6 — Log-Return Correlation Matrix (sorted by sector)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_correlation_heatmap.png")
    plt.close()
    print("  [OK] Chart 6: Correlation heatmap saved")
    return corr

# -- CHART 7: PCA of Log-Returns ---------------------------------------------

def chart_pca(frames, corr):
    common_dates = None
    for df in frames.values():
        idx = set(df.dropna(subset=["log_ret"]).index)
        common_dates = idx if common_dates is None else common_dates & idx
    common_dates = sorted(common_dates)

    ret_matrix = pd.DataFrame({
        t: frames[t].loc[common_dates, "log_ret"]
        for t in sorted(frames.keys())
    }).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ret_matrix)

    pca = PCA(n_components=10)
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_

    # PCA scatter on PC1 vs PC2 (stocks as points)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=ret_matrix.columns,
        columns=[f"PC{i+1}" for i in range(10)]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Scree plot ---
    axes[0].bar(range(1, 11), explained * 100, color="#3498db", alpha=0.85)
    axes[0].plot(range(1, 11), np.cumsum(explained) * 100,
                 "r-o", lw=1.5, ms=5, label="Cumulative")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title("Scree Plot")
    axes[0].legend()
    axes[0].axhline(80, color="green", ls="--", lw=1, label="80%")

    # --- PC1 vs PC2 scatter (stocks) ---
    sector_colors = {
        "Ngân hàng": "#e74c3c", "BĐS": "#3498db", "Thép/CN": "#2ecc71",
        "Năng lượng": "#f39c12", "Tiêu dùng": "#9b59b6",
        "Bán lẻ": "#1abc9c", "Công nghệ": "#e67e22", "Đầu tư": "#34495e",
        "Bảo hiểm": "#e91e63", "Hàng không": "#00bcd4",
        "Chứng khoán": "#ff5722", "Nông nghiệp": "#8bc34a", "Khác": "#607d8b",
    }
    for ticker in loadings.index:
        sec = SECTORS.get(ticker, "Khác")
        color = sector_colors.get(sec, "#607d8b")
        axes[1].scatter(loadings.loc[ticker, "PC1"],
                        loadings.loc[ticker, "PC2"],
                        c=color, s=80, alpha=0.9, zorder=3)
        axes[1].annotate(ticker,
                         (loadings.loc[ticker, "PC1"],
                          loadings.loc[ticker, "PC2"]),
                         fontsize=7, ha="left", va="bottom")
    axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    axes[1].set_title("PCA Loadings — Stocks (colored by sector)")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].axvline(0, color="grey", lw=0.5)
    # legend
    for sec, col in sector_colors.items():
        if any(SECTORS.get(t) == sec for t in loadings.index):
            axes[1].scatter([], [], c=col, s=40, label=sec)
    axes[1].legend(fontsize=7, loc="lower right")

    # --- PC1 time series (market factor) ---
    scores = pca.transform(X_scaled)
    pc1_series = pd.Series(scores[:, 0], index=common_dates)
    pc1_series.plot(ax=axes[2], lw=0.8, color="#2c3e50", alpha=0.85)
    axes[2].axvspan(pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-12-31"),
                    alpha=0.15, color="red", label="COVID")
    axes[2].set_title(f"PC1 over Time ({explained[0]*100:.1f}% variance — market factor)")
    axes[2].set_ylabel("PC1 Score")
    axes[2].legend(fontsize=8)

    plt.suptitle("Chart 7 — PCA Analysis of VN30 Log-Returns",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_pca_analysis.png")
    plt.close()
    print("  [OK] Chart 7: PCA analysis saved")

# -- CHART 8: Stationarity & Statistical Summary -----------------------------

def chart_stats_summary(frames):
    from statsmodels.tsa.stattools import adfuller

    records = []
    for t, df in frames.items():
        ret = df["log_ret"].dropna()
        adf_stat, adf_p, *_ = adfuller(ret, maxlag=10, autolag="AIC")
        records.append({
            "ticker": t,
            "mean":   ret.mean() * 252,        # annualized
            "std":    ret.std() * np.sqrt(252), # annualized vol
            "skew":   ret.skew(),
            "kurt":   ret.kurtosis(),           # excess kurtosis
            "adf_p":  adf_p,
            "n_days": len(ret),
            "sector": SECTORS.get(t, "Khác"),
        })

    df_s = pd.DataFrame(records).set_index("ticker")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Annualized mean return
    df_s["mean"].sort_values().plot(kind="barh", ax=axes[0, 0],
                                    color="#2ecc71", alpha=0.85)
    axes[0, 0].axvline(0, color="black", lw=0.8)
    axes[0, 0].set_title("Annualized Mean Return")

    # Annualized volatility
    df_s["std"].sort_values().plot(kind="barh", ax=axes[0, 1],
                                   color="#3498db", alpha=0.85)
    axes[0, 1].set_title("Annualized Volatility (Std)")

    # Skewness
    colors_sk = ["#e74c3c" if v < -0.5 else "#2ecc71" if v > 0.5 else "#bdc3c7"
                 for v in df_s["skew"].sort_values()]
    df_s["skew"].sort_values().plot(kind="barh", ax=axes[0, 2],
                                    color=colors_sk, alpha=0.85)
    axes[0, 2].axvline(0, color="black", lw=0.8)
    axes[0, 2].set_title("Skewness (negative = left-tail heavy)")

    # Excess kurtosis
    colors_ku = ["#e74c3c" if v > 3 else "#2ecc71" for v in df_s["kurt"].sort_values()]
    df_s["kurt"].sort_values().plot(kind="barh", ax=axes[1, 0],
                                    color=colors_ku, alpha=0.85)
    axes[1, 0].axvline(3, color="red", ls="--", lw=1, label="Fat tail threshold")
    axes[1, 0].set_title("Excess Kurtosis (>3 = fat tails)")
    axes[1, 0].legend(fontsize=8)

    # ADF p-value
    colors_adf = ["#2ecc71" if v < 0.05 else "#e74c3c" for v in df_s["adf_p"].sort_values()]
    df_s["adf_p"].sort_values().plot(kind="barh", ax=axes[1, 1],
                                     color=colors_adf, alpha=0.85)
    axes[1, 1].axvline(0.05, color="red", ls="--", lw=1, label="p=0.05")
    axes[1, 1].set_title("ADF Test p-value (green=stationary p<0.05)")
    axes[1, 1].legend(fontsize=8)

    # N trading days
    colors_n = ["#e74c3c" if v < 1500 else "#2ecc71" for v in df_s["n_days"].sort_values()]
    df_s["n_days"].sort_values().plot(kind="barh", ax=axes[1, 2],
                                      color=colors_n, alpha=0.85)
    axes[1, 2].axvline(1500, color="red", ls="--", lw=1, label="Min 1500d")
    axes[1, 2].set_title("Trading Days Available")
    axes[1, 2].legend(fontsize=8)

    plt.suptitle("Chart 8 — Statistical Properties per Stock",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_statistical_summary.png")
    plt.close()
    print("  [OK] Chart 8: Statistical summary saved")
    return df_s

# -- CHART 9: GNN Graph Density over Time ------------------------------------

def chart_gnn_graph(frames):
    common_dates = None
    for df in frames.values():
        idx = set(df.dropna(subset=["log_ret"]).index)
        common_dates = idx if common_dates is None else common_dates & idx
    common_dates = sorted(common_dates)

    ret_matrix = pd.DataFrame({
        t: frames[t].loc[common_dates, "log_ret"]
        for t in sorted(frames.keys())
    }).fillna(0)

    N = len(frames)
    max_edges = N * (N - 1) // 2

    snapshot_dates = []
    densities      = []
    avg_corrs      = []

    for i in range(CORR_WIN, len(common_dates), STRIDE_GNN):
        window = ret_matrix.iloc[i - CORR_WIN: i]
        corr   = window.corr()
        edges  = (corr.values > 0.4).sum() // 2 - N // 2  # remove diagonal
        densities.append(edges / max_edges)
        avg_corrs.append(corr.values[np.triu_indices(N, k=1)].mean())
        snapshot_dates.append(common_dates[i])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(snapshot_dates, densities, lw=1.2, color="#9b59b6", alpha=0.9)
    axes[0].fill_between(snapshot_dates, densities, alpha=0.2, color="#9b59b6")
    axes[0].set_ylabel("Graph Density\n(corr>0.4 edges / max)")
    axes[0].set_title("Chart 9 — GNN Graph Properties over Time (60-day rolling correlation)")
    axes[0].axhline(0.3, color="red", ls="--", lw=1, label="density=0.3")
    axes[0].legend(fontsize=8)

    axes[1].plot(snapshot_dates, avg_corrs, lw=1.2, color="#e67e22", alpha=0.9)
    axes[1].fill_between(snapshot_dates, avg_corrs, alpha=0.2, color="#e67e22")
    axes[1].axvspan(pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-12-31"),
                    alpha=0.15, color="red", label="COVID")
    axes[1].set_ylabel("Avg Pairwise Correlation")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_gnn_graph_dynamics.png")
    plt.close()
    print("  [OK] Chart 9: GNN graph dynamics saved")

# -- CHART 10: Feasibility Summary --------------------------------------------

def chart_feasibility(df_cnt, df_s):
    fig, ax = plt.subplots(figsize=(14, 7))

    tickers = df_cnt["ticker"].values
    lstm_ok = df_cnt["lstm_train"].values >= MIN_TRAIN
    gnn_ok  = df_cnt["gnn_snapshots"].values >= 50
    stat_ok = [df_s.loc[t, "adf_p"] < 0.05 if t in df_s.index else False
               for t in tickers]
    fat_ok  = [df_s.loc[t, "kurt"] > 1 if t in df_s.index else False
               for t in tickers]  # fat tails = good for vol modeling

    x = np.arange(len(tickers))
    w = 0.2
    ax.bar(x - 1.5*w, lstm_ok.astype(int),   w, label="LSTM data OK",  color="#3498db", alpha=0.85)
    ax.bar(x - 0.5*w, gnn_ok.astype(int),    w, label="GNN snaps OK",  color="#9b59b6", alpha=0.85)
    ax.bar(x + 0.5*w, stat_ok,               w, label="Stationary",    color="#2ecc71", alpha=0.85)
    ax.bar(x + 1.5*w, fat_ok,               w, label="Fat tails (vol)", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["[FAIL] Fail", "[OK] Pass"])
    ax.set_title("Chart 10 — Data Feasibility Summary per Stock",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_feasibility_summary.png")
    plt.close()
    print("  [OK] Chart 10: Feasibility summary saved")

# -- PRINT REPORT -------------------------------------------------------------

def print_report(frames, df_cnt, df_s):
    print("\n" + "="*65)
    print("  VN30 DATA QUALITY REPORT")
    print("="*65)

    total = len(frames)
    all_n = [len(df) for df in frames.values()]
    print(f"\n  Stocks loaded:     {total}/30")
    print(f"  Date range:        "
          f"{min(df.index[0] for df in frames.values()).date()} -> "
          f"{max(df.index[-1] for df in frames.values()).date()}")
    print(f"  Avg trading days:  {np.mean(all_n):.0f}  "
          f"(min={min(all_n)}, max={max(all_n)})")

    print(f"\n  --- LSTM Feasibility (stride=1, L={LOOKBACK}) ---")
    lstm_ok = (df_cnt["lstm_train"] >= MIN_TRAIN).sum()
    print(f"  Stocks with >={MIN_TRAIN} train samples: {lstm_ok}/{total}")
    lstm_fail = df_cnt[df_cnt["lstm_train"] < MIN_TRAIN]["ticker"].tolist()
    if lstm_fail:
        print(f"  [WARN] Below threshold: {lstm_fail}")
    else:
        print(f"  [OK] All stocks have sufficient data for LSTM")

    print(f"\n  --- GNN Feasibility (stride={STRIDE_GNN}, corr_win={CORR_WIN}) ---")
    gnn_ok = (df_cnt["gnn_snapshots"] >= 50).sum()
    avg_snaps = df_cnt["gnn_snapshots"].mean()
    print(f"  Stocks with >=50 snapshots: {gnn_ok}/{total}")
    print(f"  Avg snapshots per stock:   {avg_snaps:.0f}")

    print(f"\n  --- Statistical Quality ---")
    if len(df_s) > 0:
        stat_ok = (df_s["adf_p"] < 0.05).sum()
        fat_tail = (df_s["kurt"] > 3).sum()
        print(f"  Stationary (ADF p<0.05):   {stat_ok}/{len(df_s)}")
        print(f"  Fat tails (kurt>3):        {fat_tail}/{len(df_s)}")
        print(f"  Avg annualized vol:        {df_s['std'].mean():.1%}")
        print(f"  Avg excess kurtosis:       {df_s['kurt'].mean():.2f}")

    print(f"\n  --- Data Augmentation Options ---")
    print(f"  Current: {np.mean(all_n):.0f} days/stock × 30 stocks")
    print(f"  Option 1: stride=1 LSTM -> ~{int(np.mean(all_n)-LOOKBACK-MAX_H):,} samples/stock")
    print(f"  Option 2: OHLCV features (O,H,L,C,V) -> 5× feature richness")
    print(f"  Option 3: Rolling RV features -> add RV5, RV10, RV20 as input features")
    print(f"  Option 4: Cross-stock pooling -> train 1 LSTM on all 30 stocks (30× data)")
    print(f"  Option 5: Data synthesis -> synthetic vol paths via GARCH simulation")

    print(f"\n  Charts saved to: {OUT_DIR}/")
    print("="*65 + "\n")

# -- MAIN ---------------------------------------------------------------------

if __name__ == "__main__":
    print("\n Loading VN30 OHLCV data...")
    frames = load_all()
    print(f"  Loaded {len(frames)} stocks\n")

    print(" Generating charts...")
    chart_timeline(frames)
    df_cnt = chart_sample_counts(frames)
    chart_missing(frames)
    chart_return_dist(frames)
    chart_rolling_vol(frames)
    corr = chart_correlation(frames)
    chart_pca(frames, corr)

    print("\n Running statistical tests (ADF — may take ~30s)...")
    df_s = chart_stats_summary(frames)

    print("\n Analyzing GNN graph dynamics...")
    chart_gnn_graph(frames)
    chart_feasibility(df_cnt, df_s)

    print_report(frames, df_cnt, df_s)
    print(f" All done! Open results/data_quality/ to view charts.\n")
