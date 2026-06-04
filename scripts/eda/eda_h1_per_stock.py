"""
EDA + PCA — VN30 Per-stock Analysis for h=1 LSTM

Câu hỏi trả lời:
  1. Mỗi cổ phiếu có bao nhiêu ngày dữ liệu? Đủ không?
  2. ESS với h=1 vs h=20 khác nhau thế nào?
  3. Phân bố log-return từng stock (fat tails? skew?)
  4. Correlation giữa các stocks (graph density?)
  5. PCA: bao nhiêu PC giải thích variance? Market factor mạnh đến đâu?
  6. Train/Val/Test split thực tế cho từng stock

Chạy: python scripts/eda/eda_h1_per_stock.py
Output: results/eda_h1/
"""
import sys
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/raw/prices")
OUT_DIR     = Path("results/eda_h1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H           = 1              # forecast horizon (this EDA is h=1 specific)
LOOKBACK    = 20             # LSTM input window
TEST_START  = "2026-01-01"
VAL_RATIO   = 0.20           # 80/20 split on pre-test data, per stock
MIN_DAYS    = 500            # minimum raw days to include stock
EXCLUDE     = {"VNINDEX"}    # index, not a tradable stock
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> dict[str, pd.Series]:
    frames = {}
    for f in sorted(data_dir.glob("*_ohlcv.csv")):
        ticker = f.stem.replace("_ohlcv", "")
        if ticker in EXCLUDE:
            continue
        df = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
        frames[ticker] = df["close"]
    return frames

def get_valid_indices(log_ret: pd.Series) -> np.ndarray:
    """All indices that can form a valid h=1 window: need LOOKBACK before, 1 day after."""
    N = len(log_ret)
    return np.array([i for i in range(N) if i >= LOOKBACK and i <= N - 2])

def split_8020(log_ret: pd.Series) -> dict:
    """
    Per-stock 80/20 split on pre-test windows (chronological).
    Returns dict with n_train, n_val, n_test and the val_start_date.
    """
    dates = log_ret.index
    test_start_ts = pd.Timestamp(TEST_START)
    valid_idx = get_valid_indices(log_ret)

    pre_test_idx  = [i for i in valid_idx if dates[i] <  test_start_ts]
    test_idx      = [i for i in valid_idx if dates[i] >= test_start_ts]

    n_pre   = len(pre_test_idx)
    n_train = int(n_pre * (1 - VAL_RATIO))
    n_val   = n_pre - n_train

    val_start_date = dates[pre_test_idx[n_train]] if n_train < n_pre else None

    return {
        "n_train": n_train,
        "n_val":   n_val,
        "n_test":  len(test_idx),
        "val_start_date": val_start_date,
        "train_end_date": dates[pre_test_idx[n_train - 1]] if n_train > 0 else None,
    }

print("Loading data...")
close_dict = load_data(DATA_DIR)
tickers = sorted(close_dict.keys())

# Align to common index for PCA (use intersection of dates)
close_all = pd.DataFrame(close_dict).sort_index()
log_ret_all = np.log(close_all / close_all.shift(1))

# Per-stock stats
records = []
for ticker in tickers:
    s = close_dict[ticker]
    lr = np.log(s / s.shift(1)).dropna()
    N = len(lr)
    if N < MIN_DAYS:
        continue
    sp = split_8020(lr)
    records.append({
        "ticker":          ticker,
        "start":           lr.index[0].date(),
        "end":             lr.index[-1].date(),
        "N_raw":           N,
        "n_train":         sp["n_train"],
        "n_val":           sp["n_val"],
        "n_test":          sp["n_test"],
        "val_start":       sp["val_start_date"],
        "train_end":       sp["train_end_date"],
        "ESS_h1":          sp["n_train"],
        "ESS_h20":         sp["n_train"] // 20,
        "mean":            lr.mean() * 100,
        "std":             lr.std() * 100,
        "max_ret":         lr.max() * 100,
        "min_ret":         lr.min() * 100,
        "kurt":            float(pd.Series(lr.values).kurtosis()),
    })

df = pd.DataFrame(records).sort_values("N_raw", ascending=False).reset_index(drop=True)
tickers_ok = df["ticker"].tolist()

print(f"Stocks loaded: {len(tickers_ok)}")
print(f"\nSample count summary (80/20 split per stock):")
print(f"  Min N_raw   : {df['N_raw'].min()} ({df.loc[df['N_raw'].idxmin(),'ticker']})")
print(f"  Max N_raw   : {df['N_raw'].max()} ({df.loc[df['N_raw'].idxmax(),'ticker']})")
print(f"  Avg N_raw   : {df['N_raw'].mean():.0f}")
print(f"  Min train   : {df['n_train'].min()} ({df.loc[df['n_train'].idxmin(),'ticker']})")
print(f"  Min ESS h=1 : {df['ESS_h1'].min()} (= n_train, no overlap)")
print(f"  Min ESS h=20: {df['ESS_h20'].min()} (= n_train/20)")

# ── Chart 1: Data coverage per stock ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 1a: N_raw bar chart
colors = ["#2ecc71" if n >= 2000 else "#e67e22" if n >= 1000 else "#e74c3c"
          for n in df["N_raw"]]
axes[0].bar(range(len(df)), df["N_raw"], color=colors, alpha=0.85, edgecolor="white")
axes[0].axhline(2000, color="green",  ls="--", lw=1.2, label="2000 days (OK for LSTM)")
axes[0].axhline(1000, color="orange", ls="--", lw=1.2, label="1000 days (borderline)")
axes[0].set_xticks(range(len(df)))
axes[0].set_xticklabels(df["ticker"], rotation=45, ha="right", fontsize=9)
axes[0].set_ylabel("Trading Days")
axes[0].set_title("Chart 1a — Raw Data Coverage per Stock (green=OK, orange=borderline, red=limited)")
axes[0].legend(fontsize=9)
for i, row in df.iterrows():
    axes[0].text(i, row["N_raw"] + 50, str(row["N_raw"]),
                 ha="center", fontsize=7, color="#2c3e50")

# 1b: Stacked bar train/val/test
x = range(len(df))
axes[1].bar(x, df["n_train"], label="Train (80% pre-test)", color="#3498db", alpha=0.85)
axes[1].bar(x, df["n_val"],   bottom=df["n_train"],
            label="Val (20% pre-test)", color="#e67e22", alpha=0.85)
axes[1].bar(x, df["n_test"],  bottom=df["n_train"]+df["n_val"],
            label="Test (2026-01-01 onwards)", color="#e74c3c", alpha=0.85)
axes[1].set_xticks(range(len(df)))
axes[1].set_xticklabels(df["ticker"], rotation=45, ha="right", fontsize=9)
axes[1].set_ylabel("Windows (h=1)")
axes[1].set_title("Chart 1b — Train/Val/Test Windows per Stock (h=1, 80/20 split per stock)")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_data_coverage.png")
plt.close()
print("[OK] Chart 1: Data coverage saved")

# ── Chart 2: ESS comparison h=1 vs h=20 ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
w = 0.4
x = np.arange(len(df))

axes[0].bar(x - w/2, df["ESS_h1"],  w, label="ESS (h=1)",  color="#2ecc71", alpha=0.85)
axes[0].bar(x + w/2, df["ESS_h20"], w, label="ESS (h=20)", color="#e74c3c", alpha=0.85)
axes[0].axhline(100, color="black", ls="--", lw=1.2, label="MIN_ESS=100")
axes[0].set_xticks(x)
axes[0].set_xticklabels(df["ticker"], rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Effective Sample Size")
axes[0].set_title("Chart 2a — ESS: h=1 vs h=20\n(h=1: all stocks pass; h=20: many fail)")
axes[0].legend(fontsize=9)
axes[0].set_yscale("log")

# ESS ratio
ess_ratio = df["ESS_h1"] / df["ESS_h20"]
axes[1].bar(x, ess_ratio, color="#9b59b6", alpha=0.85)
axes[1].axhline(20, color="red", ls="--", lw=1.2, label="Ratio=20 (expected h=20)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(df["ticker"], rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("ESS_h1 / ESS_h20")
axes[1].set_title("Chart 2b — ESS Ratio: how much h=1 gains over h=20\n(ratio=20 means x20 more independent obs)")
axes[1].legend(fontsize=9)

plt.suptitle("Chart 2 — Why h=1 is data-efficient: ESS = N_train (no label overlap)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_ess_comparison.png")
plt.close()
print("[OK] Chart 2: ESS comparison saved")

# ── Chart 3: Return distribution — histogram + KDE ─────────────────────────
fig, axes = plt.subplots(5, 6, figsize=(18, 15))
axes = axes.flatten()

for i, row in df.iterrows():
    if i >= len(axes):
        break
    ax = axes[i]
    ticker = row["ticker"]
    lr = np.log(close_dict[ticker] / close_dict[ticker].shift(1)).dropna()
    lr_pct = lr.values * 100

    ax.hist(lr_pct, bins=60, density=True, color="#3498db", alpha=0.6, edgecolor="none")
    # KDE
    xr = np.linspace(lr_pct.min(), lr_pct.max(), 200)
    kde = stats.gaussian_kde(lr_pct)
    ax.plot(xr, kde(xr), lw=1.5, color="#2c3e50")
    # Normal reference
    norm_pdf = stats.norm.pdf(xr, lr_pct.mean(), lr_pct.std())
    ax.plot(xr, norm_pdf, lw=1.2, color="#e74c3c", ls="--", label="Normal")
    ax.set_title(f"{ticker}\nstd={row['std']:.2f}%  max={row['max_ret']:.1f}%",
                 fontsize=8, pad=2)
    ax.set_xlim(-12, 12)
    ax.set_yticks([])
    ax.axvline(0, color="gray", lw=0.5)

# Hide unused axes
for j in range(len(df), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Chart 3 — Return Distribution per Stock (blue=actual, red dashed=Normal)\n"
             "Fat tails (kurt>>0) confirm non-Gaussian; important for loss function choice",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_return_distributions.png")
plt.close()
print("[OK] Chart 3: Return distributions saved")

# ── Chart 4: Rolling volatility timeline ────────────────────────────────────
SAMPLE_TICKERS = [t for t in ["VCB", "HPG", "FPT", "MBB", "VNM", "SSB"] if t in tickers_ok]
fig, axes = plt.subplots(len(SAMPLE_TICKERS), 1, figsize=(16, 12), sharex=True)

test_start_ts = pd.Timestamp(TEST_START)

for ax, ticker in zip(axes, SAMPLE_TICKERS):
    lr = np.log(close_dict[ticker] / close_dict[ticker].shift(1)).dropna()
    roll_vol = lr.abs().rolling(20).mean() * 100  # 20-day rolling |ret|
    # Get per-stock val_start from df
    row = df[df["ticker"] == ticker].iloc[0]
    train_end_ts = pd.Timestamp(row["train_end"])
    val_start_ts = pd.Timestamp(row["val_start"])
    ax.plot(roll_vol.index, roll_vol.values, lw=0.8, color="#2c3e50", alpha=0.9)
    ax.axvspan(lr.index[0], train_end_ts, alpha=0.06, color="#3498db", label="Train (80%)")
    ax.axvspan(val_start_ts, test_start_ts - pd.Timedelta(days=1),
               alpha=0.1, color="#e67e22", label="Val (20%)")
    ax.axvspan(test_start_ts, lr.index[-1], alpha=0.12, color="#e74c3c", label="Test")
    ax.axvline(pd.Timestamp("2020-03-01"), color="red", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel(ticker, fontsize=9, rotation=0, labelpad=40)
    ax.set_yticks([])
    if ax == axes[0]:
        ax.legend(fontsize=8, loc="upper left", ncol=3)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
plt.suptitle("Chart 4 — Rolling 20-day |Return| (proxy for vol) over time\n"
             "Blue=Train (80%) | Orange=Val (20%) | Red=Test (2026+) | Dashed=COVID",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_rolling_vol_timeline.png")
plt.close()
print("[OK] Chart 4: Rolling vol timeline saved")

# ── Chart 5: Correlation heatmap (train period) ──────────────────────────────
# Use VCB train_end as proxy for the common train cutoff (80th percentile dates differ per stock)
vcb_train_end = pd.Timestamp(df[df["ticker"]=="VCB"]["train_end"].iloc[0])
lr_train = log_ret_all[log_ret_all.index <= vcb_train_end][tickers_ok].dropna(how="all")
# Use only stocks with sufficient overlap
corr_mat = lr_train.corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = corr_mat.isnull()
sns.heatmap(corr_mat, ax=ax, cmap="RdYlGn", center=0, vmin=-0.2, vmax=1.0,
            linewidths=0.3, linecolor="white",
            xticklabels=corr_mat.columns, yticklabels=corr_mat.index,
            mask=mask, annot=False, square=True)
ax.set_title("Chart 5 — Pearson Correlation Heatmap (train period, log-returns)\n"
             "High corr = dense GNN graph; low corr = sparser edges",
             fontsize=11, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_correlation_heatmap.png")
plt.close()
print("[OK] Chart 5: Correlation heatmap saved")

# ── Chart 6: PCA on log-returns ──────────────────────────────────────────────
# Use dates where all stocks have data (intersection)
lr_common = lr_train[tickers_ok].dropna()
print(f"\nPCA: using {len(lr_common)} common dates x {len(tickers_ok)} stocks")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(lr_common.values)

pca = PCA(n_components=min(10, len(tickers_ok)))
pca.fit(X_scaled)
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 6a: Scree plot
axes[0].bar(range(1, len(explained)+1), explained*100, color="#3498db", alpha=0.85)
axes[0].plot(range(1, len(cumulative)+1), cumulative*100, "o-", color="#e74c3c", lw=2)
axes[0].axhline(80, color="green", ls="--", lw=1, label="80% threshold")
n_80 = np.argmax(cumulative >= 0.80) + 1
axes[0].axvline(n_80, color="green", ls="--", lw=1)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title(f"Chart 6a — Scree Plot\nPC1={explained[0]*100:.1f}%, {n_80} PCs to reach 80%")
axes[0].legend(fontsize=9)
axes[0].set_xticks(range(1, len(explained)+1))

# 6b: PC1 loadings (market factor)
loadings_pc1 = pd.Series(pca.components_[0], index=tickers_ok)
colors_l = ["#2ecc71" if v > 0 else "#e74c3c" for v in loadings_pc1]
axes[1].barh(range(len(loadings_pc1)), loadings_pc1.values, color=colors_l, alpha=0.85)
axes[1].set_yticks(range(len(loadings_pc1)))
axes[1].set_yticklabels(loadings_pc1.index, fontsize=8)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title(f"Chart 6b — PC1 Loadings ({explained[0]*100:.1f}% var)\n"
                  f"Market factor: {'uniform' if loadings_pc1.std()<0.05 else 'differentiated'}")
axes[1].set_xlabel("Loading")

# 6c: PC2 loadings (sector/idiosyncratic factor)
loadings_pc2 = pd.Series(pca.components_[1], index=tickers_ok)
colors_l2 = ["#3498db" if v > 0 else "#e67e22" for v in loadings_pc2]
axes[2].barh(range(len(loadings_pc2)), loadings_pc2.values, color=colors_l2, alpha=0.85)
axes[2].set_yticks(range(len(loadings_pc2)))
axes[2].set_yticklabels(loadings_pc2.index, fontsize=8)
axes[2].axvline(0, color="black", lw=0.8)
axes[2].set_title(f"Chart 6c — PC2 Loadings ({explained[1]*100:.1f}% var)\n"
                  f"Sector/idiosyncratic factor")
axes[2].set_xlabel("Loading")

plt.suptitle("Chart 6 — PCA on Log-Returns (train period)\n"
             "PC1 = market factor strength; PC2+ = sector differentiation",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_pca_analysis.png")
plt.close()
print("[OK] Chart 6: PCA analysis saved")

# ── Chart 7: QQ plots — normality check ─────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()
sample_tickers_qq = tickers_ok[:12]

for i, ticker in enumerate(sample_tickers_qq):
    ax = axes[i]
    lr = np.log(close_dict[ticker] / close_dict[ticker].shift(1)).dropna().values
    (q_theor, q_obs), (slope, intercept, r) = stats.probplot(lr, dist="norm")
    ax.plot(q_theor, q_obs, ".", markersize=2, color="#3498db", alpha=0.5)
    ax.plot(q_theor, slope*np.array(q_theor)+intercept, "-", color="#e74c3c", lw=1.5)
    max_r = pd.Series(lr).max() * 100
    ax.set_title(f"{ticker} (max={max_r:.1f}%)", fontsize=9)
    ax.set_xlabel("Theoretical Quantile", fontsize=7)
    ax.set_ylabel("Sample Quantile", fontsize=7)
    ax.tick_params(labelsize=7)

plt.suptitle("Chart 7 — QQ Plots vs Normal Distribution\n"
             "Deviation at tails = fat-tail evidence (high kurtosis = bad for MSE loss)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "07_qq_plots.png")
plt.close()
print("[OK] Chart 7: QQ plots saved")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("  SUMMARY -- Per-stock EDA (h=1, 80/20 split per stock, test=2026-01-01 onwards)")
print(f"{'='*85}")
print("  %-5s %10s %7s %7s %12s %6s %6s %7s %7s %8s" %
      ("Ticker","Start","N_raw","Train","Val_start","Val","Test","ESS_h1","std%","max_ret%"))
print("-" * 90)
for _, row in df.iterrows():
    flag = " [low]" if row["ESS_h1"] < 500 else ""
    val_s = str(row["val_start"])[:10] if row["val_start"] is not None else "N/A"
    print("  %-5s %10s %7d %7d %12s %6d %6d %7d %7.3f %8.2f%s" % (
        row["ticker"], str(row["start"]), row["N_raw"],
        row["n_train"], val_s, row["n_val"], row["n_test"],
        row["ESS_h1"], row["std"], row["max_ret"], flag))

n_80_pct_pcs = np.searchsorted(np.cumsum(explained), 0.80) + 1
print(f"\n  Split ratio: 80% train / 20% val of pre-test windows (per stock)")
print(f"  Stocks with <500 train: {df[df['n_train']<500]['ticker'].tolist()}")
print(f"  All ESS(h=1) >= 100: {(df['ESS_h1']>=100).all()}")
print(f"  Avg kurtosis: {df['kurt'].mean():.1f} (Normal excess=0; fat tails!)")
print(f"  PC1 variance: {explained[0]*100:.1f}% | PCs for 80%%: {n_80_pct_pcs}")
print(f"\n  Charts saved to: {OUT_DIR}/")
print(f"{'='*85}")
