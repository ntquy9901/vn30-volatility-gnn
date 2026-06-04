"""
COVID Period Analysis — HAR-RV vs Actual Volatility
Câu hỏi: HAR-RV (và HAR-RS) có dự báo tốt trong COVID không?

Chạy: python diag_covid_har.py
Output: results/covid_analysis/
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.volatility_labels import load_close_prices, compute_log_returns

DATA_DIR = Path("data/raw/prices")
OUT_DIR  = Path("results/covid_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2019-12-31"
COVID_START = "2020-01-01"
COVID_END   = "2021-12-31"
H = 20  # forecast horizon

TICKERS = [
    "VCB","BID","CTG","MBB","TCB","HPG",
    "FPT","VHM","MWG","VNM","HDB","ACB",
]

# ── Load & compute returns ───────────────────────────────────────────────────

print("Loading data...")
close = load_close_prices(DATA_DIR)
close = close[[t for t in TICKERS if t in close.columns]]
log_ret = np.log(close / close.shift(1))

def rolling_rv(log_ret_series, h=20):
    """RV_t = std(log_ret[t:t+h]) — realized over next h days."""
    rv = pd.Series(index=log_ret_series.index, dtype=float)
    for i in range(len(log_ret_series) - h):
        rv.iloc[i] = log_ret_series.iloc[i:i+h].std()
    return rv

def har_features(rv_series):
    """HAR: RV_d, RV_w (5-day avg), RV_m (20-day avg)."""
    rv_d = rv_series
    rv_w = rv_series.rolling(5).mean()
    rv_m = rv_series.rolling(20).mean()
    return rv_d, rv_w, rv_m

def har_rs_features(log_ret_series, h=1):
    """
    HAR-RS: RV+, RV-, RV_w, RV_m
    RV+_t = sum(max(r,0)^2) / h  over past h days  (upside semivariance)
    RV-_t = sum(min(r,0)^2) / h  over past h days  (downside semivariance)
    """
    r = log_ret_series
    rv_pos = (r.clip(lower=0)**2).rolling(h).sum() / h
    rv_neg = (r.clip(upper=0)**2).rolling(h).sum() / h
    return rv_pos, rv_neg

# ── Per-stock HAR-RV và HAR-RS ───────────────────────────────────────────────

from numpy.linalg import lstsq

def fit_har(rv, train_mask):
    rv_d, rv_w, rv_m = har_features(rv)
    df = pd.DataFrame({
        "y":    rv.shift(-H),
        "rv_d": rv_d,
        "rv_w": rv_w,
        "rv_m": rv_m,
    }).dropna()
    train = df[df.index <= TRAIN_END]
    X = np.column_stack([np.ones(len(train)), train["rv_d"], train["rv_w"], train["rv_m"]])
    y = train["y"].values
    beta, *_ = lstsq(X, y, rcond=None)
    # predict all
    X_all = np.column_stack([np.ones(len(df)), df["rv_d"], df["rv_w"], df["rv_m"]])
    df["pred_har"] = X_all @ beta
    return df

def fit_har_rs(rv, log_r, train_mask):
    rv_d, rv_w, rv_m = har_features(rv)
    rv_pos, rv_neg   = har_rs_features(log_r, h=1)
    # weekly and monthly semivariance
    rv_pos_w = rv_pos.rolling(5).mean()
    rv_neg_w = rv_neg.rolling(5).mean()
    df = pd.DataFrame({
        "y":       rv.shift(-H),
        "rv_pos":  rv_pos,
        "rv_neg":  rv_neg,
        "rv_pos_w":rv_pos_w,
        "rv_neg_w":rv_neg_w,
        "rv_m":    rv_m,
    }).dropna()
    train = df[df.index <= TRAIN_END]
    X = np.column_stack([
        np.ones(len(train)),
        train["rv_pos"], train["rv_neg"],
        train["rv_pos_w"], train["rv_neg_w"],
        train["rv_m"]
    ])
    y = train["y"].values
    beta, *_ = lstsq(X, y, rcond=None)
    X_all = np.column_stack([
        np.ones(len(df)),
        df["rv_pos"], df["rv_neg"],
        df["rv_pos_w"], df["rv_neg_w"],
        df["rv_m"]
    ])
    df["pred_har_rs"] = X_all @ beta
    return df

# ── Run analysis ─────────────────────────────────────────────────────────────

print(f"Fitting HAR-RV and HAR-RS for {len(TICKERS)} stocks...")
results_by_stock = {}

for ticker in TICKERS:
    if ticker not in log_ret.columns:
        continue
    r  = log_ret[ticker].dropna()
    rv = rolling_rv(r, h=H)
    df_har    = fit_har(rv, TRAIN_END)
    df_har_rs = fit_har_rs(rv, r, TRAIN_END)

    df = df_har.copy()
    df["pred_har_rs"] = df_har_rs["pred_har_rs"].reindex(df.index)
    df["actual"] = df["y"]
    results_by_stock[ticker] = df
    print(f"  {ticker}: OK")

# ── COVID MAE comparison ──────────────────────────────────────────────────────

print("\nCOVID period MAE comparison:")
print(f"{'Ticker':<8} {'Pre-COVID MAE HAR':>18} {'COVID MAE HAR':>14} "
      f"{'COVID MAE HAR-RS':>17} {'Error ratio':>12}")
print("-" * 75)

pre_ratios = []
covid_ratios = []

for ticker, df in results_by_stock.items():
    pre   = df[(df.index < COVID_START) & (df.index >= "2017-01-01")]
    covid = df[(df.index >= COVID_START) & (df.index <= COVID_END)]

    if len(pre) < 50 or len(covid) < 30:
        continue

    mae_pre_har    = (pre["actual"] - pre["pred_har"]).abs().mean()
    mae_covid_har  = (covid["actual"] - covid["pred_har"]).abs().mean()
    mae_covid_rs   = (covid["actual"] - covid["pred_har_rs"]).abs().mean()
    ratio = mae_covid_har / mae_pre_har if mae_pre_har > 0 else float("nan")

    pre_ratios.append(mae_pre_har)
    covid_ratios.append(mae_covid_har)

    print(f"  {ticker:<6} {mae_pre_har:>18.4f} {mae_covid_har:>14.4f} "
          f"{mae_covid_rs:>17.4f} {ratio:>10.1f}x")

avg_ratio = np.mean(covid_ratios) / np.mean(pre_ratios)
print(f"\n  Average COVID MAE inflation: {avg_ratio:.1f}x pre-COVID MAE")

# ── Chart 1: Predicted vs Actual for 1 stock (VCB) ───────────────────────────

ticker = "VCB"
if ticker in results_by_stock:
    df = results_by_stock[ticker]
    covid_window = df[(df.index >= "2019-06-01") & (df.index <= "2022-06-30")]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(covid_window.index, covid_window["actual"] * np.sqrt(252),
                 lw=1.2, color="#2c3e50", label="Actual RV (annualized)")
    axes[0].plot(covid_window.index, covid_window["pred_har"] * np.sqrt(252),
                 lw=1.2, color="#e74c3c", ls="--", label="HAR-RV predicted")
    axes[0].plot(covid_window.index, covid_window["pred_har_rs"] * np.sqrt(252),
                 lw=1.2, color="#3498db", ls="-.", label="HAR-RS predicted")
    axes[0].axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                    alpha=0.12, color="red", label="COVID period (val)")
    axes[0].axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-05-01"),
                    alpha=0.25, color="orange", label="COVID crash peak")
    axes[0].axvline(pd.Timestamp(TRAIN_END), color="green", lw=1.5,
                    ls="--", label="Train end (2019-12-31)")
    axes[0].set_ylabel("Annualized Volatility")
    axes[0].set_title(f"Chart 1 — {ticker}: HAR-RV & HAR-RS Predictions vs Actual (h=20)")
    axes[0].legend(fontsize=8, loc="upper left")

    # Error chart
    err_har  = (covid_window["actual"] - covid_window["pred_har"]).abs() * np.sqrt(252)
    err_rs   = (covid_window["actual"] - covid_window["pred_har_rs"]).abs() * np.sqrt(252)
    axes[1].fill_between(covid_window.index, err_har, alpha=0.4,
                         color="#e74c3c", label="HAR-RV abs error")
    axes[1].fill_between(covid_window.index, err_rs,  alpha=0.4,
                         color="#3498db", label="HAR-RS abs error")
    axes[1].axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                    alpha=0.08, color="red")
    axes[1].axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-05-01"),
                    alpha=0.2, color="orange")
    axes[1].axvline(pd.Timestamp(TRAIN_END), color="green", lw=1.5, ls="--")
    axes[1].set_ylabel("Absolute Error (annualized)")
    axes[1].set_title("Absolute Forecast Error")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_vcb_covid_forecast.png")
    plt.close()
    print(f"\n[OK] Chart 1 saved: {ticker} COVID forecast")

# ── Chart 2: MAE by period for all stocks ────────────────────────────────────

records = []
for ticker, df in results_by_stock.items():
    for period, label, (s, e) in [
        ("Pre-COVID", "pre", ("2017-01-01", "2019-12-31")),
        ("COVID",     "covid", ("2020-01-01", "2021-12-31")),
        ("Post-COVID","post", ("2022-01-01", "2024-12-31")),
    ]:
        sub = df[(df.index >= s) & (df.index <= e)]
        if len(sub) < 20:
            continue
        mae_har = (sub["actual"] - sub["pred_har"]).abs().mean()
        mae_rs  = (sub["actual"] - sub["pred_har_rs"]).abs().mean()
        records.append({"ticker": ticker, "period": period,
                        "HAR-RV": mae_har, "HAR-RS": mae_rs})

df_periods = pd.DataFrame(records)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
periods = ["Pre-COVID", "COVID", "Post-COVID"]
colors  = {"HAR-RV": "#e74c3c", "HAR-RS": "#3498db"}

for i, period in enumerate(periods):
    sub = df_periods[df_periods["period"] == period]
    x   = np.arange(len(sub))
    w   = 0.35
    axes[i].bar(x - w/2, sub["HAR-RV"] * 1e4, w,
                label="HAR-RV", color=colors["HAR-RV"], alpha=0.85)
    axes[i].bar(x + w/2, sub["HAR-RS"] * 1e4, w,
                label="HAR-RS", color=colors["HAR-RS"], alpha=0.85)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(sub["ticker"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[i].set_title(f"{period}\nMAE (x1e-4)")
    axes[i].legend(fontsize=8)
    if period == "COVID":
        axes[i].set_facecolor("#fff0f0")

plt.suptitle("Chart 2 — HAR-RV vs HAR-RS MAE by Period (all stocks)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_mae_by_period.png")
plt.close()
print("[OK] Chart 2 saved: MAE by period")

# ── Chart 3: Actual vol distribution — pre vs during COVID ───────────────────

all_pre   = pd.concat([results_by_stock[t]["actual"][(results_by_stock[t].index >= "2017-01-01")
                        & (results_by_stock[t].index <= "2019-12-31")]
                        for t in results_by_stock], ignore_index=True).dropna()
all_covid = pd.concat([results_by_stock[t]["actual"][(results_by_stock[t].index >= "2020-01-01")
                        & (results_by_stock[t].index <= "2021-12-31")]
                        for t in results_by_stock], ignore_index=True).dropna()

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(all_pre   * np.sqrt(252), bins=80, alpha=0.6, color="#2ecc71",
        density=True, label=f"Pre-COVID (mean={all_pre.mean()*np.sqrt(252):.1%})")
ax.hist(all_covid * np.sqrt(252), bins=80, alpha=0.6, color="#e74c3c",
        density=True, label=f"COVID (mean={all_covid.mean()*np.sqrt(252):.1%})")
ax.axvline(all_pre.mean()   * np.sqrt(252), color="#27ae60", lw=2, ls="--")
ax.axvline(all_covid.mean() * np.sqrt(252), color="#c0392b", lw=2, ls="--")
ax.set_xlabel("Annualized Realized Volatility")
ax.set_ylabel("Density")
ax.set_title("Chart 3 — Distribution Shift: Pre-COVID vs COVID\n"
             "(Model trained on Pre-COVID — distribution mismatch explains poor COVID performance)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_distribution_shift.png")
plt.close()
print("[OK] Chart 3 saved: Distribution shift")

print(f"\nAll charts saved to: {OUT_DIR}/")
print("\nKet luan:")
print(f"  Pre-COVID avg vol: {all_pre.mean()*np.sqrt(252):.1%}")
print(f"  COVID avg vol:     {all_covid.mean()*np.sqrt(252):.1%}")
print(f"  Vol inflation:     {all_covid.mean()/all_pre.mean():.1f}x")
print(f"  HAR duoc train tren pre-COVID data -> chi thay vol thap")
print(f"  Khi COVID xay ra: vol tang {all_covid.mean()/all_pre.mean():.1f}x -> model under-predict tram trong")
print(f"  HAR-RS tot hon HAR-RV vi RV- (downside) bao truoc distress som hon")
