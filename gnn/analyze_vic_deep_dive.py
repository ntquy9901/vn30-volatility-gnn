"""
Deep Dive Analysis: Why is VIC stock so hard to forecast?

Compare VIC (poor performer) vs VNM (good performer) to identify:
1. Distribution shift between train/test
2. Volatility regime changes
3. Outliers and extreme events
4. HAR feature correlations
5. Price/volatility patterns
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml

# Project paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from baselines.har_rv_baseline import fit_har, predict_har

# Load config
with open(ROOT / 'config.yaml') as f:
    CFG = yaml.safe_load(f)
DATA_DIR = ROOT / CFG['data']['prices_dir']

print("\n" + "="*70)
print("  VIC DEEP DIVE ANALYSIS")
print("  Comparing VIC (R2=-0.47) vs VNM (R2=+0.85)")
print("="*70 + "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

TICKERS = ['VIC', 'VNM']
HORIZON = 5
GLOBAL_TEST_START = "2026-01-01"

print("[Data] Loading price data...")
close = load_close_prices(DATA_DIR, tickers=TICKERS)
print(f"  Shape: {close.shape}")

print(f"\n[Data] Computing log returns...")
log_returns = compute_log_returns(close)
print(f"  Shape: {log_returns.shape}")

print(f"\n[Data] Computing RV (h={HORIZON})...")
rv = compute_rv(close, h=HORIZON)
print(f"  VIC RV samples: {len(rv['VIC'].dropna())}")
print(f"  VNM RV samples: {len(rv['VNM'].dropna())}")

# =============================================================================
# BASIC STATISTICS
# =============================================================================

print(f"\n{'='*70}")
print(f"  BASIC STATISTICS")
print(f"{'='*70}\n")

for ticker in TICKERS:
    rv_series = rv[ticker].dropna()
    ret_series = log_returns[ticker].dropna()

    print(f"[{ticker}]")
    print(f"  Price range: ${close[ticker].min():.2f} - ${close[ticker].max():.2f}")
    print(f"  Return stats: mean={ret_series.mean():+.4f}, std={ret_series.std():.4f}")
    print(f"  RV stats: mean={rv_series.mean():.6f}, std={rv_series.std():.6f}")
    print(f"  RV range: [{rv_series.min():.6f}, {rv_series.max():.6f}]")
    print(f"  RV skewness: {rv_series.skew():+.2f}, kurtosis: {rv_series.kurtosis():.2f}")

    # Extreme volatility days
    rv_threshold = rv_series.quantile(0.95)
    extreme_days = rv_series[rv_series > rv_threshold]
    print(f"  Extreme vol days (top 5%): {len(extreme_days)} ({len(extreme_days)/len(rv_series)*100:.1f}%)")
    print()

# =============================================================================
# TRAIN/TEST SPLIT ANALYSIS
# =============================================================================

print(f"{'='*70}")
print(f"  DISTRIBUTION SHIFT ANALYSIS")
print(f"{'='*70}\n")

test_ts = pd.Timestamp(GLOBAL_TEST_START)

for ticker in TICKERS:
    rv_series = rv[ticker].dropna()

    train_rv = rv_series[rv_series.index < test_ts]
    test_rv = rv_series[rv_series.index >= test_ts]

    print(f"[{ticker}]")
    print(f"  Train period: {train_rv.index[0].date()} - {train_rv.index[-1].date()}")
    print(f"  Train RV: mean={train_rv.mean():.6f}, std={train_rv.std():.6f}")
    print(f"  Test period:  {test_rv.index[0].date()} - {test_rv.index[-1].date()}")
    print(f"  Test RV:  mean={test_rv.mean():.6f}, std={test_rv.std():.6f}")

    # Distribution shift
    mean_shift = (test_rv.mean() - train_rv.mean()) / train_rv.mean() * 100
    std_shift = (test_rv.std() - train_rv.std()) / train_rv.std() * 100

    print(f"  Mean shift: {mean_shift:+.1f}%")
    print(f"  Std shift:  {std_shift:+.1f}%")

    # KS test for distribution difference
    from scipy import stats
    ks_stat, ks_pval = stats.ks_2samp(train_rv, test_rv)
    print(f"  KS test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f} {'(DIFFERENT)' if ks_pval < 0.05 else '(similar)'}")
    print()

# =============================================================================
# HAR FEATURE CORRELATIONS
# =============================================================================

print(f"{'='*70}")
print(f"  HAR FEATURE CORRELATIONS")
print(f"{'='*70}\n")

def build_har_features(rv_series, horizon=5):
    """Build HAR features: [RV_d, RV_w, RV_m] -> target"""
    min_history = 22 + horizon
    X_list, y_list = [], []

    for i in range(min_history, len(rv_series) - horizon):
        # Target: future RV
        target = rv_series.iloc[i:i+horizon].mean()

        # Features
        rv_d = rv_series.iloc[i-1:i].mean()      # 1 day
        rv_w = rv_series.iloc[i-5:i].mean()      # 5 days
        rv_m = rv_series.iloc[i-22:i].mean()     # 22 days

        X_list.append([rv_d, rv_w, rv_m])
        y_list.append(target)

    return np.array(X_list), np.array(y_list)

for ticker in TICKERS:
    rv_series = rv[ticker].dropna()
    X, y = build_har_features(rv_series, HORIZON)

    # Calculate correlations
    rv_d_corr = np.corrcoef(X[:, 0], y)[0, 1]
    rv_w_corr = np.corrcoef(X[:, 1], y)[0, 1]
    rv_m_corr = np.corrcoef(X[:, 2], y)[0, 1]

    print(f"[{ticker}]")
    print(f"  Correlation(RV_d, target): {rv_d_corr:+.4f}")
    print(f"  Correlation(RV_w, target): {rv_w_corr:+.4f}")
    print(f"  Correlation(RV_m, target): {rv_m_corr:+.4f}")

    # Feature correlations
    feature_corr = np.corrcoef(X.T)
    print(f"  Correlation(RV_d, RV_w): {feature_corr[0, 1]:+.4f}")
    print(f"  Correlation(RV_d, RV_m): {feature_corr[0, 2]:+.4f}")
    print(f"  Correlation(RV_w, RV_m): {feature_corr[1, 2]:+.4f}")
    print()

# =============================================================================
# REGIME CHANGE DETECTION
# =============================================================================

print(f"{'='*70}")
print(f"  VOLATILITY REGIME ANALYSIS")
print(f"{'='*70}\n")

for ticker in TICKERS:
    rv_series = rv[ticker].dropna()

    # Find regime changes using rolling mean
    rv_rolling = rv_series.rolling(60).mean()  # 60-day rolling mean

    # Detect regime shifts (significant changes in rolling mean)
    rv_rolling_diff = rv_rolling.diff()
    regime_threshold = rv_rolling_diff.std() * 2

    regime_shifts = rv_rolling_diff[abs(rv_rolling_diff) > regime_threshold]

    print(f"[{ticker}]")
    print(f"  Volatility regime shifts detected: {len(regime_shifts)}")

    if len(regime_shifts) > 0:
        print(f"  Major regime changes:")
        for date, change in regime_shifts.head(5).items():
            print(f"    {date.date()}: {change:+.6f}")

    # Compare high vs low volatility regimes
    median_rv = rv_series.median()
    high_vol_days = rv_series[rv_series > median_rv]
    low_vol_days = rv_series[rv_series <= median_rv]

    print(f"  High vol regime (>{median_rv:.6f}): {len(high_vol_days)} days")
    print(f"  Low vol regime (<={median_rv:.6f}): {len(low_vol_days)} days")
    print()

# =============================================================================
# TIME SERIES PLOTS
# =============================================================================

print(f"[Plot] Generating diagnostic plots...")

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('VIC vs VNM: Volatility Forecasting Diagnostic Analysis',
             fontsize=14, fontweight='bold')

for idx, ticker in enumerate(TICKERS):
    rv_series = rv[ticker].dropna()
    train_rv = rv_series[rv_series.index < test_ts]
    test_rv = rv_series[rv_series.index >= test_ts]

    # Plot 1: RV time series with train/test split
    ax = axes[0, idx]
    ax.plot(train_rv.index, train_rv.values, label='Train', alpha=0.7, color='blue')
    ax.plot(test_rv.index, test_rv.values, label='Test', alpha=0.7, color='red')
    ax.axvline(test_ts, color='black', linestyle='--', label='Test Start')
    ax.set_title(f'{ticker}: RV Time Series (h={HORIZON})')
    ax.set_ylabel('Realized Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: RV distribution (train vs test)
    ax = axes[1, idx]
    ax.hist(train_rv.values, bins=50, alpha=0.5, label='Train', color='blue', density=True)
    ax.hist(test_rv.values, bins=50, alpha=0.5, label='Test', color='red', density=True)
    ax.set_title(f'{ticker}: RV Distribution Shift')
    ax.set_xlabel('Realized Volatility')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: HAR feature correlations
    ax = axes[2, idx]
    X, y = build_har_features(rv_series, HORIZON)

    correlations = [
        np.corrcoef(X[:, 0], y)[0, 1],  # RV_d
        np.corrcoef(X[:, 1], y)[0, 1],  # RV_w
        np.corrcoef(X[:, 2], y)[0, 1]   # RV_m
    ]

    bars = ax.bar(['RV_d', 'RV_w', 'RV_m'], correlations,
                   color=['lightcoral', 'lightblue', 'lightgreen'])
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(f'{ticker}: HAR Feature Correlations with Target')
    ax.set_ylabel('Correlation')
    ax.set_ylim([-0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:+.3f}', ha='center', va='bottom' if height > 0 else 'top')

    # Plot 4: Rolling volatility regime
    ax = axes[3, idx]
    rv_rolling = rv_series.rolling(60).mean()
    ax.plot(rv_rolling.index, rv_rolling.values, label='60-day Rolling Mean', color='purple')
    ax.fill_between(rv_series.index, rv_series.values, alpha=0.3, color='gray', label='Daily RV')
    ax.axvline(test_ts, color='black', linestyle='--', label='Test Start')
    ax.set_title(f'{ticker}: Volatility Regime Detection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Realized Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / 'results' / 'gnnhar_paper' / 'analysis' / 'vic_deep_dive_analysis.png',
            dpi=150, bbox_inches='tight')
print(f"  Saved: vic_deep_dive_analysis.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n{'='*70}")
print(f"  SUMMARY: WHY VIC FAILS")
print(f"{'='*70}\n")

print("Key Differences Between VIC (poor) and VNM (good):")
print("1. Distribution Shift: VIC has larger train-test distribution gap")
print("2. Regime Changes: VIC may have more volatility regime breaks")
print("3. Feature Correlations: Check if HAR features are less predictive for VIC")
print("4. Extreme Events: VIC may have more outlier volatility days")
print("\nRecommendations:")
print("- If VIC has high regime shifts: Use adaptive/walk-forward models")
print("- If HAR features are weak: Try alternative features (volume, technical indicators)")
print("- If distribution shift is severe: Use normalization or domain adaptation")
print(f"\n{'='*70}\n")
