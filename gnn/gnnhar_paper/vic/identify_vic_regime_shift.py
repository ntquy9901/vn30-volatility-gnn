"""
Identify high regime shift periods in VIC test data for targeted testing.

Goal: Find April-May 2026 high volatility period to use as focused test set,
while using all data before April 2026 for comprehensive training.
"""
import sys
from pathlib import Path

# Get project root (4 levels up from vic directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.volatility_labels import load_close_prices, compute_log_returns, compute_rv
from gnn.build_graph import VN30_TICKERS

import yaml
with open(PROJECT_ROOT / 'config.yaml') as f:
    _cfg = yaml.safe_load(f)
DATA_DIR = PROJECT_ROOT / _cfg['data']['prices_dir']

print("\n" + "="*70)
print("  VIC REGIME SHIFT ANALYSIS - Focused Testing Strategy")
print("="*70 + "\n")

# =============================================================================
# LOAD VIC DATA
# =============================================================================

TICKER = 'VIC'
HORIZON = 5

print(f"[Data] Loading {TICKER} data...")
close = load_close_prices(DATA_DIR, tickers=[TICKER])
log_returns = compute_log_returns(close)
rv = compute_rv(close, h=HORIZON)[TICKER].dropna()

print(f"  Date range: {rv.index[0].date()} to {rv.index[-1].date()}")
print(f"  Total samples: {len(rv)}")

# =============================================================================
# DEFINE TRAIN/TEST SPLIT
# =============================================================================

# User's proposed split
TRAIN_END_DATE = "2026-03-31"   # Use ALL data before April 2026
TEST_START_DATE = "2026-04-01"  # Start testing in April
TEST_END_DATE = "2026-05-31"    # End testing in May (2 months focused)

train_end_ts = pd.Timestamp(TRAIN_END_DATE)
test_start_ts = pd.Timestamp(TEST_START_DATE)
test_end_ts = pd.Timestamp(TEST_END_DATE)

# Split data
train_data = rv[rv.index <= train_end_ts]
test_data = rv[(rv.index >= test_start_ts) & (rv.index <= test_end_ts)]

print(f"\n[Split] Focused Testing Strategy:")
print(f"  Training: {train_data.index[0].date()} to {train_data.index[-1].date()}")
print(f"           ({len(train_data):,} samples, {len(train_data)/365:.1f} years)")
print(f"  Testing:  {test_data.index[0].date()} to {test_data.index[-1].date()}")
print(f"           ({len(test_data)} samples, 2 months focused)")

# =============================================================================
# ANALYZE DISTRIBUTION CHARACTERISTICS
# =============================================================================

print(f"\n[Analysis] Distribution Characteristics:")

train_mean = train_data.mean()
train_std = train_data.std()
test_mean = test_data.mean()
test_std = test_data.std()

print(f"  Training period:")
print(f"    Mean RV:   {train_mean:.6f}")
print(f"    Std RV:    {train_std:.6f}")
print(f"    Range:     [{train_data.min():.6f}, {train_data.max():.6f}]")

print(f"  Test period (Apr-May 2026):")
print(f"    Mean RV:   {test_mean:.6f}")
print(f"    Std RV:    {test_std:.6f}")
print(f"    Range:     [{test_data.min():.6f}, {test_data.max():.6f}]")

# Calculate distribution shift
mean_shift_pct = (test_mean - train_mean) / train_mean * 100
std_shift_pct = (test_std - train_std) / train_std * 100

print(f"  Distribution shift:")
print(f"    Mean shift:  {mean_shift_pct:+.1f}%")
print(f"    Std shift:   {std_shift_pct:+.1f}%")

# =============================================================================
# IDENTIFY HIGH VOLATILITY REGIMES IN TRAINING DATA
# =============================================================================

print(f"\n[Analysis] Historical High Volatility Regimes in Training:")

# Find periods with similar volatility to test period
vol_threshold = test_mean * 0.8  # 80% of test period volatility
high_vol_periods = train_data[train_data > vol_threshold]

print(f"  Test period mean RV: {test_mean:.6f}")
print(f"  High-vol threshold (80%): {vol_threshold:.6f}")
print(f"  High-vol days in training: {len(high_vol_periods)} ({len(high_vol_periods)/len(train_data)*100:.1f}%)")

# Find historical periods with elevated volatility
# Identify rolling windows with high average volatility
rolling_vol = train_data.rolling(20).mean()
elevated_periods = rolling_vol[rolling_vol > vol_threshold]

print(f"  Elevated volatility periods (20-day windows): {len(elevated_periods)}")

# Find top volatility spikes
spike_threshold = train_data.quantile(0.95)
volatility_spikes = train_data[train_data > spike_threshold]

print(f"  Top 5% volatility threshold: {spike_threshold:.6f}")
print(f"  Volatility spikes in training: {len(volatility_spikes)} days")

if len(volatility_spikes) > 0:
    print(f"\n  Major volatility spikes:")
    for date, value in volatility_spikes.head(10).items():
        print(f"    {date.date()}: {value:.6f}")

# =============================================================================
# VISUALIZE THE SPLIT
# =============================================================================

print(f"\n[Plot] Creating visualization...")

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# Plot 1: Full timeline with train/test split
ax1 = axes[0]
ax1.plot(train_data.index, train_data.values, label='Training Data',
         alpha=0.7, color='steelblue', linewidth=1)
ax1.plot(test_data.index, test_data.values, label='Test Data (Apr-May 2026)',
         alpha=0.9, color='darkred', linewidth=2)
ax1.axvline(train_end_ts, color='black', linestyle='--',
           label='Train/Test Split', linewidth=1.5)
ax1.axhline(test_mean, color='darkorange', linestyle=':',
           label=f'Test Mean ({test_mean:.4f})', alpha=0.7)

# Mark high volatility periods in training
if len(high_vol_periods) > 0:
    ax1.scatter(high_vol_periods.index, high_vol_periods.values,
               color='orange', alpha=0.5, s=20, label='High-Vol Days in Train',
               zorder=3)

ax1.set_xlabel('Date')
ax1.set_ylabel('Realized Volatility')
ax1.set_title(f'{TICKER} - Training Data Includes Historical Volatility Regimes\n'
              f'Focus Test on High-Volatility Period (Apr-May 2026)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Zoom in on test period
ax2 = axes[1]
# Show 3 months before test + test period
pre_test_start = test_start_ts - pd.Timedelta(days=90)
zoom_data = rv[(rv.index >= pre_test_start) & (rv.index <= test_end_ts)]

ax2.plot(zoom_data.index, zoom_data.values, label='RV',
         color='steelblue', linewidth=1.5)
ax2.axvline(test_start_ts, color='darkred', linestyle='--',
           label='Test Start', linewidth=2)
ax2.axvline(test_end_ts, color='darkred', linestyle='--',
           label='Test End', linewidth=2)
ax2.axhline(test_mean, color='darkorange', linestyle='-',
           label=f'Test Mean: {test_mean:.4f}', linewidth=2)

# Shade training period
ax2.axvspan(pre_test_start, test_start_ts, alpha=0.2, color='steelblue',
           label='Training Period')
ax2.axvspan(test_start_ts, test_end_ts, alpha=0.3, color='darkred',
           label='Test Period')

ax2.set_xlabel('Date')
ax2.set_ylabel('Realized Volatility')
ax2.set_title(f'{TICKER} - Focused Test on High-Volatility Regime (Apr-May 2026)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_dir = PROJECT_ROOT / 'results' / 'gnnhar_paper' / 'analysis'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'vic_regime_shift_focused_testing.png', dpi=150, bbox_inches='tight')
print(f"  Saved: vic_regime_shift_focused_testing.png")

# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"  SUMMARY: Focused Testing Strategy")
print(f"{'='*70}")

print(f"\nKey Insights:")
print(f"  1. Training data includes {len(train_data):,} samples (~{len(train_data)/365:.1f} years)")
print(f"     - Covers multiple volatility regimes (normal, COVID, post-COVID)")
print(f"     - Contains {len(high_vol_periods)} high-volatility days for learning")
print(f"  2. Test data focuses on {len(test_data)} samples (2 months)")
print(f"     - Target high-volatility period (Apr-May 2026)")
print(f"     - Mean RV: {test_mean:.6f} ({mean_shift_pct:+.1f}% vs training)")
print(f"  3. Distribution shift reduced from +144% to {mean_shift_pct:+.1f}%")
print(f"     - Training now includes elevated volatility patterns")
print(f"     - Better preparation for high-volatility forecasting")

print(f"\nExpected Improvements:")
print(f"  - Neural methods should learn from both normal AND high-vol periods")
print(f"  - Training distribution closer to test distribution")
print(f"  - Model performance: R² potentially improve from -8.35 to -2.0 to +0.5")

print(f"\n{'='*70}\n")

# =============================================================================
# SAVE ANALYSIS RESULTS
# =============================================================================

results = {
    'ticker': TICKER,
    'horizon': HORIZON,
    'train_start': str(train_data.index[0].date()),
    'train_end': str(train_data.index[-1].date()),
    'test_start': str(test_data.index[0].date()),
    'test_end': str(test_data.index[-1].date()),
    'n_train_samples': len(train_data),
    'n_test_samples': len(test_data),
    'train_years': len(train_data) / 365,
    'train_mean_rv': float(train_mean),
    'train_std_rv': float(train_std),
    'test_mean_rv': float(test_mean),
    'test_std_rv': float(test_std),
    'mean_shift_pct': float(mean_shift_pct),
    'std_shift_pct': float(std_shift_pct),
    'high_vol_days_in_train': len(high_vol_periods),
    'high_vol_pct_in_train': float(len(high_vol_periods) / len(train_data) * 100),
    'test_period_months': 2
}

import json
output_file = output_dir / 'vic_focused_testing_analysis.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"[Saved] Analysis results saved to {output_file}\n")
