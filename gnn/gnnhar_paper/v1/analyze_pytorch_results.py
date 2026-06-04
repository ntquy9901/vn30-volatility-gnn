"""
Analysis of PyTorch GNNHAR Results

DATE: 2026-05-31
STATUS: CRITICAL FINDING - Existing PyTorch results are SINGLE-STOCK (not multi-stock)
"""

print("\n" + "="*70)
print("  PYTORCH GNNHAR RESULTS ANALYSIS")
print("="*70 + "\n")

import pandas as pd

df = pd.read_csv('results/gnnhar_paper/gnnhar_paper_results.csv')

print(f"[Data Structure]")
print(f"  Total rows: {len(df)}")
print(f"  Unique stocks: {df['ticker'].nunique()}")
print(f"  Unique models: {df['model'].nunique()}")
print(f"  Models: {df['model'].unique()}")
print(f"  N_test per stock: {df['n_test'].unique()}")
print()

# Verify this is per-stock training
print(f"[Training Type Diagnosis]")
print(f"  Structure: {len(df)} rows = {df['ticker'].nunique()} stocks × {df['model'].nunique()} models")
print(f"  This indicates: PER-STOCK training (separate model trained per stock)")
print()

# Compare with sklearn results
print(f"[Performance Comparison]")
print(f"  sklearn GHAR (multi-stock, our implementation):")
print(f"    R² = +0.7538 (beats HAR OLS)")
print(f"    Data: 96,390 samples (30 stocks pooled together)")
print()
print(f"  PyTorch GHAR (per-stock, existing results):")
ghar_mean = df[df['model'] == 'GHAR']['gnn_r2'].mean()
print(f"    Mean R² = {ghar_mean:+.4f}")
print(f"    Data: ~1,260 samples per stock (single-stock training)")
print()

print(f"[KEY FINDING]")
print(f"  The existing PyTorch results are from SINGLE-STOCK training")
print(f"  (same approach that failed in SINGLE_STOCK_FAILURE_ANALYSIS.md)")
print()
print(f"  Single-stock GNNHAR fails because:")
print(f"    1. No graph structure (N=1 -> identity adjacency)")
print(f"    2. Insufficient data (~1,260 samples vs. 1000+ parameters)")
print(f"    3. Cannot leverage cross-stock spillover effects")
print()

print(f"[CONCLUSION]")
print(f"  PyTorch GNNHAR must be implemented with MULTI-STOCK training")
print(f"  Current implementation: PER-STOCK (incorrect for testing graph hypothesis)")
print(f"  Required: MULTI-STOCK (30 stocks pooled, ~96,000 samples)")
print()

print(f"[Next Steps]")
print(f"  We need to implement multi-stock PyTorch GNNHAR to properly test:")
print(f"    1. Whether learned graph weights improve over fixed Pearson")
print(f"    2. Whether PyTorch residual design (H1+H2) beats sklearn approach")
print(f"    3. Whether GNNHAR nonlinear models beat linear GHAR")
print()

print(f"  However, given sklearn GHAR showed weak signal (+0.0006),")
print(f"  multi-stock PyTorch GNNHAR may also struggle to beat HAR OLS baseline.")
print()

print(f"{'='*70}\n")
