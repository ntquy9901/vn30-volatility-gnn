# GNNHAR1L Inference Guide for VIC and FPT Stocks

## Quick Start

Run inference on VIC and FPT stocks using trained GNNHAR1L models:

```bash
# Basic usage (GNNHAR1L, horizon 5, Jan-May 2026)
python gnn/gnnhar_paper/infer_vic_fpt.py

# Custom horizon
python gnn/gnnhar_paper/infer_vic_fpt.py --horizon 20

# Custom date range
python gnn/gnnhar_paper/infer_vic_fpt.py --start_date 2026-02-01 --end_date 2026-05-31

# Different model
python gnn/gnnhar_paper/infer_vic_fpt.py --model GHAR --horizon 10

# GPU acceleration
python gnn/gnnhar_paper/infer_vic_fpt.py --device cuda
```

## Available Models

The following trained models are available in `models/gnnhar_paper/`:

| Model | Description | Horizons Available |
|-------|-------------|-------------------|
| **HAR** | Baseline HAR model (no graph) | 1, 5, 10, 20 |
| **GHAR** | Linear spillover (1 GCN layer) | 1, 5, 10, 20 |
| **GNNHAR1L** | Nonlinear spillover (1 GCN + MLP) | 1, 5, 10, 20 |
| **GNNHAR2L** | 2-hop spillover (2 GCN + MLP) | 1, 5, 10, 20 |
| **GNNHAR3L** | 3-hop spillover (3 GCN + MLP) | 1, 5, 10, 20 |

## Inference Process

### Step 1: Data Loading
- Loads close prices for all VN30 stocks + VNINDEX
- Computes log returns and HAR features [rv_d, rv_w, rv_m]
- Filters data for specified date range

### Step 2: Feature Preparation
- Extracts VIC and FPT stock indices (VIC: node 28, FPT: node 5)
- Builds feature matrix: (n_dates, 31_nodes, 3_features)
- Computes target RV values for evaluation

### Step 3: Graph Construction
- Builds static adjacency matrix from correlations
- Uses pre-2026 data only (no lookahead bias)
- Default threshold: 0.3 (adjustable with `--corr_threshold`)

### Step 4: Model Loading
- Loads 5 trained models for specified horizon
- Each model trained with different random seed
- Ensemble average for robustness

### Step 5: Prediction & Evaluation
- Runs inference on all loaded models
- Averages predictions across ensemble
- Computes metrics: R², MAE, RMSE

## Output Files

Results are saved to `results/gnnhar_paper/`:

### 1. Summary CSV
Format: `inference_summary_h{HORIZON}_{MODEL}_{TIMESTAMP}.csv`

| Column | Description |
|--------|-------------|
| model | Model name used |
| horizon | Forecast horizon |
| stock | Stock ticker (VIC/FPT) |
| r2 | R² score (higher = better) |
| mae | Mean Absolute Error (lower = better) |
| rmse | Root Mean Squared Error (lower = better) |
| n_samples | Number of test samples |

### 2. Detailed CSV
Format: `inference_detailed_h{HORIZON}_{MODEL}_{TIMESTAMP}.csv`

| Column | Description |
|--------|-------------|
| model | Model name |
| horizon | Forecast horizon |
| stock | Stock ticker |
| date | Date of prediction |
| actual_rv | Actual realized volatility |
| predicted_rv | Predicted realized volatility |
| error | Prediction error (actual - predicted) |
| abs_error | Absolute prediction error |

### 3. Visualization PNG
Format: `{STOCK}_h{HORIZON}_{MODEL}_{TIMESTAMP}.png`

Two subplots:
- **Top**: Time series comparison (actual vs predicted RV over time)
- **Bottom**: Scatter plot (predicted vs actual, with perfect prediction line)

## Example Output

### Console Output
```
====================================================================
  GNNHAR Inference - VIC & FPT Stocks
  Model: GNNHAR1L, Horizon: h5
  Period: 2026-01-01 to 2026-05-31
  Device: cuda
====================================================================

[1] Loading data...
  Loaded 2834 dates x 31 stocks

[1] Preparing inference data...
  Inference period: 92 dates (2026-01-01 to 2026-05-31)
  VIC: index 28
  FPT: index 5

[1] Building adjacency matrix...
  Adjacency matrix: 31x31, density 0.156

[2] Loading GNNHAR1L models for h5...
  Found 5 trained models
  Loaded metadata: ['n_hid', 'horizon', 'model_name', 'train_dates']

[3] Running inference with 5 models...
  Ensemble prediction shape: (92, 31)

[4] Evaluating predictions...
  VIC: R²=0.7245, MAE=0.004548, RMSE=0.006774, n=92
  FPT: R²=0.6891, MAE=0.005120, RMSE=0.007231, n=92

[5] Generating plots...
  Saved: results/gnnhar_paper/VIC_h5_GNNHAR1L_20260603_143022.png
  Saved: results/gnnhar_paper/FPT_h5_GNNHAR1L_20260603_143023.png

  Saved summary: results/gnnhar_paper/inference_summary_h5_GNNHAR1L_20260603_143022.csv
  Saved details: results/gnnhar_paper/inference_detailed_h5_GNNHAR1L_20260603_143022.csv

====================================================================
  Inference complete!
  Results saved to: results/gnnhar_paper
====================================================================
```

### Summary CSV Example
```csv
model,horizon,stock,r2,mae,rmse,n_samples
GNNHAR1L,5,VIC,0.7245,0.004548,0.006774,92
GNNHAR1L,5,FPT,0.6891,0.005120,0.007231,92
```

### Detailed CSV Example
```csv
model,horizon,stock,date,actual_rv,predicted_rv,error,abs_error
GNNHAR1L,5,VIC,2026-01-01,0.012345,0.011987,0.000358,0.000358
GNNHAR1L,5,VIC,2026-01-02,0.015678,0.016234,-0.000556,0.000556
...
GNNHAR1L,5,FPT,2026-01-01,0.010987,0.011234,-0.000247,0.000247
...
```

## Understanding the Results

### R² Score Interpretation
- **R² = 1.0**: Perfect prediction
- **R² = 0.0**: No better than predicting mean
- **R² < 0.0**: Worse than predicting mean
- **Good performance**: R² > 0.7 for volatility forecasting

### MAE/RMSE Interpretation
- Values depend on RV scale (typically 0.005-0.015 for daily RV)
- Lower is better
- RMSE > MAE indicates some large errors
- RMSE ≈ MAE indicates uniform error distribution

## Advanced Usage

### Custom Stock Inference
Edit `infer_vic_fpt.py` to modify `TARGET_STOCKS`:

```python
# Inference for specific stocks
TARGET_STOCKS = ['VCB', 'MBB', 'VIC']  # Add any VN30 stocks
```

### Different Date Ranges
Test different periods:

```bash
# Pre-COVID period
python infer_vic_fpt.py --start_date 2018-01-01 --end_date 2019-12-31

# COVID period
python infer_vic_fpt.py --start_date 2020-01-01 --end_date 2021-12-31

# Recent period
python infer_vic_fpt.py --start_date 2024-01-01 --end_date 2024-12-31
```

### Model Comparison
Compare multiple models:

```bash
# Run all models for horizon 5
for model in HAR GHAR GNNHAR1L GNNHAR2L GNNHAR3L; do
    python infer_vic_fpt.py --model $model --horizon 5
done
```

## Troubleshooting

### Issue: "No trained models found"
**Solution**: Ensure models are trained first:
```bash
python gnn/gnnhar_paper/train_gnnhar_paper.py --horizon 5
```

### Issue: "Stock not found in data"
**Solution**: Stock must be in VN30_TICKERS. Check:
```python
from gnn.build_graph import VN30_TICKERS
print(VN30_TICKERS)  # List available stocks
```

### Issue: Low R² scores
**Possible causes**:
1. Test period includes high volatility (COVID, market crash)
2. Insufficient training data for period
3. Model needs retraining with recent data

### Issue: CUDA out of memory
**Solution**: Use CPU instead:
```bash
python infer_vic_fpt.py --device cpu
```

## Technical Details

### Input Features
- **rv_d**: Daily RV proxy (absolute return)
- **rv_w**: Weekly RV (5-day moving average)
- **rv_m**: Monthly RV (22-day moving average)

### Graph Construction
- **Nodes**: 31 (30 VN30 stocks + VNINDEX)
- **Edges**: Correlation-based (|corr| ≥ threshold)
- **Static**: Computed from pre-2026 data only

### Model Architecture (GNNHAR1L)
```
H1 = Linear(3, 1)           # Local HAR prediction
H2 = GCN(3, hidden)         # Graph spillover
H2 = ReLU(H2)
H2 = Linear(hidden, 1)      # Nonlinear transform
Output = H1 + H2            # Residual connection
```

### Ensemble Method
- 5 models trained with different random seeds
- All models used for prediction (no screening)
- Simple average of predictions

## Next Steps

1. **Run baseline comparison**: Compare GNNHAR1L vs HAR vs GHAR
2. **Analyze errors**: Look for patterns in prediction errors
3. **Time series analysis**: Examine performance over different sub-periods
4. **Hyperparameter tuning**: Adjust correlation threshold, model architecture
5. **Extended testing**: Test on more stocks and longer time periods

## Contact & Support

For issues or questions:
1. Check training logs: `gnn/gnnhar_paper/training_output.log`
2. Verify model files: `models/gnnhar_paper/h{HORIZON}/`
3. Review training documentation: `gnn/gnnhar_paper/docs/`