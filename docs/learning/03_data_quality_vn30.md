# Learning Notes: Data Quality Analysis — VN30

**Nguồn:** Data quality check script — 2026-05-17
**Script:** src/data_quality_check.py
**Charts:** results/data_quality/ (10 charts)

---

## Kết quả chạy (2026-05-17)

```
Stocks:      30/30 loaded
Date range:  2013-08-21 -> 2026-05-08
Avg days:    2,609  (min=1,277, max=2,964)
```

---

## LSTM Feasibility

- Stride=1, lookback=20, max_h=20
- **30/30 stocks** vuot nguong MIN_TRAIN=2,000 raw samples
- Avg raw samples/stock: ~2,567
- **ESS/stock = 2,567 / 20 = 128** independent observations

**Verdict: LSTM feasible nhung o ranh gioi data-limited.**

---

## GNN Feasibility

- Stride=20, first snapshot tai t=200 (Moirai2 window)
- **30/30 stocks** co >= 50 snapshots
- Avg snapshots: ~120 per stock (= 120 graph snapshots)

**Verdict: GNN feasible.**

---

## Statistical Quality

| Metric | Ket qua |
|---|---|
| Stationary (ADF p<0.05) | 30/30 |
| Fat tails (kurt > 3) | 12/30 |
| Avg excess kurtosis | 26.15 (!!) |
| Avg annualized vol | 34% |

**Note:** Excess kurtosis trung binh 26.15 la rat cao (S&P500 ~ 3-5).
Cho thay VN30 co nhieu ngay bat thuong (trading halts, circuit breakers).
Day la ly do RV forecasting kho hon tren thi truong VN so voi thi truong phat trien.

---

## GNN Graph Dynamics

- Pearson correlation tinh tren cua so 60 ngay
- Graph density (corr > 0.4): thay doi theo thoi gian
- Trong COVID 2020: correlation tang dot bien (all stocks move together)
- Ngoai khung hoang: correlation thap hon, graph sparse hon

**Y nghia cho GNN:** Graph structure khong on dinh — nen update graph
theo tung snapshot (monthly) thay vi dung 1 static graph.

---

## Chien Luoc Tang Du Lieu (Data Augmentation)

| Option | ESS tang | Do phuc tap | Recommend |
|---|---|---|---|
| Cross-stock pooling (30 stocks) | x30 | Thap | Yes — uu tien 1 |
| Shorter horizon h=1 | x20 | Thap | Yes — uu tien 2 |
| OHLCV features (O,H,L,V) | Khong | Thap | Yes — them signal |
| Rolling RV features | Khong | Thap | Yes — them signal |
| GARCH simulation | ~x2-5 | Trung binh | Can xem xet |
| Foundation model (TimesFM) | N/A | Cao | Cho Phase 2 |

---

## Charts da tao

| File | Noi dung |
|---|---|
| 01_timeline.png | Availability per stock, train/val/test split |
| 02_sample_counts.png | LSTM samples, GNN snapshots per stock |
| 03_missing_values.png | NaN va zero-volume per stock |
| 04_return_distributions.png | Histogram log-returns vs normal |
| 05_rolling_volatility.png | Rolling RV20 theo thoi gian |
| 06_correlation_heatmap.png | Pearson correlation matrix (sector sorted) |
| 07_pca_analysis.png | Scree plot + PC1/PC2 loadings + PC1 time series |
| 08_statistical_summary.png | Mean/vol/skew/kurt/ADF per stock |
| 09_gnn_graph_dynamics.png | Graph density + avg correlation over time |
| 10_feasibility_summary.png | Pass/fail per stock per criterion |

---

*Last updated: 2026-05-17*
