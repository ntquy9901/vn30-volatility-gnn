# EDA + PCA Results — VN30 Per-stock (h=1)

**Script:** `scripts/eda/eda_h1_per_stock.py`
**Run date:** 2026-05-17
**Data:** `data/raw/prices/` — 30 VN30 stocks
**Charts:** `results/eda_h1/`

---

## Config

```
H          = 1              # forecast horizon
LOOKBACK   = 20             # LSTM input window
VAL_RATIO  = 0.20           # 80/20 split on pre-test, per stock
TEST_START = 2026-01-01
```

**Split logic:** Với mỗi stock, lấy toàn bộ valid windows trước 2026-01-01,
chia 80% đầu = train, 20% cuối = val (chronological). Val_start_date khác nhau
theo từng stock tùy lịch sử niêm yết.

---

## 1. Sample Counts per Stock (80/20 split)

| Ticker | Start      | N_raw | Train | Val_start  | Val | Test | ESS_h1 | std%  | max_ret% |
|--------|------------|-------|-------|------------|-----|------|--------|-------|----------|
| VNM    | 2006-01-20 | 5,062 | 3,964 | 2022-01-12 | 991 |   86 |  3,964 | 1.796 |    10.12 |
| STB    | 2006-07-13 | 4,945 | 3,870 | 2022-02-21 | 968 |   86 |  3,870 | 2.299 |     6.77 |
| ACB    | 2006-11-22 | 4,850 | 3,794 | 2022-03-18 | 949 |   86 |  3,794 | 2.048 |    11.90 |
| FPT    | 2006-12-14 | 4,836 | 3,783 | 2022-03-23 | 946 |   86 |  3,783 | 1.938 |     6.79 |
| SSI    | 2006-12-18 | 4,824 | 3,773 | 2022-03-25 | 944 |   86 |  3,773 | 2.502 |     9.47 |
| VIC    | 2007-09-20 | 4,648 | 3,632 | 2022-05-18 | 909 |   86 |  3,632 | 2.176 |     6.77 |
| HPG    | 2007-11-16 | 4,607 | 3,600 | 2022-05-31 | 900 |   86 |  3,600 | 2.224 |     7.00 |
| SHB    | 2009-04-21 | 4,257 | 3,320 | 2022-09-08 | 830 |   86 |  3,320 | 2.548 |     9.72 |
| BVH    | 2009-06-26 | 4,214 | 3,285 | 2022-09-20 | 822 |   86 |  3,285 | 2.469 |     6.76 |
| VCB    | 2009-07-01 | 4,211 | 3,283 | 2022-09-21 | 821 |   86 |  3,283 | 1.884 |     6.81 |
| CTG    | 2009-07-17 | 4,199 | 3,273 | 2022-09-23 | 819 |   86 |  3,273 | 2.070 |     6.83 |
| MSN    | 2009-11-06 | 4,120 | 3,210 | 2022-10-17 | 803 |   86 |  3,210 | 2.153 |     6.76 |
| PDR    | 2010-08-02 | 3,938 | 3,064 | 2022-12-06 | 767 |   86 |  3,064 | 2.372 |     6.84 |
| MBB    | 2011-11-02 | 3,625 | 2,814 | 2023-03-13 | 704 |   86 |  2,814 | 1.799 |     6.84 |
| GAS    | 2012-05-22 | 3,490 | 2,706 | 2023-04-19 | 677 |   86 |  2,706 | 2.144 |     6.77 |
| VHM    | 2011-11-11 | 3,408 | 2,640 | 2023-05-16 | 661 |   86 |  2,640 | 3.227 |   130.40 |
| BID    | 2014-01-27 | 3,065 | 2,366 | 2023-08-21 | 592 |   86 |  2,366 | 2.181 |     6.79 |
| MWG    | 2014-07-15 | 2,955 | 2,278 | 2023-09-22 | 570 |   86 |  2,278 | 2.175 |     6.80 |
| SAB    | 2016-12-07 | 2,354 | 1,797 | 2024-03-18 | 450 |   86 |  1,797 | 1.800 |     6.76 |
| NVL    | 2016-12-29 | 2,338 | 1,784 | 2024-03-21 | 447 |   86 |  1,784 | 2.282 |     6.76 |
| VIB    | 2017-01-10 | 2,324 | 1,773 | 2024-03-26 | 444 |   86 |  1,773 | 2.113 |    10.28 |
| VJC    | 2017-03-01 | 2,300 | 1,754 | 2024-04-02 | 439 |   86 |  1,754 | 1.794 |     6.77 |
| PLX    | 2017-04-24 | 2,263 | 1,724 | 2024-04-11 | 432 |   86 |  1,724 | 2.165 |     6.76 |
| HDB    | 2018-01-08 | 2,082 | 1,580 | 2024-06-07 | 395 |   86 |  1,580 | 2.095 |     6.80 |
| BCM    | 2018-02-22 | 2,047 | 1,552 | 2024-06-18 | 388 |   86 |  1,552 | 2.648 |    14.73 |
| POW    | 2018-03-07 | 2,036 | 1,543 | 2024-06-20 | 386 |   86 |  1,543 | 2.278 |     9.73 |
| GVR    | 2018-03-22 | 2,028 | 1,536 | 2024-06-21 | 385 |   86 |  1,536 | 2.976 |    15.51 |
| TPB    | 2018-04-20 | 2,013 | 1,524 | 2024-06-26 | 382 |   86 |  1,524 | 2.065 |     6.77 |
| TCB    | 2018-06-05 | 1,984 | 1,501 | 2024-07-04 | 376 |   86 |  1,501 | 2.070 |     6.77 |
| SSB    | 2021-03-25 | 1,281 |   939 | 2025-01-22 | 235 |   86 |    939 | 1.663 |     6.76 |

**ESS_h1 = n_train** (h=1 không có label overlap → ESS = N_raw).

---

## 2. Key Findings — Sample Size

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| Min train (SSB)   | **939**   | ESS=939 >> 100, OK cho LSTM |
| Min val (SSB)     | **235**   | Ít nhưng đủ để early stopping |
| Test / stock      | **86**    | Cố định cho tất cả stocks |
| Stocks dưới 500 train | **0** | Tất cả 30 stocks đủ data |
| All ESS(h=1) >= 100 | **True** | Không drop stock nào |

**Val_start_date khác nhau theo stock:**
- Stocks niêm yết từ 2006 (VNM, STB...): val từ 2022 → val kéo dài ~4 năm (991 windows)
- Stocks niêm yết từ 2018 (TCB, HDB...): val từ 2024-06 → val ~1.5 năm (376-395 windows)
- SSB (niêm yết 2021): val từ 2025-01 → val chỉ ~1 năm (235 windows)

**Implication:** Early stopping và hyperparameter tuning trên val set khác nhau về độ dài — cần cẩn thận khi so sánh val loss cross-stock.

---

## 3. Return Distribution

| Metric | Giá trị | Ý nghĩa thực tế |
|--------|---------|----------------|
| Avg daily std | **~2.1%** | Ngày thường giá dao động ±2.1% — cao hơn S&P500 (~1%) |
| Max std (VHM) | **3.227%** | Biến động cao nhất trong VN30 |
| Min std (VJC) | **1.794%** | Biến động thấp nhất |
| VHM max_ret | **130.40%** | Dị thường — data error hoặc corporate action cần kiểm tra |
| GVR max_ret | **15.51%** | Ngày tăng mạnh nhất của GVR |
| Phần lớn stocks | max_ret ~6.8% | Biên độ ±7%/ngày của HOSE (circuit breaker) |

**VHM max_ret=130.40%:** Không phải return thực — rõ ràng là data error hoặc
điều chỉnh giá khi niêm yết lại. Cần winsorize hoặc loại ngày này trước khi train.

**Tại sao phần lớn max_ret ≈ 6.76–6.84%?** HOSE áp dụng biên độ dao động ±7%/ngày.
Hầu hết stocks chạm trần/sàn tối đa 1–2 lần trong lịch sử → max_ret phản ánh thực tế.

**Implication cho loss function:** VHM 130%, BCM 14.7%, GVR 15.5% sẽ dominate MSE.
Nên dùng **Huber loss** hoặc clip `|ret| > 3*std` trước khi train.

---

## 4. PCA Results

**Setup:** 369 common dates (intersection khi VCB train_end ≈ 2022-09), 30 stocks.

| PC | Variance Explained |
|----|-------------------|
| PC1 | **37.3%** |
| PC1..11 | **80%** (cần 11 PCs) |

**PC1 (37.3%) = Market Factor**
- Tất cả 30 stocks load cùng chiều (positive) lên PC1
- Thị trường lên/xuống kéo tất cả cùng chiều
- 37.3% = market factor vừa phải (S&P500 ~25%, VN retail market cao hơn)

**11 PCs để giải thích 80% variance:**
- Variance khá phân tán sau PC1
- Có nhiều idiosyncratic factors (sector, company-specific)
- GNN có thể học được cross-stock spillover sau khi control market factor

**Implication cho GNN:**
- PC1 cao → correlation heatmap dày (nhiều edges với threshold 0.4)
- Sector clusters rõ (Banks, BDS, Steel) → sector membership feature có giá trị
- Sau PC1, 11 factors còn lại → graph message passing cần ít nhất 2 layers để capture

---

## 5. Action Items

| # | Issue | Hành động |
|---|-------|-----------|
| A1 | VHM max_ret=130.4% | Kiểm tra ngày niêm yết VHM, winsorize nếu cần |
| A2 | BCM/GVR max_ret >14% | Clip |ret| > 3*std hoặc dùng Huber loss |
| A3 | Val_start khác nhau | Khi aggregate val metric, dùng macro-average (unweighted mean) |
| A4 | SSB val=235 windows | Monitor early stopping SSB riêng |

---

## 6. Charts

| File | Nội dung |
|------|----------|
| `results/eda_h1/01_data_coverage.png` | N_raw bars + stacked train/val/test |
| `results/eda_h1/02_ess_comparison.png` | ESS h=1 vs h=20 (log scale) |
| `results/eda_h1/03_return_distributions.png` | Histogram+KDE vs Normal, 30 stocks |
| `results/eda_h1/04_rolling_vol_timeline.png` | Rolling vol + train/val/test shading |
| `results/eda_h1/05_correlation_heatmap.png` | Pearson heatmap train period |
| `results/eda_h1/06_pca_analysis.png` | Scree + PC1/PC2 loadings |
| `results/eda_h1/07_qq_plots.png` | QQ plots 12 stocks |

---

*Generated: 2026-05-17 | Script: `scripts/eda/eda_h1_per_stock.py`*
