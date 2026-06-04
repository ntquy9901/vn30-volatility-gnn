# Learning Notes: Sample Size Requirements for LSTM & GNN

**Nguồn:** Research session VN30 volatility — 2026-05-17
**Câu hỏi gốc:** Căn cứ nào để biết có đủ dữ liệu để train LSTM/GNN?

---

## Kết luận trước

> **Không có paper nào đưa ra threshold cố định** như ">=500 samples là đủ."
> Cách đúng là tính **Effective Sample Size (ESS)** và so với số tham số của model.

---

## 1. Effective Sample Size (ESS) — Khái niệm cốt lõi

**Nguồn:** López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 7

Khi dùng stride=1 và horizon=h, các labels chồng lấp nhau:

```
Window t=20:  label = std(ret[20:40])   # dùng ngày 20-39
Window t=21:  label = std(ret[21:41])   # dùng ngày 21-40 (19/20 ngày trùng!)
Overlap = (h-1)/h = 19/20 = 95%
```

**Công thức ESS:**
```
ESS = N_raw / h
```

**Ví dụ VN30 LSTM (h=20, stride=1):**
```
N_raw = 2,458 windows/stock
ESS   = 2,458 / 20 = 123 independent observations per stock
```

=> Bạn có 2,458 raw samples nhưng chỉ ~**123 thực sự độc lập**.

---

## 2. Tại sao HAR-RV thắng với dữ liệu nhỏ — Gauss-Markov

**HAR-RV (OLS)** có 3 tham số. Với ESS=123:
```
ratio = 123 / 3 = 41 obs/param   -> BLUE (Best Linear Unbiased Estimator)
```

**LSTM nhỏ nhất** (hidden=32) có ~4,000 tham số. Với ESS=123:
```
ratio = 123 / 4,000 = 0.03 obs/param  -> severe overfitting là tất yếu
```

**Kết luận từ Gauss-Markov theorem:** Với ít tham số và đủ ESS, OLS là estimator tốt nhất có thể. Neural network không thể cạnh tranh ở regime này.

---

## 3. Bằng chứng từ Paper

### Paper 1 — Branco, Rubesam & Zevallos (JEF 2024)
> *"Forecasting Realized Volatility: Does Anything Beat Linear Models?"*
> Journal of Empirical Finance, Vol. 78, 2024

- **Dataset:** 10 global stock market indices, Jan 2000 - Dec 2021 (~5,500 ngày/index)
- **Kết quả:** "No evidence that nonlinear ML models can statistically outperform linear models in general"
- **Implicit finding:** Ngay cả với ~5,500 ngày (ESS ~275/index cho h=20), linear vẫn thắng
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0927539824000598

### Paper 2 — Gu, Kelly & Xiu (RFS 2020)
> *"Empirical Asset Pricing via Machine Learning"*
> Review of Financial Studies, 2020

- **Dataset:** ~30,000 US stocks × 60 năm → hàng triệu obs
- **Kết quả:** ML (neural network, gradient boosting) thắng linear ở panel data lớn
- **Key insight:** ML chỉ thắng khi có **panel data cực lớn**, không thể áp dụng per-stock
- **Link:** https://academic.oup.com/rfs/article/33/5/2223/5758276

### Paper 3 — Advances in RV Forecasting Review (Springer 2025)
> *"Advances in Forecasting Realized Volatility"*
> Financial Innovation, Springer, 2025

- **Kết quả:** "Neural models need large datasets (thousands of series/samples) to outperform simple HAR consistently"
- **Link:** https://link.springer.com/article/10.1186/s40854-025-00809-5

---

## 4. Bảng ESS cho VN30 theo từng Model

| Model | Stride | H | N_raw/stock | ESS/stock | ESS tổng (x30) |
|---|---|---|---|---|---|
| HAR-RV (OLS) | 1 | 20 | ~2,500 | ~125 | — (fit riêng) |
| LSTM per-stock | 1 | 20 | 2,458 | **123** | — (fit riêng) |
| LSTM cross-stock | 1 | 20 | 2,458 | 123 | **3,690** |
| MLP batch | 20 | 20 | ~97 snaps | ~97 | 2,910 |
| GNN batch | 20 | 20 | ~97 snaps | ~97 | 2,910 (all 30) |

**Cross-stock LSTM** (1 model cho 30 stocks, pooled training) là cách duy nhất để
tăng ESS mà không cần thêm dữ liệu lịch sử.

---

## 5. Khi nào LSTM có thể beat HAR?

Từ tổng hợp literature (không có threshold cố định, nhưng có pattern):

| Điều kiện | Feasibility |
|---|---|
| Per-stock LSTM, h=20, ~123 ESS | Thất bại — quá ít data |
| Cross-stock pool, 30 stocks, 3,690 ESS | Marginal — có thể tiếp cận HAR |
| Panel data toàn thị trường (>10,000 stocks) | LSTM thắng (Gu et al. 2020) |
| Foundation model fine-tuned (TimesFM) | Có thể thắng HAR (Goel et al. 2025) |
| h=1 thay vì h=20 (ESS=N_raw) | ESS cao hơn 20 lần -> LSTM feasible hơn |

---

## 6. Cách lập luận trong Thesis

```
Bước 1: Tính ESS
  ESS = N_raw / h = 2,458 / 20 ≈ 123 independent obs/stock

Bước 2: So sánh obs/param ratio
  HAR-RV:  123 / 3    = 41.0  -> BLUE theo Gauss-Markov
  LSTM:    123 / 4000 = 0.03  -> severe overfitting regime

Bước 3: Cite Branco et al. 2024
  "No evidence that nonlinear ML can statistically outperform
   linear models" — even with 10 indices x 22 years

Bước 4: Giải thích strategy của mình
  - Per-stock LSTM là data-limited theo định nghĩa (ESS=123)
  - Cross-stock pooling tang ESS 30x -> 3,690 -> feasible hơn
  - HAR-RS (semivariance) là most practical improvement (same data, 1-4% gain)
```

---

## 7. Chiến lược tăng dữ liệu (Data Augmentation)

| Option | Cách làm | ESS tăng thêm |
|---|---|---|
| Cross-stock pooling | Train 1 model cho 30 stocks cùng lúc | x30 |
| Shorter horizon | h=1 thay h=20 | x20 |
| OHLCV features | Thêm O,H,L,V vào input | Không tăng ESS, tăng signal |
| Rolling RV features | Thêm RV5, RV10, RV20 | Không tăng ESS, tăng signal |
| GARCH simulation | Sinh thêm synthetic paths | Tăng N_raw (quality uncertain) |

---

## 8. Threshold được dùng trong data_quality_check.py

```python
# src/data_quality_check.py
MIN_ESS   = 100   # minimum effective independent samples (Lopez de Prado 2018)
MIN_TRAIN = MIN_ESS * MAX_H  # = 100 * 20 = 2000 raw samples minimum
```

**Giải thích:** MIN_ESS=100 là threshold tối thiểu chấp nhận được để train bất kỳ model nào.
Con số này không có trong paper cụ thể — đây là engineering judgment dựa trên:
- ESS < 30: không đủ cho bất kỳ nonlinear model nào
- ESS 30-100: chỉ đủ cho models rất ít params (HAR, Ridge)
- ESS 100-500: feasible cho LSTM nhỏ (hidden <= 16-32) với regularization mạnh
- ESS > 500: LSTM bắt đầu ổn định

---

*Last updated: 2026-05-17*
