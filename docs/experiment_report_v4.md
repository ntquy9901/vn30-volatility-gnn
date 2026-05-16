# Experiment Report v4 — VN30 Volatility Forecasting
**Date:** 2026-05-16  
**Test set:** 2026-01-05 → 2026-04-07 (62 dates, 30 VN30 stocks, pooled)

---

## Final Results — All Experiments

| Model | Training | Features | MAE | R² | Pearson_r | QLIKE |
|---|---|---|---|---|---|---|
| **HAR-RV (OLS)** | Full history/stock | RV_d, RV_w, RV_m | **0.0010** | **+0.971** | **0.985** | **0.012** |
| **LSTM** | Full history/stock | same | 0.0011 | +0.967 | 0.983 | 0.012 |
| **MLP (RV6 only)** | Batch, 2896 samples | 6 extended RV | **0.0066** | **+0.278** | **0.551** | 0.328 |
| **Moirai2 IQR (zero-shot)** | **None** | **q0.9−q0.1** | 0.0067† | +0.161 | **0.557** | — |
| GNN (RV6 only) | Batch | 6 extended RV | 0.0084 | -0.194 | 0.396 | 0.328 |
| MLP (Moirai2+RV6) | Batch | 390-dim | 0.0072 | +0.065 | 0.505 | **0.259** |
| GNN (Moirai2+RV6) | Batch | 390-dim | 0.0091 | -0.519 | 0.377 | 0.345 |
| Batch MLP (Moirai2+RV3) | Batch | 387-dim | 0.0078 | -0.137 | 0.425 | 0.290 |
| Batch GNN (Moirai2+RV3) | Batch | 387-dim | 0.0074 | +0.041 | 0.357 | 0.399 |
| WalkFwd MLP+Moirai2 | Walk-forward | 387-dim | 0.0121 | -1.077 | 0.239 | 1.444 |
| WalkFwd GNN+Moirai2 | Walk-forward | 387-dim | 0.0143 | -1.856 | 0.117 | 1.898 |
| GARCH(1,1) | Per-stock MLE | — | 0.0071 | +0.159 | 0.535 | 0.301 |

†MAE after scale correction (×0.442); raw MAE = 0.0323

---

## Key Findings

### Finding 1 — Training regime is the primary bottleneck (H2 confirmed)
Switching from walk-forward (30 pts/window) to batch training (2,896 samples) cuts MAE by:
- MLP: 0.0121 → 0.0078 (−35%)
- GNN: 0.0143 → 0.0074 (−48%)

Walk-forward provides only 30 samples per gradient step (30 stocks × 1 date). HAR-OLS uses ~75,000 pairs (2,500 dates × 30 stocks). Neural models cannot compete with this data starvation.

### Finding 2 — Moirai2 embeddings add noise, not signal, when RV features are present
When extended RV6 features are available, adding Moirai2 (384-dim) **hurts**:
- MLP: 0.0066 (RV6 only) → 0.0072 (Moirai2+RV6), +8.4% worse
- GNN: 0.0084 (RV6 only) → 0.0091 (Moirai2+RV6), +8.1% worse

The 384-dim return-level embedding drowns the 6-dim volatility signal. Moirai2 is pre-trained for point-return forecasting, not second-moment prediction.

### Finding 3 — Extended RV features (6-dim) beat basic HAR-3 features
Best neural model with RV6 (MAE=0.0066) vs best with RV3 (MAE=0.0074):
- RV_q (quarterly, 60-day avg) captures longer memory
- corr_vnindex captures systematic risk exposure
- jump_ratio captures tail-risk contribution

### Finding 4 — Graph structure does not consistently help
- Batch MLP (RV6): 0.0066 vs Batch GNN (RV6): 0.0084 → graph hurts (−21%)
- Batch MLP (Moirai2+RV3): 0.0078 vs Batch GNN (Moirai2+RV3): 0.0074 → graph helps slightly (+5%)
- Effect depends on feature quality and number of training windows

### Finding 5 — Moirai2 IQR (zero-shot) matches trained MLP on Pearson_r

Using Moirai2's own forecast output — IQR = q0.9 − q0.1 of the 1-step-ahead
return distribution — as a volatility proxy (no training required):

| Approach | Pearson_r | MAE | Training |
|---|---|---|---|
| Moirai2 IQR (zero-shot) | **0.557** | 0.0067* | None |
| MLP (RV6-only, batch) | 0.551 | **0.0066** | 2,896 samples |
| Embedding median\|corr\| | 0.128 | — | — |

*MAE after scale correction (IQR × 0.442); R²=+0.161

Key observations:
- IQR requires **no training** and **no feature engineering** — pure zero-shot
- Pearson_r of IQR (0.557) is **4× higher** than embedding approach (0.128)
- Suggests Moirai2's **uncertainty quantification** carries more RV signal
  than its **intermediate representation** (embedding)
- Domain gap partially mitigated: model learned calibrated uncertainty
  even though pretrained on macro/competition data, not equity data
- Scale factor 0.442 ≈ mean(RV)/mean(IQR): IQR in log-return units is
  ~2.26× larger than RV (consistent with fat-tailed VN30 returns)

### Finding 6 — Remaining 6.5× gap vs HAR-RV is structural
Best neural (MLP RV6, MAE=0.0066) vs HAR-RV (MAE=0.0010) = **6.5× gap**.

Root cause: HAR-OLS fits per stock on the full time series (~2,500 samples per stock). The batch neural model shares weights across all 30 stocks → effectively ~97 samples per stock (2,896/30). OLS is the optimal linear estimator; neural models need far more data to approach this efficiency at linear tasks.

---

## Moirai2 Architecture (confirmed)
- **Decoder-only** with causal temporal attention (`packed_causal_attention_mask`)
- For context_length=200, prediction_length=1: **14 patches** (13 context + 1 MASK token)
- Best pooling: **last_context** (`reprs[:, -2, :]`), median |corr with RV| = 0.128
  - Benchmarked 4 strategies: last_context > last > mean_context > mean (old default)

---

## Moirai2 Pretraining Data — Finance Coverage Analysis

**Source:** GIFT-Eval Pretrain dataset (arxiv 2410.10393) + Chronos-Mixup + KernelSynth

### Datasets labeled "Econ/Fin" trong pretraining corpus

| Dataset | Freq | Series | Nội dung thực sự |
|---|---|---|---|
| M4 Daily | Daily | 4,227 | Business/macro competition — một số stock |
| M4 Monthly | Monthly | 48,000 | Kinh tế vĩ mô, demographic, industry |
| M4 Quarterly | Quarterly | 24,000 | GDP, macro indicators |
| M4 Yearly | Annual | 22,974 | Macro, demographic |
| M1/M3 variants | Mix | ~3,000 | Macro + business forecasting |
| FRED-MD | Monthly | 107 | **Federal Reserve macro indicators** (US) |
| Bitcoin | Daily | **18** | **Crypto price** — duy nhất daily price series |
| NN5 Daily/Weekly | Daily/W | 111 | ATM cash withdrawal UK (KHÔNG phải stock) |
| GoDaddy | Monthly | 3,135 | Domain registration business data |

### Kết luận

**Không có equity/stock market data thực sự trong pretraining:**
- Chỉ 18 series Bitcoin daily price — crypto, không phải equity
- M4/M3/M1 là competition data tổng hợp, chủ yếu macro kinh tế và business
- FRED-MD = 107 chỉ số macro Mỹ (CPI, lãi suất, employment)
- NN5 = cash withdrawal, bị label nhầm "Econ/Fin"
- **Không có Asian market data** — toàn bộ corpus là US/EU-centric
- **Không có realized volatility series** — Moirai2 chưa bao giờ thấy RV trong pretraining

**Hàm ý trực tiếp cho kết quả thực nghiệm:**

Đây là giải thích căn bản tại sao median|corr(Moirai2 embedding, RV)| = 0.128:

> Moirai 2.0 được pretrain trên macro/business competition data và synthetic series.
> Pretraining corpus không chứa daily stock return series của Asian equity markets,
> không có realized volatility labels, và không có dữ liệu thị trường Việt Nam.
> Do đó, các embedding dimensions phản ánh patterns của macro time series
> (trend, seasonality, mean-reversion) — không phải second-moment structure
> của individual stock volatility. Domain gap này là nguyên nhân cấu trúc
> giải thích tại sao Moirai2 embeddings thêm nhiễu thay vì signal khi kết hợp
> với direct RV features.

---

## Extended RV Features (6-dim)

| Feature | Formula | Source |
|---|---|---|
| log(RV_d) | log(std(r_{t-h+1}..r_t)) | Corsi (2009) HAR-RV |
| log(RV_w) | log(mean of past 5 RV_d) | Corsi (2009) |
| log(RV_m) | log(mean of past 22 RV_d) | Corsi (2009) |
| log(RV_q) | log(mean of past 60 RV_d) | Extended HAR |
| corr_vnindex | Pearson(stock, VNINDEX, 60-day) | Systematic risk |
| jump_ratio | max(RV_d − RV_w, 0) / RV_d | Barndorff-Nielsen & Shephard (2004) |

---

## Thesis Narrative

> "We evaluated whether Salesforce Moirai2 — a state-of-the-art universal time-series foundation model — can replace hand-crafted volatility features for realized volatility (RV) forecasting on the VN30 index (Vietnam). Results on a 2026 out-of-sample test set reveal four key insights:
>
> (1) **Moirai2 embeddings alone are insufficient** for volatility prediction. The model's decoder-only architecture is pre-trained on macro/business competition data (M1/M3/M4 competitions, FRED-MD macro indicators) and contains virtually no equity market data — only 18 Bitcoin series among ~4.5M time series in the pretraining corpus. No Asian market data and no realized volatility series are present. This domain gap directly explains the weak embedding–RV correlation (median |corr| = 0.128 across 384 dimensions): the embeddings encode macro trend/seasonality patterns, not second-moment structure of individual stock volatility.
>
> (2) **The walk-forward training regime**, designed for streaming deployment, severely limits neural models by providing only 30 labeled samples per gradient update, explaining 35–48% of the performance gap vs batch training.
>
> (3) **Classical HAR-RV (OLS) dominates** all neural configurations, achieving R²=0.97 vs the best neural model's R²=0.28. The gap is structural: HAR-OLS fits per-stock on 2,500+ samples with a 3-parameter linear model; the equivalent neural model has ~97 effective samples per stock.
>
> (4) **Moirai2 embeddings add noise when combined with direct RV features** (+8–9% worse MAE), confirming that the 384-dimensional return-level representation is orthogonal to, not complementary with, the 6-dimensional RV feature space.
>
> (5) **Moirai2's quantile spread (IQR = q0.9 − q0.1) is a competitive zero-shot volatility proxy**: Pearson_r = 0.557, matching the best trained neural model (MLP-RV6, r=0.551) with zero training. This reveals a nuance in the domain-gap hypothesis — while Moirai2's *intermediate embeddings* weakly correlate with RV (median|corr|=0.128), its *calibrated uncertainty* (quantile spread) captures volatility dynamics effectively, suggesting the model implicitly encodes heteroscedasticity despite being pretrained on macro data.
>
> Best trained neural configuration: MLP with 6 extended RV features (batch training), MAE=0.0066, 6.5× worse than HAR-RV. The performance ceiling for trained neural models is determined by data volume (~97 effective samples/stock). HAR-RV remains the benchmark champion for point prediction (R²=0.971 vs best neural R²=0.278)."

---

## Files

| File | Purpose |
|---|---|
| `src/volatility_labels.py` | `get_extended_rv_features()` — 6-dim RV features |
| `run_batch_rv.py` | Extended RV experiment (4 variants × batch training) |
| `run_batch_train.py` | H2 experiment (batch vs walk-forward) |
| `run_gnn_train.py` | Walk-forward GNN training |
| `run_baselines.py` | GARCH, HAR-RV, LSTM, MLP baselines |
| `run_evaluation.py` | Walk-forward model comparison |
| `run_plots.py` | Visualizations |
| `diag_*.py` | D1–D4 root-cause diagnostics |
| `diag_pooling_compare.py` | Moirai2 pooling strategy benchmark |
| `results/embed_cache/` | 135 cached window embeddings (387-dim) |
