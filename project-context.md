# Project Context — Moirai2 + GNN Volatility Prediction (VN30)

---

## External Resources (Official)

### Moirai 2.0-R-Small — HuggingFace Model Card
**URL:** https://huggingface.co/Salesforce/moirai-2.0-R-small  
**License:** CC-BY-NC-4.0 (non-commercial, research only)  
**Parameters:** 11.4M (Small), 87.1M (Base), 305M (Large) — Small is best on GIFT-Eval  
**Downloads:** ~623K/month

**Architecture (confirmed decoder-only):**
- Decoder-only autoregressive transformer với causal self-attention
- Input: patches của time series + binary missing-value indicator
  - Patch embedding: `z_i = SiLU(W(x̂ᵢ) + b) + x̂ᵢ` (residual block, 2p → d dims)
  - Instance normalization từ **30% đầu** của series (tránh lookahead trong causal arch)
- Output: 9 quantile predictions (q=0.1 → 0.9) per time step
- Loss: Pinball/quantile loss (không phải MSE, không phải NLL)
- Multi-token prediction: predict n_token patches đồng thời per output step
- 50% patch-level random masking for inference robustness
- KV cache: 4× speedup cho repeated queries, 17× cho long horizons
- Training: 100K steps, batch=256, AdamW lr=1e-3, bf16

**Key behavioral note for this project:**
Moirai 2.0 học để dự báo **phân phối giá trị return tương lai**, không phải volatility.
Embeddings ở intermediate layers phản ánh "thông tin cần để dự báo return t+1",
không phải second-moment (variance) structure. Điều này giải thích tại sao
median|corr(embedding_dim, RV)| = 0.128 — signal có nhưng rất diluted.

**IQR Quantile Spread — Volatility Proxy chưa khai thác:**
Vì output = 9 quantiles của return, spread `IQR = q90 - q10` tự nhiên đo
uncertainty của model ~ implicit volatility estimate. Chưa được thử trong experiments.

**Usage:**
```bash
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts && pip install -e '.[notebook]'
# Example: github.com/SalesforceAIResearch/uni2ts/blob/main/example/moirai_forecast.ipynb
```

**Citation:**
```bibtex
@article{liu2025moirai,
  title={Moirai 2.0: When less is more for time series forecasting},
  author={Liu, Chenghao and Aksu, Taha and Liu, Juncheng and Liu, Xu and
          Yan, Hanshu and Pham, Quang and Savarese, Silvio and
          Sahoo, Doyen and Xiong, Caiming and Li, Junnan},
  journal={arXiv preprint arXiv:2511.11698},
  year={2025}
}
```

---

### GIFT-Eval Pretrain Dataset — HuggingFace
**URL:** https://huggingface.co/datasets/Salesforce/GiftEvalPretrain  
**License:** Apache 2.0  
**Đây là pretraining data của Moirai 2.0**

**Quy mô:**
- 4.5 triệu univariate và multivariate time series
- 230 tỷ data points
- 975 GB
- 13 tần số (frequencies) khác nhau
- 7 domain khác nhau
- 71 univariate datasets + 17 multivariate datasets

**Format (Parquet):**
```
item_id   | start                 | freq | target        | past_feat_dynamic_real
"0"       | 2016-02-29T05:00:00   | "30T"| [55.0, 123.0] | [[0.032, 0.032, ...]]
```

**Mục đích:** Pretrain foundation models cho zero-shot time series forecasting mà
không bị data leakage với GIFT-Eval benchmark.

**Models trained on this data (18+):** Amazon Chronos-2, AutoGluon Chronos-2,
ByteDance Timer-S1, Datadog Toto-Open-Base-1.0, NX-AI TiRex, v.v.

**Lưu ý quan trọng:** KHÔNG có Vietnamese stock data, KHÔNG có volatility series.
Moirai 2.0 chưa bao giờ thấy VN30 hay RV prediction tasks trong pretraining.

---

### GIFT-Eval Benchmark Paper
**URL:** https://arxiv.org/abs/2410.10393  
**Title:** "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation"  
**Authors:** Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu,
             Silvio Savarese, Caiming Xiong, Doyen Sahoo  
**Note:** arxiv 2410.10393 là GIFT-Eval benchmark paper, KHÔNG phải Moirai 2.0 paper.
          Moirai 2.0 paper = arxiv 2511.11698.

**GIFT-Eval Benchmark:**
- 23 datasets, 144,000+ time series, 177 triệu data points
- 7 domains, 10 frequencies, nhiều prediction horizons
- Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval
- Moirai 2.0 Small xếp hạng #5 MASE (0.728), #6 CRPS (0.516) trong 30 models

**Moirai 2.0 Ablation kết quả (từ paper 2511.11698):**
- Từ Moirai 1.0-small baseline (MASE=0.946):
  - +Decoder-only arch → 0.929
  - +New 36M-series corpus → 0.850  ← đóng góp lớn nhất
  - +Quantile loss → 0.744
  - +Recursive decoding → 0.736
  - +Multi-token prediction + random masking → 0.728 (final)
- Kết luận: **data quality >> architecture change** trong Moirai 2.0

---

## Project Goal
Luận văn tốt nghiệp: dự đoán **realized volatility** của 30 cổ phiếu VN30
bằng cách kết hợp Moirai2 foundation model (feature extraction) với Graph
Neural Network (graph aggregation). Contribution chính: sử dụng spillover
effect qua graph structure để cải thiện dự báo so với mô hình per-stock.

---

## Data

**Source:** 30 cổ phiếu VN30 — HOSE (Sàn HCMC)
**Format:** OHLCV daily
```
date        open    high    low     close   volume
6/18/2014   2.35    2.35    2.34    2.35    139572
```
**Range:** ~2014–2024, ~2500 trading days per stock
**Trading days/year:** ~240–252 (HOSE), ~20/tháng

**VN30 Sector Labels (hardcoded prior knowledge):**
```
Ngân hàng:    VCB, BID, CTG, MBB, TCB, HDB, VPB, STB, ACB, SSB
Bất động sản: VHM, VRE, NVL, PDR, KDH
Thép/CN:      HPG, HSG, NKG
Bán lẻ/Khác:  MWG, PNJ, FPT, ...
```

---

## Architecture Pipeline

```
30 stocks OHLCV
   ↓ close price → log-return = log(P_t / P_{t-1})
   ↓ rolling window 200 ngày, stride 20 ngày

Moirai2-small (FROZEN — weights locked, no fine-tuning)
  - target_dim=1 (log-return only)
  - past_feat_dynamic_real_dim=0
  - context_length=200, patch_size=16 (thực tế từ pretrained config.json)
  - 14 patches: 13 context patches + 1 MASK token (prediction_length=1)
  - DECODER-ONLY với causal attention (không phải BERT-style encoder)
  - Hook trên transformer backbone → reprs shape: (30, 14, 384)
  - Pooling: last_context = reprs[:, -2, :] → (30, 384)
    (index -2 = last context patch trước MASK token, median|corr với RV|=0.128)
    Benchmark: last_context > last > mean_context > mean (old default)

Multi-relational Graph (updated monthly)
  - Edge exists if: Pearson_corr > 0.4 OR same_sector = True
  - edge_attr = [pearson_corr, same_sector_flag]  shape: (E, 2)
  - edge_index shape: (2, E)
  - Pearson computed on 60-day rolling log-returns

2-layer GraphSAGE (TRAINABLE)
  - SAGEConv(384 → 64) → ReLU → Dropout(0.3)
  - SAGEConv(64 → 32)  → ReLU → Dropout(0.3)

MLP Prediction Head (TRAINABLE)
  - Linear(32 → 16) → ReLU → Linear(16 → 1)
  - Output: predicted realized volatility (scalar per stock)
```

---

## Label Definition — Realized Volatility

**Formula:**
```
RV_t(h) = std( log-returns from t+1 to t+h )
         = sqrt( sum((r_i - r_mean)^2) / (h-1) )   [sample std, ddof=1]
```

**In code:**
```python
log_returns = np.log(df_close / df_close.shift(1))
realized_vol = log_returns.shift(-1).rolling(h).std()
# Label at time t = std of next h log-returns
```

**Horizon h=20** (≈ 1 tháng giao dịch HOSE)

**Academic sources (cite these in thesis):**
- Andersen, Bollerslev, Diebold & Labys (2003) — *Econometrica* — định nghĩa RV chuẩn
- Corsi (2009) — *Journal of Financial Econometrics* — HAR-RV model, dùng h=5,22,66
- Barndorff-Nielsen & Shephard (2002) — *J. Royal Statistical Society*

**Ý nghĩa kinh tế:**
- RV = mức biến động giá thực tế trong h ngày tới
- RV cao → rủi ro lớn → nhà đầu tư cần giảm vị thế, nới stop-loss
- RV thấp → thị trường ổn → có thể tăng vị thế

**Ví dụ VCB (5 ngày):**
```
Returns: [+1.41%, −1.63%, +0.82%, +1.86%, −0.92%]
mean = +0.308%
RV(5) = std(returns, ddof=1) = 1.512%/ngày
→ VCB biến động ±1.5%/ngày trong tuần tới
→ Annualized: 1.512% × sqrt(252) ≈ 24%/năm
```

---

## Key Architectural Decisions & Rationale

### 0. VNINDEX as Virtual Hub Node (approved 2026-05-15)
- **Decision:** Thêm VNINDEX làm node thứ 31 trong graph (virtual hub node)
- **Data:** `VNINDEX_ohlcv.csv` — 2957 ngày, cùng date range với VN30
- **Topology:** VNINDEX → tất cả 30 VN30 stocks (hub edges), không có chiều ngược lại
- **Edge attr:** `[corr(VNINDEX, stock_i), 0]` — same_sector=0 vì VNINDEX không thuộc sector
- **Training:** VNINDEX là context-only node, mask khỏi loss (`loss_mask[0] = False`)
- **Rationale:** VNINDEX encode systematic risk + market regime — thông tin mà per-stock embeddings không capture được. Hub node pattern (virtual node / supernode) có precedent: Gilmer et al. (2017) NIPS, Zhao et al. (2021)
- **COVID insight:** VNINDEX embedding sẽ bắt crash signal sớm nhất → propagate qua hub edges → tất cả 30 stocks nhận cảnh báo trong cùng 1 GNN layer

```python
# Graph: 31 nodes (node_0=VNINDEX, node_1..30=VN30)
# Hub edges: VNINDEX → all stocks
vnindex_edges = [[0]*30, list(range(1, 31))]
vnindex_attr  = [[corr(VNINDEX, stock_i), 0] for i in range(30)]
# Stock-stock edges: Pearson>0.4 OR same_sector (unchanged)
# loss_mask[0] = False  ← VNINDEX không có prediction label
```

### 1. Moirai2 FROZEN (không fine-tune)
- Moirai2-small: **11.4M params** (không phải ~70M — đó là Moirai 1.0-Large)
  - Moirai 2.0 Small=11.4M, Base=87.1M, Large=305M
  - Small là best trên GIFT-Eval (counterintuitive — scaling hurts)
- VN30 data: 30 × 2500 = 75,000 points → tỷ lệ 0.15 params/point → có thể fine-tune head
- Moirai 2.0 pretrained trên 295B observations (36M series)
- Chỉ GNN + MLP head (~50K params) được train trên VN30

**Moirai 1.0 vs 2.0 — so sánh kiến trúc:**
| | Moirai 1.0 | Moirai 2.0 |
|---|---|---|
| Architecture | Masked encoder | Decoder-only |
| Output | 100 mixture samples | 9 quantiles (0.1→0.9) |
| Loss | Negative log-likelihood | Pinball/quantile |
| Patch size | Multi-size | Single size |
| Inference | Slow | KV cache, 2-4× faster |
| Best size | Large (305M) | Small (11.4M) |

### 2. Log-return làm input (không phải raw price)
- Moirai2 pre-trained trên stationary-ish series
- Log-return ≈ stationary, không có unit root
- Raw price có trend → mismatch với pre-training distribution

### 3. Chỉ dùng close price (không dùng OHLCV đầy đủ)
- target_dim=1 đơn giản, tránh overfit với dataset nhỏ
- Intraday features (high-low) có thể thêm sau nếu cần ablation

### 4. Pearson + Sector (không dùng Granger Causality)
- Granger: complex, directed graph, cần nhiều data để ổn định
- Pearson + Sector: đơn giản, interpretable, đủ cho baseline
- Sector membership = prior domain knowledge, không cần tính toán

### 5. GraphSAGE thay vì GCN
- GraphSAGE: inductive, robust khi graph thưa (một số node có ít neighbor)
- GCN: cần normalize degree, kém stable với isolated nodes
- Với 30 nodes, GraphSAGE tốt hơn và nhanh hơn EvolveGCN/TGAT

### 6. Correlation graph update monthly (không real-time)
- Real-time rolling → noisy, graph topology thay đổi quá nhanh
- Monthly update → stable signal, GNN có thể học pattern

---

## Walk-forward Validation

```
Data split theo thời gian (no leakage):
  Train:  data đến hết 2019-12-31
  Val:    2020-01-01 → 2021-12-31  (bao gồm COVID crash → test robustness)
  Test:   2022-01-01 → 2024-12-31

Rolling step: 20 ngày (1 tháng)
```

**Tại sao COVID nằm trong val:** Giai đoạn extreme volatility (Mar 2020)
là test case quan trọng nhất — model phải cảnh báo được spillover effect.

---

## Baselines for Thesis

| Model | Features | Graph | Mục đích |
|---|---|---|---|
| GARCH(1,1) per stock | — | None | Classical benchmark |
| MLP + Moirai2 | embeddings (384) | None | Ablation: does graph help? |
| GraphSAGE + handcrafted | mean, std, skew, RSI | Corr+Sect | Ablation: does Moirai2 help? |
| **GraphSAGE + Moirai2** | **embeddings (384)** | **Corr+Sect** | **Proposed model** |

**Evaluation metrics:** MAE, RMSE, Pearson-r (pred vs actual RV)

---

## Case Study — COVID Spillover (thesis narrative)

**Scenario:** Ngày 14/02/2020, VCB đang ở 88,000 VND.

- GARCH(per-stock): dự báo RV ≈ 1.3% → không thấy nguy hiểm
- GNN + Moirai2: embedding của BID, CTG đã bất thường 3 ngày trước
  → graph propagation → GNN dự báo RV VCB = 3.2% → cảnh báo sớm

**Impact:**
- Investor giảm vị thế sớm → tiết kiệm 102M/500M khi crash (34% drawdown)

**Academic argument:**
> "Cross-asset spillover information, encoded via Moirai2 embeddings and
>  aggregated through GraphSAGE, provides earlier volatility signals than
>  per-stock GARCH models, especially during systemic market stress."

---

## Files Structure

```
moirai/
├── src/
│   ├── data_loader.py       ← Load CSV → GluonTS
│   ├── embed_extractor.py   ← Moirai2 hook → (N, 384) embeddings [DONE]
│   └── volatility_labels.py ← RV label computation [TODO]
├── gnn/
│   ├── build_graph.py       ← Pearson+sector → edge_index, edge_attr [TODO]
│   ├── model.py             ← GraphSAGE + MLP head [TODO]
│   └── train.py             ← Walk-forward training loop [TODO]
├── notebooks/
│   ├── run_demo.ipynb       ← Moirai2 zero-shot demo [DONE]
│   ├── run_own_data.ipynb   ← Custom CSV inference [DONE]
│   └── 04_moirai2_gnn.ipynb ← End-to-end GNN pipeline [TODO]
├── results/                 ← Metrics, plots
└── docs/
    └── gnn-architecture.pdf ← Original architecture document
```

---

## Dependencies

```
# Installed
uni2ts==2.0.0, gluonts, torch==2.2.0, huggingface_hub, pandas, matplotlib
dgl==1.1.2       ← GNN backend (avoids torchdata/graphbolt conflict với DGL>=2.0)
arch             ← GARCH baseline
statsmodels      ← HAR-RV (OLS), LSTM baselines
scikit-learn     ← metrics, preprocessing

# NOT installed (không dùng)
torch-geometric  ← bị conflict; dùng DGL thay thế
```

## Environment

```
HF_HOME=D:\hf_cache           ← Model cache (C: drive too small)
HF_HUB_DISABLE_SYMLINKS_WARNING=1
Python 3.10.11, CPU only (no GPU)
```

## Experiment Results Summary (latest — v4)

**Test set:** 2026-01-05 → 2026-04-07 (62 dates, 30 VN30 stocks, pooled = 1,860 pairs)

| Model | MAE | R² | Pearson_r | Notes |
|---|---|---|---|---|
| HAR-RV (OLS) | **0.0010** | **+0.971** | **0.985** | Per-stock, 2500 train samples |
| LSTM | 0.0011 | +0.967 | 0.983 | Per-stock baseline |
| **MLP (RV6-only)** | **0.0066** | **+0.278** | **0.551** | Best neural — batch trained |
| GNN (RV6-only) | 0.0084 | -0.194 | 0.396 | Batch trained |
| MLP (Moirai2+RV6) | 0.0072 | +0.065 | 0.505 | Moirai2 hurts (+8.4% worse) |
| GNN (Moirai2+RV6) | 0.0091 | -0.519 | 0.377 | Moirai2 hurts (+8.1% worse) |
| Batch GNN (Moirai2+RV3) | 0.0074 | +0.041 | 0.357 | H2 experiment |
| Batch MLP (Moirai2+RV3) | 0.0078 | -0.137 | 0.425 | H2 experiment |
| WalkFwd MLP+Moirai2 | 0.0121 | -1.077 | 0.239 | Walk-forward baseline |
| WalkFwd GNN+Moirai2 | 0.0143 | -1.856 | 0.117 | Walk-forward baseline |
| GARCH(1,1) | 0.0071 | +0.159 | 0.535 | Per-stock MLE |

**Key findings:**
1. Walk-forward training = primary bottleneck (H2 CONFIRMED): batch cuts MAE 35-48%
2. Moirai2 embeddings ADD NOISE when RV features present (+8-9% worse)
3. Extended RV6 > RV3 (6.5× gap vs HAR-RV remains structural)
4. HAR-RV dominates: per-stock OLS, 2500+ samples, 3-param linear model

**Extended RV6 features (6-dim, in src/volatility_labels.py):**
- log(RV_d): past-h daily realized volatility
- log(RV_w): 5-day avg RV (weekly)
- log(RV_m): 22-day avg RV (monthly)
- log(RV_q): 60-day avg RV (quarterly) — extended HAR
- corr_vnindex: rolling 60-day Pearson corr with VNINDEX returns
- jump_ratio: max(RV_d - RV_w, 0) / RV_d — Barndorff-Nielsen & Shephard (2004)
