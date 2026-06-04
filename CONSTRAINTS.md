# VN30 Volatility Forecasting — Design Constraints

Đây là file ràng buộc kỹ thuật bắt buộc cho project.
**Khi brainstorm, plan, hay implement: phải tuân theo tất cả rules dưới đây.**
Không được vi phạm trừ khi có lý do rõ ràng được ghi nhận tại đây.

---

## RULE 1 — Multi-Horizon: 1 Model, 4 Horizons

**Constraint:** Model chính phải dự đoán đồng thời h=1, 5, 10, 20 ngày trong 1 forward pass.

**Implementation:**
- Dùng chiến lược **MIMO** (Multi-Input Multi-Output) — 1 backbone + 4 output heads
- **KHÔNG dùng Recursive strategy** (teacher forcing / error accumulation)
- **KHÔNG train 4 model riêng biệt** trừ khi có lý do đặc biệt (Direct strategy)
- Horizons phải configurable qua 1 parameter: `HORIZONS = [1, 5, 10, 20]`

### NGOẠI LỆ — SISO Ablation Experiments (2026-05-18)

**Cho phép** train SISO (1 model, 1 horizon) với mục đích ablation study kiểm tra MIMO gradient interference hypothesis.

**Điều kiện:**
- Script SISO phải để riêng trong `baselines/train_lstm_siso_*.py`
- Output paths riêng biệt (không ghi đè MIMO results)
- Kết quả so sánh phải đi kèm với MIMO results trong cùng bảng
- Chỉ dùng cùng HAR features, hyperparameters, data split với MIMO để đảm bảo controlled comparison

**Lý do cho phép:** Learning curve analysis cho thấy h=10/h=20 degradation kéo MIMO loss lên, trigger early stopping sớm. SISO h=5 test hypothesis: MIMO interference vs ESS bottleneck. Kết quả có giá trị luận văn dù outcome nào.

```python
HORIZONS = [1, 5, 10, 20]   # configurable — thay đổi ở đây, không hardcode

class Model(nn.Module):
    def __init__(self, horizons=HORIZONS):
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x):
        feat = self.backbone(x)
        return torch.cat([head(feat) for head in self.heads], dim=1)  # (B, 4)
```

**Lý do:** MIMO không có teacher forcing, 1 forward pass, rõ ràng khi báo cáo.

---

## RULE 2 — Training Monitoring: Phải hiển thị progress và lưu curves

**Constraint:** Mọi training loop phải:
1. In ra loss từng horizon sau mỗi epoch
2. Lưu training curve ra file PNG sau khi train xong

**Implementation bắt buộc:**

```python
# In mỗi epoch — BẮT BUỘC
print(f"Epoch {epoch:3d}/{n_epochs} | "
      f"Train: {train_loss:.4f} "
      f"[H1={h1:.4f} H5={h5:.4f} H10={h10:.4f} H20={h20:.4f}] | "
      f"Val: {val_loss:.4f} | "
      f"LR: {scheduler.get_last_lr()[0]:.2e}")

# Plot sau khi train xong — BẮT BUỘC
def plot_training_curves(train_losses, val_losses, val_per_horizon,
                         horizons, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Val")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    for i, h in enumerate(horizons):
        axes[1].plot(val_per_horizon[i], label=f"H={h}")
    axes[1].set_title("Val Loss per Horizon")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Training curve saved: {save_path}")
```

**Lý do:** Cần kiểm tra convergence và có hình để báo cáo cho thầy.

---

## RULE 3 — Data Split: Phải in rõ khi khởi động

**Constraint:** Mọi script training phải in ra thông tin split trước khi train.

**Splits chuẩn:**
```
Train: 2014-xx-xx -> 2019-12-31
Val:   2020-01-01 -> 2021-12-31
Test:  2022-01-01 -> 2024-xx-xx
```

**Implementation bắt buộc:**

```python
def print_data_split(train_df, val_df, test_df, model_name=""):
    print(f"\n{'='*55}")
    print(f"  DATA SPLIT — {model_name}")
    print(f"{'='*55}")
    print(f"  Train: {train_df.index[0].date()} -> {train_df.index[-1].date()}"
          f"  ({len(train_df):,} days)")
    print(f"  Val:   {val_df.index[0].date()} -> {val_df.index[-1].date()}"
          f"  ({len(val_df):,} days)")
    print(f"  Test:  {test_df.index[0].date()} -> {test_df.index[-1].date()}"
          f"  ({len(test_df):,} days)")
    print(f"{'='*55}\n")
```

**Với LSTM (per-stock, stride=1):**
```
  Model:    LSTM per-stock (VCB)
  Train:    2014-07-01 -> 2019-12-31  (1,629 windows, ESS~81)
  Val:      2020-01-01 -> 2021-12-31  (  501 windows, ESS~25)
  Test:     2022-01-01 -> 2024-12-31  (  730 windows, ESS~37)
```

**Với GNN/MLP (batch, stride=20):**
```
  Model:    GNN batch (30 stocks)
  Train:    2014-07-01 -> 2019-12-31  ( ~82 snapshots)
  Val:      2020-01-01 -> 2021-12-31  ( ~25 snapshots)
  Test:     2022-01-01 -> 2024-12-31  ( ~37 snapshots)
```

**Lý do:** Minh bạch cho báo cáo, tránh data leakage, dễ debug.

---

## RULE 4 — LSTM: Dùng MIMO, stride=1, in verbose

**Constraint cho LSTM:**
- **stride = 1** (per-stock, ~2,458 samples/stock)
- **MIMO heads** — không recursive, không separate models
- **In verbose** lúc load data và lúc train
- Lookback L=20 ngày (configurable qua `LOOKBACK`)

**Data organization bắt buộc:**

```python
LOOKBACK  = 20           # configurable
HORIZONS  = [1, 5, 10, 20]  # configurable
MAX_H     = max(HORIZONS)

# X[t] = log_return[t-LOOKBACK : t]        shape: (LOOKBACK, n_features)
# y[t] = [RV_h for h in HORIZONS]          shape: (len(HORIZONS),)
# Valid t: LOOKBACK <= t <= N - MAX_H

# In khi load:
print(f"  Stock {ticker}: {N} days | "
      f"{n_train} train | {n_val} val | {n_test} test samples")
print(f"  ESS train = {n_train // MAX_H} independent observations")
```

**Lý do:** stride=1 tối đa hóa samples; ESS cần in ra để đánh giá data sufficiency.

---

## RULE 6 — Global Data Split & Training Monitoring (NEW)

**Constraint:** Áp dụng cho TẤT CẢ loại model (LSTM, GNN, MLP, HAR, v.v.)

### Data Split bắt buộc:
- **Test set:** 2026-01-01 onwards (boundary chung cho tất cả models)
- **Train/Val split:** 80/20 từ data trước 2026-01-01
  - Train: 80% đầu tiên của data trước 2026-01-01
  - Val: 20% cuối cùng của data trước 2026-01-01

**Implementation:**

```python
GLOBAL_TEST_START = "2026-01-01"
TRAIN_VAL_SPLIT_RATIO = 0.8

# Ví dụ: 2,400 samples trước 2026-01-01
# → Train: 1,920 samples (80%)
# → Val: 480 samples (20%)
# → Test: từ 2026-01-01 onwards
```

### Training Monitoring bắt buộc:

1. **In progress từng epoch ra console:**
```python
print(f"Epoch {epoch:3d}/{n_epochs} | "
      f"Train: {train_loss:.4f} [H1={h1:.4f}] | "
      f"Val: {val_loss:.4f} [H1={h1:.4f}] | "
      f"LR: {lr:.2e}")
```

2. **Learning curves:** Vẽ và lưu PNG sau mỗi training
   - Code vẽ learning curve phải nằm **trong training script** (không riêng file)
   - Save: `results/{model}_{config}_curves.png`
   - Hiển thị: Train loss, Val loss, per-horizon breakdown

**Lý do:** Cần theo dõi convergence real-time, transparent khi báo cáo, đảm bảo consistency qua tất cả models.

---

## Summary Table

| Rule | Nội dung | Mức độ |
|---|---|---|
| R1 | MIMO multi-horizon (h=1,5,10,20 configurable); SISO cho phép trong ablation | BẮT BUỘC + ngoại lệ |
| R2 | In loss từng horizon + lưu training curve PNG | BẮT BUỘC |
| R3 | Print data split rõ ràng trước khi train | BẮT BUỘC |
| R4 | LSTM stride=1, MIMO, in ESS | BẮT BUỘC |
| R5 | Tuân theo file CONSTRAINTS.md này | BẮT BUỘC |
| R6 | Global data split (80/20 train/val, test từ 2026-01-01) + console + learning curves | BẮT BUỘC |

---

## Model Implementation Checklist (MANDATORY)

**For every loss function implementation:** 

Learned from the v1.3_LOSS_FIX bug correction. Before using any loss function in training, verify:

- [ ] **Line-by-line comparison** with paper source code
- [ ] **Parameter order verification** (which is target, which is prediction?)
- [ ] **Epsilon placement check** (guards both division AND log?)
- [ ] **Edge case testing** (zero values, near-zero values, perfect prediction)
- [ ] **Gradient flow validation** (no NaN/inf gradients)
- [ ] **Paper test value verification** (if available in paper)
- [ ] **Minimum loss documentation** (not all losses minimize to 0)
- [ ] **Comprehensive test suite** (cover all edge cases)
- [ ] **Clear function naming** (avoid misleading names)
- [ ] **Mathematical documentation** (formula, behavior, asymmetry)

**Reference:** See `docs/bug_fix_v1.3_loss_ratio.md` for why this checklist matters.

---

## Loss Function Architecture Decision

### Background: Q-LIKE vs MSE for Volatility Forecasting

The project uses two different loss functions for different purposes:

**Q-LIKE Ratio Loss (GNNHAR models):**
- Formula: `L = mean(y_true/(y_pred+eps) - log(y_true/(y_pred+eps)+eps))`
- Properties: Scale-invariant, asymmetric penalty (under-prediction penalized more)
- Use case: GNNHAR paper reproduction, risk management focus
- Stability: More complex, requires guardrails (ratio clipping, gradient clipping)
- Interpretation: Loss minimum = 1.0 (not 0.0)

**MSE on Z-Scored Residuals (baselines, recommended):**
- Formula: `L = mean((y_pred - y_true)^2)` on z-scored data
- Properties: Scale-dependent, symmetric penalty, numerically stable
- Use case: All other models, production systems
- Stability: Well-understood, no special guardrails needed
- Interpretation: Loss minimum = 0.0

### Architectural Trade-offs

**Why Q-LIKE for GNNHAR:**
1. **Paper reproduction:** Match GNNHAR paper implementation exactly
2. **Risk management:** Asymmetric penalty aligns with volatility risk management
3. **Scale-invariance:** Handles different volatility regimes naturally

**Why MSE for other models:**
1. **Stability:** Numerically stable, no edge cases
2. **Interpretability:** R², MAE, RMSE map directly to business metrics
3. **Simplicity:** Well-understood optimization landscape
4. **Standardization:** Works with z-scored normalization for fair comparison

### Recommendation

**For this project:**
- Use MSE on z-scored residuals for all models except GNNHAR paper reproduction
- For GNNHAR models, use Q-LIKE ratio loss with implemented guardrails:
  - Ratio clipping (clip_min=1e-4, clip_max=1e4, default enabled)
  - Gradient clipping (max_norm=1.0, default enabled)
  - Ratio monitoring during training (every 10 epochs, default enabled)
- Monitor training stability and fall back to MSE if issues arise

**For production systems:**
- Prefer MSE on z-scored residuals for stability and interpretability
- Consider Q-LIKE only if asymmetric penalty is explicitly required
- Always implement guardrails when using ratio-based losses

### Implementation Requirements

When using Q-LIKE ratio loss, the following guardrails are MANDATORY:

- [x] Ratio clipping implemented (clip_min=1e-4, clip_max=1e4)
- [x] Gradient clipping implemented (max_norm=1.0)
- [x] Ratio monitoring during training (every 10 epochs)
- [x] Comprehensive test suite (edge cases covered)
- [ ] Training stability verified across multiple seeds
- [ ] Results validated against MSE baseline

**See also:** `docs/learning/06_gnnhar_ratio_loss.md` for detailed explanation

---

## Effective Sample Size Reference

```
ESS = N_raw / max_horizon   (Lopez de Prado 2018, Ch.7)

Per-stock LSTM (h=20, stride=1):
  ESS ≈ 2,458 / 20 = 123  -> data-limited regime

Cross-stock pooled (30 stocks):
  ESS ≈ 123 × 30 = 3,690  -> feasible cho LSTM

HAR-RV (h=20, stride=20):
  ESS ≈ 2,500 / 20 = 125, params=3 -> obs/param=41 -> BLUE
```

---

*Created: 2026-05-17 | Project: VN30 Volatility Forecasting (Moirai2 + GNN)*
