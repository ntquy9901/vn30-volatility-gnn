# Learning Notes: Multi-Step Time Series Forecasting

**Nguồn:** Session brainstorm VN30 volatility — 2026-05-17
**Mục đích:** Review kiến thức về chiến lược dự đoán multi-step cho LSTM/GNN

---

## 1. Teacher Forcing là gì?

**Vấn đề:** Khi train LSTM autoregressive (recursive), bạn dùng ground truth làm input.
Khi inference, bạn dùng prediction của chính model. Hai phân phối input khác nhau → performance drop.

```
TRAINING (teacher forcing):
t=1: [x₁..x₂₂]         → RV̂₁
t=2: [x₁..x₂₂, RV_TRUE₁] → RV̂₂   ← ground truth làm input
t=3: [x₁..x₂₂, RV_TRUE₂] → RV̂₃

INFERENCE (free running):
t=1: [x₁..x₂₂]          → RV̂₁
t=2: [x₁..x₂₂, RV̂₁]    → RV̂₂   ← prediction làm input (khác training!)
t=3: [x₁..x₂₂, RV̂₁, RV̂₂] → RV̂₃
```

**Fix:** Scheduled Sampling — trộn dần prediction vào input khi train:
```python
# Epoch 1: 100% ground truth
# Epoch 10: 50% ground truth, 50% prediction
# Epoch 20: 100% prediction
use_gt = max(0, 1 - epoch/n_epochs)
inp = y_true[t] if random() < use_gt else model_pred[t]
```

---

## 2. Bốn Chiến Lược Multi-Step Forecasting

**Source:** https://letsdatascience.com/blog/multi-step-time-series-forecasting-recursive-direct-and-hybrid-strategies

### 2.1 Recursive Strategy
- 1 model, chạy H lần, feeding prediction vào bước tiếp theo
- **Pros:** Ít tham số, đơn giản
- **Cons:** Error compounds — sai 1 bước → sai lũy tích H bước
- **Best for:** H ngắn (1-5 bước), series ổn định
- **Với VN30:** Không nên — H=20 quá dài, error compounds mạnh

### 2.2 Direct Strategy
- H model riêng biệt, mỗi model predict 1 horizon
- **Pros:** Không error accumulation, mỗi model chuyên biệt
- **Cons:** Train/serve H models riêng biệt, tốn compute
- **Best for:** H dài (10+ bước), đủ data
- **Với VN30:** Được, nhưng nặng (4 LSTM riêng)

### 2.3 MIMO — Multi-Input Multi-Output
- 1 model, 1 forward pass, output vector H chiều
- **Pros:** Coherent predictions, không error accumulation, hiệu quả nhất
- **Cons:** Cần đủ data để học H outputs đồng thời
- **Best for:** Neural networks (LSTM, Transformer)
- **Với VN30:** **RECOMMEND** — 1 LSTM + 4 output heads

```python
class MIMO_LSTM(nn.Module):
    def __init__(self, input_size, hidden, horizons=[1,5,10,20]):
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])

    def forward(self, x):
        _, (h, _) = self.lstm(x)   # h: (1, batch, hidden)
        h = h.squeeze(0)
        return torch.cat([head(h) for head in self.heads], dim=1)  # (batch, 4)

pred = model(X)     # (batch, 4)
y    = labels       # (batch, 4) — [RV1, RV5, RV10, RV20]
loss = MSE(pred, y)
```

### 2.4 DirRec — Hybrid
- H model chained, model sau nhận thêm prediction của model trước
- **Pros:** Capture inter-step dependency, stable hơn recursive
- **Cons:** Sequential, vẫn có error accumulation nhẹ
- **Với VN30:** Quá phức tạp cho thesis

---

## 3. So Sánh Tổng Hợp

| Strategy | Teacher Forcing? | Models | Error Acc. | Recommend VN30 |
|---|---|---|---|---|
| Recursive | CÓ | 1 | Cao | Không |
| Direct | Không | H=4 | Không | Được (nặng) |
| **MIMO** | **Không** | **1** | **Không** | **Best** |
| DirRec | Một phần | H=4 | Thấp | Quá phức tạp |

---

## 4. Data Organization cho LSTM MIMO (VN30)

**Setup:** 1 cổ phiếu, lookback L=20 ngày, stride=1, horizons=[1,5,10,20]

```
Total days ≈ 2,500 | Lookback L=20 | Max horizon=20
Valid windows = 2,500 - 20 - 20 = 2,460 windows/stock

Split:
  Train: t=20  -> t=1750  -> 1,731 samples
  Val:   t=1751 -> t=2000  ->   250 samples
  Test:  t=2001 -> t=2479  ->   479 samples

X shape: (N, 20, features)
y shape: (N, 4)   # [RV1, RV5, RV10, RV20]
```

```python
def compute_rv(log_returns, t, h):
    return log_returns[t:t+h].std()

for t in range(L, N - max_h):
    y[t] = [compute_rv(log_ret, t, 1),
            compute_rv(log_ret, t, 5),
            compute_rv(log_ret, t, 10),
            compute_rv(log_ret, t, 20)]
```

---

## 5. Training Monitoring Best Practices

```python
print(f"Epoch {e:3d} | "
      f"Train: {loss:.4f} [H1={l1:.4f} H5={l5:.4f} H10={l10:.4f} H20={l20:.4f}] | "
      f"Val: {val_loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses,   label='Val')
axes[0].set_title('Total Loss')
axes[0].legend()
for i, h in enumerate([1, 5, 10, 20]):
    axes[1].plot(val_losses_per_h[i], label=f'H={h}')
axes[1].set_title('Val Loss per Horizon')
axes[1].legend()
plt.savefig('results/training_curve.png')
```

---

## 6. Key Rules cho VN30 Thesis

1. Dùng MIMO (1 model, 4 output heads) — không recursive
2. Không teacher forcing trong MIMO (không có autoregressive step)
3. stride=1 cho LSTM per-stock; stride=h cho MLP/GNN batch
4. In loss từng horizon mỗi epoch để debug convergence
5. Plot training curves riêng cho từng horizon
6. Print data split rõ ràng khi khởi động training

---

*Last updated: 2026-05-17*
