# Learning Notes: GraphSAGE + HAR Residual Training cho VN30

**Nguon:** Implementation session GNN+HAR SISO — 2026-05-27  
**Cau hoi goc:** GraphSAGE la gi? Tai sao dung GNN cho du bao volatility? HAR residual training hoat dong nhu the nao?

---

## 1. Graph Neural Network (GNN) — Khai niem co ban

### Van de can giai quyet

HAR-RV la mo hinh per-stock: du bao VCB chi dua tren lich su VCB, khong biet TCB hay BID dang bien dong the nao. Nhung tren thi truong, cac stocks trong cung nganh (ngan hang, nang luong...) co xu huong bien dong cung nhau do chia se:
- Cung nguon rui ro (lai suat, quy dinh, gia hang hoa)
- Cung nha dau tu co the (institutional investors trade nhieu nganh cung luc)

**Y tuong GNN:** xay dung mot mang trong do moi stock la mot node, cac stocks co tuong quan cao la edges. De model hoc cach dung thong tin tu "lang gieng" (neighbors) cua moi stock.

### Graph la gi trong ngu canh nay?

```
Nodes: 30 VN30 stocks
Edges: stock i -- stock j neu:
  - Pearson(log_returns_i, log_returns_j) > 0.4 (co-movement manh)
  - HOAC cung sector (bank, energy, ...)
```

Moi ngay t, GNN nhan mot "snapshot":
- **Input** (X): ma tran (30, 3) -- [rv_d, rv_w, rv_m] cua 30 stocks tai ngay t
- **Output**: vector (30,) -- HAR residual du doan cho 30 stocks

---

## 2. GraphSAGE — Thuat toan

### Tai sao khong dung GCN (Graph Convolutional Network)?

GCN yeu cau normalized adjacency matrix (D^{-1/2} A D^{-1/2}), khuech dai nhieu cho nodes co degree thap (stocks co it ket noi). VN30 chi co 30 nodes voi graph thoa, mot so nganh nhu aviation (VJC), rubber (GVR) it ket noi -> GCN co the cho embedding kem.

**GraphSAGE** (Hamilton et al. 2017) robust hon: no sample neighbors, lay aggregate, concat voi self:

### Cong thuc GraphSAGE (Mean Aggregator)

Cho moi node i tai layer l:

```
h_neighbors = mean( h_j^(l-1)  cho j trong N(i) )
h_i^(l)     = W^(l) * concat( h_i^(l-1), h_neighbors ) + b^(l)
h_i^(l)     = activation( h_i^(l) )
```

- `N(i)`: tap hop neighbors cua node i (co ca self-loop thi bao gom ca i)
- `W^(l)`: ma tran trong so co the hoc (shared giua tat ca nodes)
- `concat`: ghep embedding cua node voi trung binh neighbors

**Viet lai bang code:**
```python
self.conv1 = SAGEConv(in_channels=3, out_channels=16, aggregator_type="mean")
# Moi buoc forward: x_new = W * concat(x_self, mean(x_neighbors)) + b
```

### 2 layers co y nghia gi?

- **Layer 1**: moi stock nhin thay cac stocks ket noi truc tiep (1-hop)
  - VCB thay TCB, BID, MBB (cung la ngan hang)
- **Layer 2**: moi stock nhin thay 2-hop neighbors
  - VCB -> TCB -> HPG (HPG vay von ngan hang, tuong quan gian tiep)

2 layers la du cho N=30 stocks; them layers se overfit (model nhin thay qua nhieu trong graph nho).

---

## 3. HAR Residual Training — Y tuong chinh

### Van de khi train truc tiep

Neu GNN du bao RV truc tiep, no phai hoc lai toan bo cau truc HAR (lag-1, lag-5, lag-20) tu so lieu it. Voi chi 149 training snapshots va ESS=123, day la task qua kho -> overfit, R2 am.

### Giai phap: Train tren phan du (residual)

```
y_residual[t] = y_actual[t] - y_HAR[t]
```

GNN hoc du doan `y_residual`, sau do:

```
y_final = y_HAR + GNN_residual
y_final = clip(y_final, 0, inf)  # RV >= 0 luon luon
```

**Dam bao quan trong (floor guarantee):**  
Neu GNN output = 0 (khong hoc duoc gi), `y_final = y_HAR`.  
GNN *khong the* lam ket qua toi hon HAR tru phi no du doan sai mot cach tich cuc.

**Tai sao phan du de hoc hon?**
- Phan du co mean ~ 0 (HAR da bat duoc xu huong chinh)
- Bien do nho hon (GNN chi can sua cac sai so bien), khong phai hoc toan bo signal
- Nhieu it hon: loai bo phan HAR da giai thich tot -> phan con lai co ti-signal/noise cao hon

---

## 4. Z-score Normalization — Tai sao bat buoc?

### Van de scale

Cac VN30 stocks co muc vol rat khac nhau:

| Stock | RV trung binh (daily) | 
|---|---|
| GAS (dau khi) | ~0.020 |
| SSB (ngan hang nho) | ~0.006 |
| VNM (FMCG) | ~0.010 |

Neu train tren raw residuals, MSE Loss bi thong tri boi GAS va NVL (residuals lon hon 3x).
Model se:
- Tap trung toi uu hoa cho GAS/NVL
- Bo qua SSB, VNM (residuals nho -> dong gop it vao loss)

### Giai phap: Z-score tren residual per stock

```python
# Tinh tren TRAIN only (tranh data leakage)
rv_mu  = residual_train.mean(axis=0)  # (30,): ~ 0 cho moi stock
rv_sig = residual_train.std(axis=0)   # (30,): do lenh chuan cua residual moi stock

# Normalize
residual_norm = (residual - rv_mu) / rv_sig  # mean=0, std=1 cho moi stock
```

Sau khi z-score, moi stock dong gop ngang bang vao MSE loss. Model hoc deu tren tat ca 30 stocks.

**Undo khi du bao (denormalize):**
```python
gnn_res = pred_norm * rv_sig + rv_mu  # tra ve don vi goc
```

---

## 5. DropEdge — Chinh quy hoa tren Graph

**Van de:** VN30 chi co 30 nodes. Model co the "nho" chính xac cau truc graph training va overfit.

**DropEdge (Rong et al. 2020):** Trong moi epoch training, ngau nhien loai bo p% edges:

```python
DROP_EDGE_P = 0.2  # xoa 20% edges moi epoch

mask = torch.rand(g.num_edges()) > DROP_EDGE_P
g_dropped = dgl.graph((src[mask], dst[mask]))
# Dung g_dropped cho forward pass, giu g goc cho validation
```

Moi epoch, model gap mot graph hoi khac -> buoc phai hoc representations robust hon, khong qua fit cau truc cu the.

Tuong tu Dropout nhung o muc do graph (edge level) thay vi neuron level.

---

## 6. Snapshot va Stride

### Snapshot la gi?

Thay vi xu ly chuoi thoi gian lien tuc (nhu LSTM), GNN xu ly tung "snapshot" doc lap:

```
Snapshot tai ngay t:
  Input X[t]:  (30, 3) -- HAR features [rv_d, rv_w, rv_m] tai ngay t
  Label y[t]:  (30,)   -- RV target std(ret[t+1 : t+H])
```

Moi snapshot la 1 "buc anh" cua thi truong. GNN hoc tu nhieu buc anh.

### Van de overlap giua labels

Voi h=5, stride=1:
```
y[t]   = std(ret[t+1 : t+5])  -- dung ngay t+1, t+2, t+3, t+4, t+5
y[t+1] = std(ret[t+2 : t+6])  -- dung ngay t+2, t+3, t+4, t+5, t+6
```
4/5 returns trung nhau -> labels cua 2 snapshots lien tiep tuong quan 80%. Model thay data "moi" nhung thuc ra labels rat giong nhau.

### Giai phap: Stride = H

```python
STRIDE_H = {1: 5, 5: 5, 10: 10, 20: 20}
```

Voi h=5, stride=5: moi 5 ngay lay 1 snapshot. Cac labels khong con trung nhau (0% overlap).

**Danh doi:** it snapshots hon. stride=5, h=5: ~300 ngay / 5 = 60 snapshots train.  
Nhung ESS (Effective Sample Size) tinh tren phan du khong overlap thuc su cao hon.

---

## 7. Static Graph vs Dynamic Graph

Mo hinh nay dung **static graph** (1 graph duy nhat, xay dung 1 lan tu toan bo training period):

```
Graph(train_end_date):
  edges = Pearson(log_returns[2006 : 2025-12]) > 0.4  OR  same sector
```

**Uu diem:**
- On dinh: Pearson tren nhieu nam it bi nhieu hon Pearson cua so 60 ngay
- Hieu qua: chi xay dung 1 lan
- Phu hop voi VN30: cau truc nganh it thay doi (cac ngan hang van la ngan hang)

**Nhuoc diem:**
- Bo qua su thay doi tuong quan theo thoi gian (vi du: stocks co the tach roi nhau sau COVID)
- Khong bat duoc "crisis correlation" (trong COVID, tat ca stocks co xu huong tuong quan cao hon)

---

## 8. Ket qua thuc tien tren VN30

| h | GNN R2 | HAR R2 | Delta | GNN > HAR |
|---|---|---|---|---|
| h=1 | -0.029 | -0.028 | -0.001 | 18/30 |
| h=5 | +0.687 | +0.631 | +0.055 | 30/30 |
| h=10 | +0.883 | +0.837 | +0.046 | 27/30 |
| h=20 | +0.936 | +0.900 | +0.036 | 29/30 |

**Phan tich:**
- **h=5,10,20**: GNN ro rang thang HAR. HAR residual training + graph structure cua nganh cho phep GNN su dung thong tin ngang hang hieu qua hon.
- **h=1**: GNN gan ngang bang HAR. Volatility 1-ngay rat nhieu nhieu (noisy); thong tin tu neighbors it giu hanh hon HAR's simple AR.
- **Stocks GNN thua tai h=1**: GAS, PLX, NVL, GVR (dau khi, xang dau, bat dong san) -- vol cua cac stocks nay bi dan dat boi yeu to ben ngoai graph (gia dau the gioi, chinh sach PVN) ma HAR bat duoc qua lag features.

---

## 9. Cac Gap nho quan trong

| Dieu nen nho | Li do |
|---|---|
| Z-score tinh tu TRAIN only | Data leakage neu dung val/test |
| Clip final_pred >= 0 | RV = std() luon >= 0 |
| DropEdge chi trong training | Val/test dung graph day du |
| HAR refit tren train only | Tranh leakage tu val dates |
| feat_ok AND tgt_ok | Ca 2 phai non-NaN moi lay snapshot |
| stride = H cho train, stride = 1 cho test | Test phai danh gia tren moi ngay |
