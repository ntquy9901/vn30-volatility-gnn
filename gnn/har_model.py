"""
GNN-HAR SISO model: 2-layer GraphSAGE + single output head.

Architecture overview:
  GraphSAGE (Hamilton et al. 2017, "Inductive Representation Learning on Large Graphs"):
    Moi node tap hop (aggregate) features tu cac neighbors, concat voi features
    cua chinh no, roi ap dung linear transform -> embedding moi.
    2 layers = moi node co the "nhin" duoc den 2 buoc (2 hops) trong graph.

Design choices:
  - Aggregator: "mean" (trung binh cac neighbor embeddings). Don gian, on dinh
    voi sparse graph (N=30 stocks, khong phai tat ca deu ket noi nhau).
  - ELU activation: smooth, cho phep gia tri am nho. Tranh "dying ReLU"
    (neurons bi chet vinh vien khi output luon = 0).
  - Output head: Linear (khong activation). Residual co the am hoac duong.

Input:  (30, 3)  -- HAR features [rv_d, rv_w, rv_m] cho 30 stocks tai thoi diem t
Output: (30,)    -- du doan HAR residual (da z-score) cho 1 horizon h
Moi horizon h trong [1, 5, 10, 20] train 1 model rieng.

Xem them: docs/learning/05_graphsage_gnn_har.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GNNHARModel(nn.Module):
    """2-layer GraphSAGE du doan HAR residuals cho 30 VN30 stocks."""

    def __init__(self, in_channels: int = 3, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        # SAGEConv: h_i_new = W * concat(h_i, mean(h_j for j in Neighbors(i))) + b
        # "mean" aggregator: lay trung binh embedding cua tat ca neighbors (ke ca self-loop).
        # Phu hop voi sparse graph vi khong bi anh huong boi so luong neighbors khac nhau.
        self.conv1 = SAGEConv(in_channels, hidden, aggregator_type="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggregator_type="mean")
        # Dropout: trong moi forward pass (training), ngau nhien zero out `dropout` phan
        # tram neurons. Ngan model phu thuoc qua vao bat ky neuron cu the nao -> giam overfit.
        # Rate thap (0.05-0.2) vi hidden=16 qua nho; dropout cao se con qua it neurons active
        # (xem CLAUDE.md C5: rate * hidden >= 10 neurons toi thieu).
        self.drop  = nn.Dropout(dropout)
        # Head: map moi stock tu 16-dim embedding -> 1 scalar residual.
        # Khong co activation vi residual co the am (GNN du doan thap hon HAR) hoac duong.
        self.head  = nn.Linear(hidden, 1)

    def forward(self, g, x: torch.Tensor) -> torch.Tensor:
        """
        g : DGLGraph -- 30-node VN30 graph (edges = correlation + sector)
        x : (30, in_channels) -- snapshot features cho tat ca stocks tai thoi diem t
        Returns (30,) -- z-scored HAR residual du doan cho moi stock

        Luong thong tin: moi stock tap hop tin hieu volatility tu cac stocks tuong quan
        (vi du: tat ca ngan hang deu chiu anh huong lai suat), cho phep GNN sua sai so
        cua HAR can thong tin tu nhieu stocks dong thoi.
        """
        # Layer 1: moi node tap hop 1-hop neighbors -> (30, hidden)
        x = F.elu(self.conv1(g, x))
        # ELU(x) = x neu x > 0; = alpha*(exp(x) - 1) neu x <= 0 (alpha ~ 1).
        # Khac ReLU o cho: ELU co gradient khac 0 cho ca gia tri am -> neurons khong "chet".
        x = self.drop(x)
        # Layer 2: nodes da co embeddings phong phu tu layer 1 (nhin thay 1-hop),
        # nay tiep tuc tap hop -> moi node hieu duoc den 2-hop neighbors -> (30, hidden)
        x = F.elu(self.conv2(g, x))
        x = self.drop(x)
        # (30, 1) -> (30,): 1 scalar output cho moi stock
        return self.head(x).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
