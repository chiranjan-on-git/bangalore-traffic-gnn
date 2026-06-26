"""
model.py
Spatio-temporal Graph Neural Network for multi-step traffic forecasting.

Architecture:
  1. SPACE - at each input timestep, stacked graph-convolution layers mix each
     road segment's state with its connected neighbours, capturing how
     congestion propagates across the network.
  2. TIME  - a GRU consumes the sequence of graph-embedded states per segment,
     learning temporal dynamics (build-up and dissipation of jams).
  3. DECODE - a linear head emits OUT_STEPS future speed ratios per segment.

Graph convolution is implemented as the normalised-adjacency matmul that a GCN
mathematically *is*:  H' = ReLU( A_hat @ (H W) )  with  A_hat = D^-1/2 (A+I) D^-1/2.
This is one dense GPU matmul per layer, which runs natively and fast on Apple
MPS / CUDA / CPU - unlike message-passing scatter ops, which fall back to CPU on
MPS and are painfully slow. At this graph size (~2.6k segments) the dense
adjacency is ~28 MB, so dense is both simplest and fastest. For a much larger
graph (>10k nodes) switch A_hat to a sparse tensor.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

import config as C


class STGNN(nn.Module):
    def __init__(self, in_feats, n_nodes, hidden=C.HIDDEN,
                 out_steps=C.OUT_STEPS, gcn_layers=C.GCN_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.out_steps = out_steps

        self.gcn = nn.ModuleList()
        d = in_feats
        for _ in range(gcn_layers):
            self.gcn.append(nn.Linear(d, hidden))
            d = hidden

        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_steps),
        )
        self._A_hat = None  # cached normalised adjacency [N, N]

    def _build_A_hat(self, edge_index, device):
        n = self.n_nodes
        A = torch.zeros(n, n, device=device)
        loops = torch.arange(n, device=device)
        row = torch.cat([edge_index[0].to(device), loops])  # add self-loops
        col = torch.cat([edge_index[1].to(device), loops])
        A[row, col] = 1.0
        deg = A.sum(dim=1)
        dinv = deg.pow(-0.5)
        dinv[torch.isinf(dinv)] = 0.0
        return dinv.unsqueeze(1) * A * dinv.unsqueeze(0)     # D^-1/2 (A+I) D^-1/2

    def forward(self, x, edge_index):
        # x: [B, IN, N, F]  ->  out: [B, OUT, N]
        B, T, N, Fdim = x.shape
        if self._A_hat is None or self._A_hat.device != x.device:
            self._A_hat = self._build_A_hat(edge_index, x.device)
        A = self._A_hat

        spatial = []
        for t in range(T):
            h = x[:, t]                                      # [B, N, F]
            for lin in self.gcn:
                h = torch.relu(torch.matmul(A, lin(h)))      # A_hat @ (H W)
            h = self.dropout(h)
            spatial.append(h)                                # [B, N, H]

        seq = torch.stack(spatial, dim=1)                    # [B, T, N, H]
        seq = seq.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden)
        _, hn = self.gru(seq)                                # [1, B*N, H]
        y = self.head(hn.squeeze(0))                         # [B*N, OUT]
        y = torch.sigmoid(y)                                 # speed ratios in (0,1)
        return y.reshape(B, N, self.out_steps).permute(0, 2, 1)


def save_checkpoint(model, mean, std, in_feats, n_nodes, extra=None):
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean, "std": std,
        "in_feats": in_feats, "n_nodes": n_nodes,
        "hidden": C.HIDDEN, "out_steps": C.OUT_STEPS,
        "gcn_layers": C.GCN_LAYERS, "in_steps": C.IN_STEPS,
        "extra": extra or {},
    }, C.MODEL_PATH)


def load_checkpoint(device="cpu"):
    ckpt = torch.load(C.MODEL_PATH, map_location=device, weights_only=False)
    model = STGNN(
        in_feats=ckpt["in_feats"], n_nodes=ckpt["n_nodes"],
        hidden=ckpt["hidden"], out_steps=ckpt["out_steps"],
        gcn_layers=ckpt["gcn_layers"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt