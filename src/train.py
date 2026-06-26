"""
train.py
Trains the spatio-temporal GNN on the generated history and reports MAE / RMSE.
Uses Apple Silicon (MPS), CUDA, or CPU automatically.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from src import road_network
from src.dataset import build_splits
from src.model import STGNN, save_checkpoint
from src.traffic_generator import FEATURE_NAMES


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, edge_index, device):
    model.eval()
    abs_err, sq_err, count = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, edge_index)
        diff = (pred - y).abs()
        abs_err += diff.sum().item()
        sq_err += (diff ** 2).sum().item()
        count += diff.numel()
    mae = abs_err / count
    rmse = (sq_err / count) ** 0.5
    return mae, rmse


def train():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = pick_device()
    print(f"Device: {device}")

    links, edge_index_np = road_network.load()
    data = np.load(C.TRAFFIC_PATH)
    X, speed_ratio = data["X"], data["speed_ratio"]
    n_nodes = X.shape[1]
    in_feats = X.shape[2]
    print(f"Loaded series X={X.shape}, links={n_nodes}, features={in_feats}")

    train_ds, val_ds, test_ds, mean, std = build_splits(X, speed_ratio)
    print(f"Windows -> train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE)

    model = STGNN(in_feats=in_feats, n_nodes=n_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    lossf = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    best_val = float("inf")
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x, edge_index), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)
        val_mae, val_rmse = evaluate(model, val_loader, edge_index, device)
        flag = ""
        if val_mae < best_val:
            best_val = val_mae
            save_checkpoint(model.cpu(), mean, std, in_feats, n_nodes,
                            extra={"feature_names": FEATURE_NAMES})
            model.to(device)
            flag = "  <- saved"
        print(f"epoch {epoch:2d} | train mse {train_loss:.4f} | "
              f"val MAE {val_mae:.4f} RMSE {val_rmse:.4f}{flag}")

    # Final test with the best checkpoint.
    from src.model import load_checkpoint
    model, _ = load_checkpoint(device)
    test_mae, test_rmse = evaluate(model, test_loader, edge_index, device)
    print("-" * 50)
    print(f"TEST  MAE {test_mae:.4f}  RMSE {test_rmse:.4f}  "
          f"(speed-ratio units; ~{test_mae*100:.1f}% mean error)")
    print(f"Saved model -> {C.MODEL_PATH}")


if __name__ == "__main__":
    train()
