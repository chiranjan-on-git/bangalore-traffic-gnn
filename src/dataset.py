"""
dataset.py
Turns the [T, N, F] feature series into supervised spatio-temporal windows:
  input  = past IN_STEPS timesteps of all features      -> [IN_STEPS, N, F]
  target = next OUT_STEPS timesteps of speed_ratio       -> [OUT_STEPS, N]

Splits chronologically (train | val | test), fits a feature scaler on train only.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


class TrafficWindows(Dataset):
    def __init__(self, X, speed_ratio, indices, mean, std):
        self.X = X
        self.y = speed_ratio
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        t = self.indices[k]
        x = self.X[t:t + C.IN_STEPS]                       # [IN, N, F]
        y = self.y[t + C.IN_STEPS:t + C.IN_STEPS + C.OUT_STEPS]  # [OUT, N]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def build_splits(X, speed_ratio):
    """Return train/val/test datasets plus the fitted scaler (mean, std)."""
    T = X.shape[0]
    win = C.IN_STEPS + C.OUT_STEPS
    last_start = T - win                                    # inclusive
    starts = np.arange(0, last_start + 1)

    n = len(starts)
    n_test = int(n * C.TEST_FRACTION)
    n_val = int(n * C.VAL_FRACTION)
    n_train = n - n_val - n_test

    train_idx = starts[:n_train]
    val_idx = starts[n_train:n_train + n_val]
    test_idx = starts[n_train + n_val:]

    # Fit scaler on the timesteps that belong to training windows only.
    train_end_t = train_idx[-1] + C.IN_STEPS
    flat = X[:train_end_t].reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0

    make = lambda idx: TrafficWindows(X, speed_ratio, idx, mean, std)
    return make(train_idx), make(val_idx), make(test_idx), mean, std
