from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesSegDataset(Dataset):
    """X: (N, L) float32 ; y: (N, T) int64."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, f"X must be (N,L), got {X.shape}"
        assert y.ndim == 2, f"y must be (N,T), got {y.shape}"
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (F,T)
        y = torch.from_numpy(self.y[idx])  # (T,)
        return x, y

class SpectrogramSegDataset(Dataset):
    """X: (N, F, T) float32 ; y: (N, T) int64."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, f"X must be (N,F,T), got {X.shape}"
        assert y.ndim == 2, f"y must be (N,T), got {y.shape}"
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (F,T)
        y = torch.from_numpy(self.y[idx])  # (T,)
        return x, y

@dataclass
class NormalizationStats:
    mean: float
    std: float

def compute_norm_stats(X_train: np.ndarray, eps: float = 1e-8) -> NormalizationStats:
    m = float(X_train.mean())
    s = float(X_train.std())
    return NormalizationStats(mean=m, std=max(s, eps))

def apply_norm(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (X - stats.mean) / stats.std
