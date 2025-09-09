from __future__ import annotations

import numpy as np
from typing import Dict


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    if probs.ndim == 2 and probs.shape[1] > 1:
        # multiclass: use max prob and correctness
        conf = probs.max(axis=1)
        y_pred = probs.argmax(axis=1)
        acc = (y_pred == y_true).astype(float)
    else:
        conf = probs.reshape(-1)
        y_pred = (conf >= 0.5).astype(int)
        acc = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(m):
            continue
        gap = abs(acc[m].mean() - conf[m].mean())
        ece_val += (m.mean()) * gap
    return float(ece_val)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    if probs.ndim == 2 and probs.shape[1] > 1:
        Y = np.eye(probs.shape[1])[y_true]
        return float(np.mean(np.sum((probs - Y) ** 2, axis=1)))
    p = probs.reshape(-1)
    return float(np.mean((p - y_true) ** 2))


def coverage_width(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    covered = (y_true >= lower) & (y_true <= upper)
    width = (upper - lower)
    return {
        "coverage": float(np.mean(covered)),
        "avg_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }

