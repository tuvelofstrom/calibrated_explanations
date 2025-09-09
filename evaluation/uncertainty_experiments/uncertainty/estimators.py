from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict


def ensemble_disagreement_classification(samples_proba: np.ndarray) -> np.ndarray:
    """samples_proba: [S, N, C] ensemble probability samples.
    Returns per-point variance/entropy proxy: std of p(class=1) for binary; mean var over C for multiclass.
    """
    if samples_proba.ndim == 3 and samples_proba.shape[2] > 1:
        var = samples_proba.var(axis=0).mean(axis=1)
        return var
    # Binary case: samples are [S,N] or [S,N,1/2]
    if samples_proba.ndim == 3:
        p1 = samples_proba[..., 1]
    else:
        p1 = samples_proba
    return p1.var(axis=0)


def ensemble_disagreement_regression(samples_pred: np.ndarray) -> np.ndarray:
    """samples_pred: [S, N] ensemble predictions. Returns variance across samples per point."""
    return samples_pred.var(axis=0)


def knn_inverse_density(X: np.ndarray, k: int = 20) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(X)))
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # Use average distance to k-th neighbors as inverse density proxy
    inv_density = dists.mean(axis=1)
    return inv_density


def nonconformity_regression(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred)


