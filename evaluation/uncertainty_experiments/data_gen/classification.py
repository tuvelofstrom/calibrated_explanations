from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Tuple


def make_binary_gaussian(
    n_train: int,
    n_cal: int,
    n_test: int,
    dims: int = 2,
    overlap_scale: float = 1.0,
    label_noise_map: Optional[Dict[str, float]] = None,
    holes: Optional[List[List[float]]] = None,
    shift: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Feature-dependent Bayes error via position-dependent means
    def class_means(X):
        # Means vary with x1 to induce varying overlap
        m0 = np.zeros(X.shape[1])
        m1 = np.zeros(X.shape[1])
        if X.shape[1] >= 2:
            m0[:2] = np.array([-0.6, -0.2])
            m1[:2] = np.array([0.6, 0.2])
        return m0, m1

    def sample_split(n):
        X = rng.uniform(-1.0, 1.0, size=(n, dims))
        m0, m1 = class_means(X)
        # Covariance fixed, means fixed; overlap induced by global positioning and optional scaling
        cov = np.eye(dims) * (0.4 * overlap_scale) ** 2
        n0 = n // 2
        n1 = n - n0
        X0 = rng.multivariate_normal(m0, cov, size=n0)
        X1 = rng.multivariate_normal(m1, cov, size=n1)
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        return X, y

    Xtr, ytr = sample_split(n_train)
    Xca, yca = sample_split(n_cal)
    Xte, yte = sample_split(n_test)
    Xte = Xte + np.array([0.0, shift] + [0.0] * max(0, dims - 2))

    if holes:
        mask = np.ones(len(Xtr), dtype=bool)
        for h in holes:
            x1_min, x2_min, x1_max, x2_max = h
            in_hole = (Xtr[:, 0] >= x1_min) & (Xtr[:, 0] <= x1_max) & (Xtr[:, 1] >= x2_min) & (Xtr[:, 1] <= x2_max)
            mask &= ~in_hole
        Xtr, ytr = Xtr[mask], ytr[mask]

    # Feature-conditional label noise η(x): flip with probability increasing with x1
    def eta(X):
        return 0.02 + 0.18 * (1.0 / (1.0 + np.exp(-2.0 * X[:, 0])))

    def flip_labels(X, y):
        p = eta(X)
        flips = rng.uniform(0, 1, size=len(y)) < p
        y_noisy = y.copy()
        y_noisy[flips] = 1 - y_noisy[flips]
        return y_noisy, p

    ytr_noisy, eta_tr = flip_labels(Xtr, ytr)
    yca_noisy, eta_ca = flip_labels(Xca, yca)
    yte_noisy, eta_te = flip_labels(Xte, yte)

    return {
        "X_train": Xtr,
        "y_train": ytr_noisy,
        "X_cal": Xca,
        "y_cal": yca_noisy,
        "X_test": Xte,
        "y_test": yte_noisy,
        "eta_train": eta_tr,
        "eta_cal": eta_ca,
        "eta_test": eta_te,
    }

