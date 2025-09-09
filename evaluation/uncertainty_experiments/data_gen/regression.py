from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List


def base_function(X: np.ndarray) -> np.ndarray:
    """Mixed smooth + piecewise base for 2D; extends trivially to >2D.

    f(x) = sin(2πx1) + 0.5*x2 + ReLU(x1+x2-0.5)
    For dims>2, treat remaining dims as linear with small weights.
    """
    x1 = X[:, 0]
    x2 = X[:, 1] if X.shape[1] > 1 else 0.0
    rem = 0.0
    if X.shape[1] > 2:
        rem = (X[:, 2:] @ np.linspace(0.05, 0.01, X.shape[1] - 2))
    return np.sin(2 * np.pi * x1) + 0.5 * x2 + np.maximum(0.0, x1 + x2 - 0.5) + rem


def hetero_sigma(X: np.ndarray, a: float = 1.5, b: float = -1.0, base: float = 0.1, scale: float = 0.8) -> np.ndarray:
    z = a * X[:, 0] + b * (X[:, 1] if X.shape[1] > 1 else 0.0)
    return base + scale * (1.0 / (1.0 + np.exp(-z)))


def make_grid(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(n, dims))


def remove_holes(X: np.ndarray, y: np.ndarray, holes: Optional[List[List[float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not holes:
        return X, y
    mask = np.ones(len(X), dtype=bool)
    for h in holes:
        x1_min, x2_min, x1_max, x2_max = h
        in_hole = (X[:, 0] >= x1_min) & (X[:, 0] <= x1_max) & (X[:, 1] >= x2_min) & (X[:, 1] <= x2_max)
        mask &= ~in_hole
    return X[mask], y[mask]


def generate_regression(
    n_train: int,
    n_cal: int,
    n_test: int,
    dims: int = 2,
    holes: Optional[List[List[float]]] = None,
    shift: float = 0.0,
    hetero: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xtr = make_grid(n_train, dims, rng)
    Xca = make_grid(n_cal, dims, rng)
    Xte = make_grid(n_test, dims, rng) + np.array([0.0, shift] + [0.0] * max(0, dims - 2))

    sigma_fn = (lambda X: hetero_sigma(X, **hetero)) if hetero else (lambda X: np.full(X.shape[0], 0.1))

    f_tr = base_function(Xtr)
    f_ca = base_function(Xca)
    f_te = base_function(Xte)

    sig_tr = sigma_fn(Xtr)
    sig_ca = sigma_fn(Xca)
    sig_te = sigma_fn(Xte)

    ytr = f_tr + rng.normal(0.0, sig_tr)
    yca = f_ca + rng.normal(0.0, sig_ca)
    yte = f_te + rng.normal(0.0, sig_te)

    if holes:
        Xtr, ytr = remove_holes(Xtr, ytr, holes)

    return {
        "X_train": Xtr,
        "y_train": ytr,
        "X_cal": Xca,
        "y_cal": yca,
        "X_test": Xte,
        "y_test": yte,
        "sigma_train": sig_tr,
        "sigma_cal": sig_ca,
        "sigma_test": sig_te,
        "f_test": f_te,
    }

