from __future__ import annotations

import numpy as np
from typing import Callable

try:
    # Prefer in-repo VennAbers to align with CE
    from calibrated_explanations._VennAbers import VennAbers
except Exception:  # pragma: no cover - fallback to external package if available
    VennAbers = None  # type: ignore


def fit_venn_abers(learner, X_cal: np.ndarray, y_cal: np.ndarray, predict_function: Callable | None = None):
    if VennAbers is None:
        raise ImportError("VennAbers not available. Ensure 'calibrated_explanations' is installed.")
    va = VennAbers(X_cal, y_cal, learner, difficulty_estimator=None, predict_function=predict_function)
    return va


def predict_calibrated(va, X: np.ndarray) -> np.ndarray:
    # Returns calibrated probabilities for binary or multiclass
    return va.predict(X)

