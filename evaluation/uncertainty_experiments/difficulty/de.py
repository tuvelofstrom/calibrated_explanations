from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class CompositeDE:
    """Composite Difficulty Estimator.

    Expected to be compatible with CalibratedExplainer: has `.fitted` and `.apply(X)`.
    Here we store per-point components computed on a reference set X_ref, and apply returns
    the same-scale difficulty for an input X by nearest-neighbor lookup.
    For the initial sandbox, we use an identity mapping for X==X_ref.
    """

    weights: Dict[str, float]
    X_ref: np.ndarray
    components: Dict[str, np.ndarray]
    fitted: bool = True

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        s = np.std(v) + 1e-12
        m = np.mean(v)
        return (v - m) / s

    def score(self) -> np.ndarray:
        total = np.zeros(len(self.X_ref))
        for name, w in self.weights.items():
            if name not in self.components:
                continue
            total = total + float(w) * self._normalize(self.components[name])
        # Shift to positive range
        total = (total - total.min()) / (total.max() - total.min() + 1e-12)
        return total

    def apply(self, X: np.ndarray) -> np.ndarray:
        # For now, if X matches X_ref length and content, return computed scores.
        # Otherwise, fall back to nearest-neighbor transfer.
        if X.shape == self.X_ref.shape and np.allclose(X, self.X_ref):
            return self.score()
        # NN transfer
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.X_ref)
            _, idx = nn.kneighbors(X)
            return self.score()[idx[:, 0]]
        except Exception:
            return np.full(len(X), float(np.mean(self.score())))

