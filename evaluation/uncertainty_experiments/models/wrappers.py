from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class ModelWrapper:
    model: Any
    task: str  # 'classification' or 'regression'

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_many(self, X: np.ndarray, n_samples: int = 20, seed: int = 42) -> np.ndarray:
        """For models with bagging/ensembles, approximate disagreement via resampling trees.
        If not available, returns a single prediction repeated.
        """
        rng = np.random.default_rng(seed)
        if isinstance(self.model, (RandomForestClassifier, RandomForestRegressor)):
            preds = []
            trees = self.model.estimators_
            for _ in range(n_samples):
                idx = rng.choice(len(trees), size=len(trees), replace=True)
                if self.task == "classification":
                    # average probabilities across sampled trees
                    ps = [trees[i].predict_proba(X) for i in idx]
                    preds.append(np.mean(ps, axis=0))
                else:
                    ys = [trees[i].predict(X) for i in idx]
                    preds.append(np.mean(ys, axis=0))
            return np.stack(preds, axis=0)
        # Fallback
        p = self.predict(X)
        return np.repeat(p[np.newaxis, ...], n_samples, axis=0)


def make_model(task: str, mtype: str, params: Dict) -> ModelWrapper:
    if task == "classification":
        if mtype == "dt":
            model = DecisionTreeClassifier(**{k: v for k, v in params.items() if v is not None})
        elif mtype == "rf":
            model = RandomForestClassifier(**{k: v for k, v in params.items() if v is not None})
        else:
            raise ValueError("Unknown model type for classification: %s" % mtype)
    else:
        if mtype == "dt":
            model = DecisionTreeRegressor(**{k: v for k, v in params.items() if v is not None})
        elif mtype == "rf":
            model = RandomForestRegressor(**{k: v for k, v in params.items() if v is not None})
        else:
            raise ValueError("Unknown model type for regression: %s" % mtype)
    return ModelWrapper(model=model, task=task)

