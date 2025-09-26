from __future__ import annotations

"""Stump-on-proba baseline (depth-1 tree on probabilities; no plugins).

Trains a DecisionTreeRegressor with max_depth=1 on the model’s calibrated
probabilities (binary classification). For each instance, the single split
defines the rule. Weight proxy and interval proxy are computed as in the
surrogate baseline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + (z * z) / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class Stump:
    est: DecisionTreeRegressor
    feature_names: List[str]

    def predict_leaf_values(self, X: np.ndarray) -> np.ndarray:
        p = self.est.predict(X)
        return np.clip(p, 0.0, 1.0)

    def leaf_indices(self, X: np.ndarray) -> np.ndarray:
        return self.est.apply(X)


def fit_baseline(model: Any, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Stump:
    max_depth = 1
    proba = model.predict_proba(X_train)
    p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.asarray(proba).reshape(-1)
    est = DecisionTreeRegressor(max_depth=max_depth, random_state=kwargs.get("seed", 42))
    est.fit(X_train, p1)
    names = [str(i) for i in range(X_train.shape[1])]
    try:
        names = list(getattr(model, "feature_names_in_", names))
    except Exception:
        pass
    return Stump(est=est, feature_names=names)


def explain_rules(
    baseline: Stump,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_names: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    tree = baseline.est.tree_
    feat_names = list(feature_names) if feature_names is not None else baseline.feature_names

    # Identify root split
    root = 0
    fid = int(tree.feature[root])
    thr = float(tree.threshold[root])
    fname = feat_names[fid] if 0 <= fid < len(feat_names) else str(fid)

    # Compute leaf stats using calibration labels
    leaf_idx_cal = baseline.leaf_indices(X_cal)
    uniq = np.unique(leaf_idx_cal)
    stats: Dict[int, Tuple[int, int, float, float, float]] = {}
    for nid in uniq:
        m = leaf_idx_cal == nid
        n = int(np.sum(m))
        k = int(np.sum(y_cal[m] == 1))
        p_hat = k / n if n > 0 else float("nan")
        lo, hi = _wilson_interval(k, n) if n > 0 else (float("nan"), float("nan"))
        stats[int(nid)] = (n, k, p_hat, lo, hi)

    out: List[Dict[str, Any]] = []
    leaves_test = baseline.leaf_indices(X_test)
    parent_val = tree.value[root][0][0] / tree.weighted_n_node_samples[root] if tree.weighted_n_node_samples[root] > 0 else 0.5
    for i, x in enumerate(X_test):
        leaf_val = baseline.predict_leaf_values(x.reshape(1, -1))[0]
        y_true = int(y_test[i])
        y_pred_base = int(leaf_val >= 0.5)
        # Determine side and rule
        if x[fid] <= thr:
            antecedent = f"{fname} <= {thr:.4g}"
        else:
            antecedent = f"{fname} > {thr:.4g}"
        rule_id = _md5(antecedent)
        node_id = int(leaves_test[i])
        n, k, p_hat, lo, hi = stats.get(node_id, (0, 0, float("nan"), float("nan"), float("nan")))
        w = leaf_val - parent_val
        out.append(
            {
                "baseline": "stump_on_proba",
                "point_id": i,
                "rule_rank": 0,
                "rule_id": rule_id,
                "antecedent_str": antecedent,
                "w": float(w) if not math.isnan(w) else None,
                "w_low": float(lo) if not math.isnan(lo) else None,
                "w_high": float(hi) if not math.isnan(hi) else None,
                "w_width": (float(hi - lo) if (not math.isnan(lo) and not math.isnan(hi)) else None),
                "p_pred_base": float(leaf_val),
                "y_true": y_true,
                "y_pred_base": y_pred_base,
            }
        )

    return out

