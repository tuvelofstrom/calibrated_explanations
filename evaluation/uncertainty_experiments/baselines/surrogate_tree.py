from __future__ import annotations

"""Surrogate shallow tree baseline (no plugins, pure sklearn).

Trains a shallow DecisionTreeRegressor on the model’s calibrated probabilities
(binary classification). For each test instance, extracts the decision path as
a conjunction of predicates and reports a simple rule weight proxy and an
interval proxy based on calibration set label proportions at the leaf.

Exports two convenience functions:
- fit_baseline(model, X_train, y_train, **kwargs) -> baseline
- explain_rules(baseline, X_cal, y_cal, X_test, y_test, feature_names=None) -> list[dict]

Record schema (aligned with CE’s rules where feasible):
  {
    'baseline': 'surrogate_tree',
    'point_id': int,
    'rule_rank': 0,
    'rule_id': md5(canonical antecedent_str),
    'antecedent_str': str,
    'w': float | None,               # leaf_p - parent_p (signed)
    'w_low': float | None,           # Wilson interval (leaf proportion)
    'w_high': float | None,
    'w_width': float | None,
    'p_pred_base': float,            # leaf probability prediction
    'y_true': int,
    'y_pred_base': int,              # 1 if p_pred_base>=0.5 else 0
  }
Optional covariates (inv_density, eta) can be attached by the caller/runner.
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


def _format_threshold(t: float) -> str:
    return f"{t:.4g}"  # compact


def _path_antecedent(tree, feature_names: List[str], x: np.ndarray) -> Tuple[str, List[Tuple[int, str, float]]]:
    """Build conjunction of predicates for a single instance path.

    Returns antecedent string and a list of (feature_index, op, threshold).
    """
    node = 0
    clauses: List[str] = []
    triples: List[Tuple[int, str, float]] = []
    while tree.children_left[node] != tree.children_right[node]:
        fid = tree.feature[node]
        thr = float(tree.threshold[node])
        fname = feature_names[fid] if 0 <= fid < len(feature_names) else str(fid)
        if x[fid] <= thr:
            op = "<="
            clauses.append(f"{fname} <= {_format_threshold(thr)}")
            triples.append((fid, op, thr))
            node = tree.children_left[node]
        else:
            op = ">"
            clauses.append(f"{fname} > {_format_threshold(thr)}")
            triples.append((fid, op, thr))
            node = tree.children_right[node]
    return " and ".join(clauses) if clauses else "TRUE", triples


@dataclass
class SurrogateTree:
    est: DecisionTreeRegressor
    feature_names: List[str]

    def predict_leaf_values(self, X: np.ndarray) -> np.ndarray:
        """Return leaf value for each instance, clipped to [0,1]."""
        p = self.est.predict(X)
        return np.clip(p, 0.0, 1.0)

    def leaf_indices(self, X: np.ndarray) -> np.ndarray:
        return self.est.apply(X)


def fit_baseline(model: Any, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> SurrogateTree:
    """Fit a surrogate regressor tree on the model’s calibrated probabilities.

    - model: fitted estimator with predict_proba(X) for classification
    - X_train, y_train: used only to derive targets from model
    - kwargs: max_depth (int, default=3)
    """
    max_depth = int(kwargs.get("max_depth", 3))
    # Targets: probability of class 1
    proba = model.predict_proba(X_train)
    p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.asarray(proba).reshape(-1)
    est = DecisionTreeRegressor(max_depth=max_depth, random_state=kwargs.get("seed", 42))
    est.fit(X_train, p1)
    # Feature names best-effort
    names = [str(i) for i in range(X_train.shape[1])]
    try:
        names = list(getattr(model, "feature_names_in_", names))
    except Exception:
        pass
    return SurrogateTree(est=est, feature_names=names)


def explain_rules(
    baseline: SurrogateTree,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_names: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract one rule per instance from the surrogate tree.

    - Weight proxy w = leaf_p - parent_p (signed). Parent_p approximated by the
      prediction at the parent node encountered last along the path.
    - Interval proxy from Wilson interval at the leaf using (k = #positives in cal leaf, n = leaf count).
    """
    tree = baseline.est.tree_
    feat_names = list(feature_names) if feature_names is not None else baseline.feature_names

    # Precompute leaf membership on calibration set
    leaf_idx_cal = baseline.leaf_indices(X_cal)
    uniq_leaves = np.unique(leaf_idx_cal)
    # Map node_id -> (n, k, p_hat, low, high)
    leaf_stats: Dict[int, Tuple[int, int, float, float, float]] = {}
    for nid in uniq_leaves:
        mask = leaf_idx_cal == nid
        n = int(np.sum(mask))
        if n <= 0:
            leaf_stats[nid] = (0, 0, float("nan"), float("nan"), float("nan"))
            continue
        k = int(np.sum(y_cal[mask] == 1))
        p_hat = k / n
        lo, hi = _wilson_interval(k, n)
        leaf_stats[nid] = (n, k, p_hat, lo, hi)

    # Helper to get parent prediction along path (approx: walk and store prev node value)
    def _parent_pred(x: np.ndarray) -> Tuple[float, float, str]:
        node = 0
        prev_val = tree.value[node][0][0] / tree.weighted_n_node_samples[node] if tree.weighted_n_node_samples[node] > 0 else 0.5
        # travel until leaf
        while tree.children_left[node] != tree.children_right[node]:
            fid = tree.feature[node]
            thr = float(tree.threshold[node])
            val_here = tree.value[node][0][0] / tree.weighted_n_node_samples[node] if tree.weighted_n_node_samples[node] > 0 else prev_val
            if x[fid] <= thr:
                prev_val = val_here
                node = tree.children_left[node]
            else:
                prev_val = val_here
                node = tree.children_right[node]
        leaf_node = node
        leaf_val = baseline.predict_leaf_values(x.reshape(1, -1))[0]
        antecedent, _ = _path_antecedent(tree, feat_names, x)
        return float(prev_val), float(leaf_val), antecedent

    out: List[Dict[str, Any]] = []
    leaf_idx_test = baseline.leaf_indices(X_test)

    for i, x in enumerate(X_test):
        parent_p, leaf_p, antecedent = _parent_pred(x)
        w = leaf_p - parent_p
        y_true = int(y_test[i])
        y_pred_base = int(leaf_p >= 0.5)
        node_id = leaf_idx_test[i]
        n, k, p_hat, lo, hi = leaf_stats.get(int(node_id), (0, 0, float("nan"), float("nan"), float("nan")))
        rule_id = _md5(antecedent)
        out.append(
            {
                "baseline": "surrogate_tree",
                "point_id": i,
                "rule_rank": 0,
                "rule_id": rule_id,
                "antecedent_str": antecedent,
                "w": float(w) if not math.isnan(w) else None,
                "w_low": float(lo) if not math.isnan(lo) else None,
                "w_high": float(hi) if not math.isnan(hi) else None,
                "w_width": (float(hi - lo) if (not math.isnan(lo) and not math.isnan(hi)) else None),
                "p_pred_base": float(leaf_p),
                "y_true": y_true,
                "y_pred_base": y_pred_base,
            }
        )

    return out

