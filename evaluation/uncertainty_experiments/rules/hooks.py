from __future__ import annotations

"""Helpers to extract local CE factual rules (per-rule weights with uncertainty).

This module focuses on M2.1 plumbing for factual explanations only.

Primary entry point:
- extract_factual_rule_records_for_classification(...): list[dict]

It calibrates a WrapCalibratedExplainer on the provided learner and
returns one record per (instance, rule) pair, with a deterministic rule_id.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import hashlib

import numpy as np

try:  # local import guarded to avoid hard dependency when unused
    from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
    from calibrated_explanations.explanations.explanation import FactualExplanation
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
except Exception as _exc:  # pragma: no cover - defensive
    WrapCalibratedExplainer = None  # type: ignore
    FactualExplanation = None  # type: ignore
    CalibratedExplanations = None  # type: ignore


def _stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _rank_indices_by_weight(exp: FactualExplanation, rules: Dict[str, Any]) -> List[int]:
    """Return rule indices ordered by importance (descending).

    Uses the same logic as plotting: rank by absolute weight (with width tiebreak).
    """
    weights = np.asarray(rules["weight"])
    width = np.asarray(rules["weight_high"]) - np.asarray(rules["weight_low"])
    # exp._rank_features returns ascending; reverse for descending
    order = list(exp._rank_features(feature_weights=weights, width=width, num_to_show=len(weights)))  # noqa: SLF001
    order.reverse()
    return order


def _predicted_class_from_p(p: float) -> int:
    try:
        return int(p >= 0.5)
    except Exception:
        return 0


def extract_factual_rule_records_for_classification(
    learner: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    de_test: Optional[np.ndarray] = None,
    inv_density_test: Optional[np.ndarray] = None,
    eta_test: Optional[np.ndarray] = None,
    feature_names: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract per-instance factual rule records with calibrated p_rule (classification).

    Parameters
    ----------
    learner: fitted sklearn-like estimator
    X_cal, y_cal: calibration data used by CE
    X_test, y_test: test data and ground-truth labels
    de_test: optional Difficulty Estimator values aligned with X_test
    inv_density_test: optional inverse-density proxy aligned with X_test
    eta_test: optional aleatoric proxy (classification overlap) aligned with X_test
    feature_names: optional list of feature names to attach for readability
    seed: CE seed for discretization repeatability

    Returns
    -------
    list of dicts, one per (point_id, rule) with fields:
      {point_id, explanation_type, rule_rank, rule_id, antecedent_str,
       p_rule, y_true, y_pred, de, inv_density, eta}
    """
    if WrapCalibratedExplainer is None:  # pragma: no cover - defensive
        raise RuntimeError("calibrated_explanations is not available")

    w = WrapCalibratedExplainer(learner)
    w.calibrate(X_cal, y_cal, seed=seed, feature_names=list(feature_names) if feature_names else None)
    ce: CalibratedExplanations = w.explain_factual(X_test)

    # Helper to fetch aligned uncertainty components safely
    def _safe_get(arr: Optional[np.ndarray], i: int) -> Optional[float]:
        if arr is None:
            return None
        try:
            return float(arr[i])
        except Exception:
            return None

    out: List[Dict[str, Any]] = []
    for i, exp in enumerate(ce.explanations):
        # Ground truth and model-level prediction for point i
        y_true = int(y_test[i]) if y_test is not None else None
        # Base calibrated probability (binary assumed here)
        y_pred_prob = float(exp.prediction.get("predict"))
        y_pred = _predicted_class_from_p(y_pred_prob)

        rules = exp._get_rules()  # noqa: SLF001 (internal but stable)
        if len(rules.get("rule", [])) == 0:
            continue  # no rules to record

        order = _rank_indices_by_weight(exp, rules)
        for rank_pos, idx in enumerate(order):
            feature_index = int(rules["feature"][idx])
            antecedent_str = str(rules["rule"][idx]).strip()
            rule_id = _stable_hash(antecedent_str)
            # Weight-based uncertainty (feature effect)
            w = float(rules["weight"][idx])
            w_low = float(rules["weight_low"][idx])
            w_high = float(rules["weight_high"][idx])
            # Rule-level counterfactual summary used to derive weight
            try:
                rule_predict = float(rules["predict"][idx])
            except Exception:
                rule_predict = None
            try:
                rule_predict_low = float(rules["predict_low"][idx])
            except Exception:
                rule_predict_low = None
            try:
                rule_predict_high = float(rules["predict_high"][idx])
            except Exception:
                rule_predict_high = None

            out.append(
                {
                    "point_id": i,
                    "explanation_type": "factual",
                    "rule_rank": rank_pos,
                    "rule_id": rule_id,
                    "feature_index": feature_index,
                    "feature_name": (list(ce.feature_names)[feature_index] if ce.feature_names else str(feature_index)),
                    "antecedent_str": antecedent_str,
                    "p_pred": y_pred_prob,
                    "w": w,
                    "w_low": w_low,
                    "w_high": w_high,
                    "w_width": float(w_high - w_low),
                    "rule_predict": rule_predict,
                    "rule_predict_low": rule_predict_low,
                    "rule_predict_high": rule_predict_high,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "de": _safe_get(de_test, i),
                    "inv_density": _safe_get(inv_density_test, i),
                    "eta": _safe_get(eta_test, i),
                }
            )

    return out


def extract_alternative_rule_records_for_classification(
    learner: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_names: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract per-instance alternative rule records using CE explore_alternatives.

    Records are aligned with the factual schema but tagged explanation_type='alternative'.
    Includes: rule_id, antecedent_str, weight (effect) and uncertainty, base prediction,
    and labels.
    """
    if WrapCalibratedExplainer is None:  # pragma: no cover - defensive
        raise RuntimeError("calibrated_explanations is not available")

    w = WrapCalibratedExplainer(learner)
    w.calibrate(X_cal, y_cal, seed=seed, feature_names=list(feature_names) if feature_names else None)
    ce_alt = w.explore_alternatives(X_test)

    out: List[Dict[str, Any]] = []
    for i, exp in enumerate(ce_alt.explanations):
        # Base prediction (from exp.prediction) and true label
        try:
            p_base = float(exp.prediction.get("predict"))
        except Exception:
            p_base = None
        y_true = int(y_test[i]) if y_test is not None else None

        # Alternative rules
        rules = exp._get_rules()  # noqa: SLF001
        if not rules.get("rule"):
            continue
        # Rank by absolute weight (same as plot convention)
        weights = np.asarray(rules.get("weight", []))
        order = list(np.argsort(np.abs(weights))) if weights.size > 0 else list(range(len(rules.get("rule", []))))
        order = list(reversed(order))
        for rank_pos, idx in enumerate(order):
            antecedent_str = str(rules["rule"][idx]).strip()
            rule_id = _stable_hash(antecedent_str)
            try:
                wv = float(rules["weight"][idx])
                w_low = float(rules["weight_low"][idx])
                w_high = float(rules["weight_high"][idx])
            except Exception:
                wv = w_low = w_high = None
            w_width = (float(w_high - w_low) if (w_low is not None and w_high is not None) else None)
            out.append(
                {
                    "point_id": i,
                    "explanation_type": "alternative",
                    "rule_rank": rank_pos,
                    "rule_id": rule_id,
                    "feature_index": None,
                    "feature_name": None,
                    "antecedent_str": antecedent_str,
                    "p_pred": p_base,
                    "w": wv,
                    "w_low": w_low,
                    "w_high": w_high,
                    "w_width": w_width,
                    "y_true": y_true,
                    "y_pred": (int(p_base >= 0.5) if p_base is not None else None),
                }
            )

    return out


def extract_factual_rule_records_for_thresholded_regression(
    learner: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    t_values: Iterable[float],
    *,
    de_test: Optional[np.ndarray] = None,
    inv_density_test: Optional[np.ndarray] = None,
    sigma_test: Optional[np.ndarray] = None,
    feature_names: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract per-instance factual rule records for thresholded regression over t_values.

    For each threshold t in t_values, compute P(Y>t | rule) with uncertainty and add to the record.
    """
    if WrapCalibratedExplainer is None:  # pragma: no cover - defensive
        raise RuntimeError("calibrated_explanations is not available")

    w = WrapCalibratedExplainer(learner)
    w.calibrate(X_cal, y_cal, seed=seed, feature_names=list(feature_names) if feature_names else None)

    # Pre-build per-threshold explanations to avoid recomputing per-rule later
    exps_by_t: Dict[str, Any] = {}
    for t in t_values:
        exp_set = w.explain_factual(X_test, threshold=float(t))
        key = f"t={float(t):.4g}"
        exps_by_t[key] = exp_set

    # Helper to fetch aligned uncertainty components safely
    def _safe_get(arr: Optional[np.ndarray], i: int) -> Optional[float]:
        if arr is None:
            return None
        try:
            return float(arr[i])
        except Exception:
            return None

    out: List[Dict[str, Any]] = []
    n = len(X_test)
    # We iterate using one baseline explanation set (any t) for rule structure and weights
    # Choose the first t as reference for ranking and rule list; realizations are consistent across t
    any_key = next(iter(exps_by_t.keys()))
    ce_ref = exps_by_t[any_key]

    for i, exp in enumerate(ce_ref.explanations):
        rules = exp._get_rules()  # noqa: SLF001
        if len(rules.get("rule", [])) == 0:
            continue
        order = _rank_indices_by_weight(exp, rules)

        # Instance-level thresholded probabilities per t for provenance
        p_pred_t: Dict[str, float] = {}
        for key, ce_set in exps_by_t.items():
            e_i = ce_set.explanations[i]
            p_pred_t[key] = float(e_i.prediction.get("predict"))

        for rank_pos, idx in enumerate(order):
            feature_index = int(rules["feature"][idx])
            antecedent_str = str(rules["rule"][idx]).strip()
            rule_id = _stable_hash(antecedent_str)

            # Weight uncertainty (same across t)
            wv = float(rules["weight"][idx])
            w_low = float(rules["weight_low"][idx])
            w_high = float(rules["weight_high"][idx])

            # Collect per-threshold rule probabilities and intervals for this feature
            p_rule_t: Dict[str, float] = {}
            p_rule_t_low: Dict[str, float] = {}
            p_rule_t_high: Dict[str, float] = {}
            p_rule_t_width: Dict[str, float] = {}
            for key, ce_set in exps_by_t.items():
                e_i = ce_set.explanations[i]
                p = _p_rule_for_feature(e_i, feature_index)
                lo, hi = _p_rule_interval_for_feature(e_i, feature_index)
                if p is not None:
                    p_rule_t[key] = float(p)
                if lo is not None:
                    p_rule_t_low[key] = float(lo)
                if hi is not None:
                    p_rule_t_high[key] = float(hi)
                if lo is not None and hi is not None:
                    p_rule_t_width[key] = float(hi - lo)

            # Support in reference discretization (consistent across t)
            cc, total, frac = _support_for_feature(exp, feature_index)

            out.append(
                {
                    "point_id": i,
                    "explanation_type": "factual",
                    "rule_rank": rank_pos,
                    "rule_id": rule_id,
                    "feature_index": feature_index,
                    "feature_name": (list(ce_ref.feature_names)[feature_index] if ce_ref.feature_names else str(feature_index)),
                    "antecedent_str": antecedent_str,
                    # weight (effect) uncertainty
                    "w": wv,
                    "w_low": w_low,
                    "w_high": w_high,
                    "w_width": float(w_high - w_low),
                    # per-threshold calibrated rule probabilities and intervals
                    "p_rule_t": p_rule_t,
                    "p_rule_t_low": p_rule_t_low,
                    "p_rule_t_high": p_rule_t_high,
                    "p_rule_t_width": p_rule_t_width,
                    # instance-level per-threshold base probabilities
                    "p_pred_t": p_pred_t,
                    # labels and covariates
                    "y_true": float(y_test[i]) if y_test is not None else None,
                    "de": _safe_get(de_test, i),
                    "inv_density": _safe_get(inv_density_test, i),
                    "sigma": _safe_get(sigma_test, i),
                }
            )

    return out
