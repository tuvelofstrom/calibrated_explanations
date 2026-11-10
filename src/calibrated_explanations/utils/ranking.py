"""Canonical rule ranking for factual, FAST and alternative CE explanations.

ADR-042 D1-D9: a single authority for scores, deterministic tie-breaking,
non-finite ordering, and top-k selection. Never mutate predicted values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from .exceptions import ValidationError


def canonical_rank_indices(
    primary: Sequence[float],
    widths: Sequence[float] | None = None,
    *,
    top_k: int | None = None,
) -> list[int]:
    """Return native indices strongest first, by score, narrow width and index.

    Parameters
    ----------
    primary : sequence of float
        Larger primary values rank earlier. NaN scores always rank last.
    widths : sequence of float, optional
        Smaller interval widths win score ties. Non-finite widths lose ties.
    top_k : int or None, default=None
        Limit to the strongest k rules; None returns all, zero returns none.

    Returns
    -------
    list of int
        Stable strongest-first indices.

    Raises
    ------
    ValidationError
        If k is invalid or the arrays are misaligned.
    """
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0
    ):
        raise ValidationError(
            "top_k must be a nonnegative integer or None",
            details={"param": "top_k", "value": repr(top_k)},
        )
    count = len(primary)
    if widths is not None and len(widths) != count:
        raise ValidationError(
            "Ranking scores and widths must be aligned",
            details={"scores": count, "widths": len(widths)},
        )

    def key(index: int) -> tuple[bool, float, float, int]:
        score = float(primary[index])
        width = math.inf if widths is None else float(widths[index])
        if not math.isfinite(width):
            width = math.inf
        return (math.isnan(score), 0.0 if math.isnan(score) else -score, width, index)

    order = sorted(range(count), key=key)
    return order if top_k is None else order[: int(top_k)]


def rank_rule_indices(
    rules: Mapping[str, Sequence[Any]],
    *,
    explanation_type: str,
    base_predict: float | None = None,
    probability: bool = True,
    rnk_metric: str | None = None,
    rnk_weight: float = 0.5,
    top_k: int | None = None,
) -> list[int]:
    """Rank a CE native rule table without changing any values or its order.

    Factual and FAST use magnitude of calibrated feature weight. Alternative
    default uses the ensured score of its predicted result and interval.
    Classification and thresholded regression orient probabilities to the
    original predicted class; plain regression uses the original target units.

    Parameters
    ----------
    rules : mapping of str to sequences
        Native rule columns for weights and predictions, with lower/upper bounds.
    explanation_type : str
        One of factual, fast, or alternative.
    base_predict : float, optional
        Original probability, needed for probability alternative scores.
    probability : bool, default=True
        False selects raw regression predictions rather than probability orientation.
    rnk_metric : str or None, optional
        Alternative ensured, feature_weight, uncertainty; only feature_weight for factual.
    rnk_weight : float, default=0.5
        Ensured trade-off in [-1, 1]; negative weights favour counter-alternatives.
    top_k : int or None, optional
        Restrict selection to a strongest-first prefix.

    Returns
    -------
    list of int
        Native indices of ranked rules.

    Raises
    ------
    ValidationError
        For invalid parameters or misaligned rule-table columns.
    """
    if explanation_type not in ("factual", "fast", "alternative"):
        raise ValidationError(
            "Unknown explanation type for ranking",
            details={"explanation_type": explanation_type},
        )
    weights = rules.get("weight", ())
    n_rules = len(weights)

    def array(name: str) -> Sequence[Any]:
        values = rules.get(name, ())
        if len(values) != n_rules:
            raise ValidationError(
                "Rule ranking requires aligned native rule columns",
                details={"field": name, "expected": n_rules, "received": len(values)},
            )
        return values

    if explanation_type != "alternative":
        if rnk_metric not in (None, "feature_weight"):
            raise ValidationError(
                "Factual and FAST explanations rank by feature_weight only",
                details={"param": "rnk_metric", "value": rnk_metric},
            )
        lows, highs = array("weight_low"), array("weight_high")
        scores = [abs(float(w)) for w in weights]
        widths = [abs(float(hi) - float(lo)) for lo, hi in zip(lows, highs)]
        return canonical_rank_indices(scores, widths, top_k=top_k)

    metric = "ensured" if rnk_metric is None else rnk_metric
    if metric not in ("ensured", "feature_weight", "uncertainty"):
        raise ValidationError(
            "Unknown alternative ranking metric",
            details={"param": "rnk_metric", "value": metric},
        )
    lows, highs = array("predict_low"), array("predict_high")
    widths = [float(hi) - float(lo) for lo, hi in zip(lows, highs)]
    if metric == "feature_weight":
        return canonical_rank_indices([abs(float(w)) for w in weights], widths, top_k=top_k)

    weight = 0.0 if metric == "uncertainty" else float(rnk_weight)
    if not math.isfinite(weight) or not -1.0 <= weight <= 1.0:
        raise ValidationError(
            "rnk_weight must be finite and in [-1, 1]",
            details={"param": "rnk_weight", "value": repr(rnk_weight)},
        )
    if probability:
        if base_predict is None:
            raise ValidationError(
                "Probability alternative ranking requires base_predict",
                details={"param": "base_predict"},
            )
        orient_positive = float(base_predict) > 0.5
    else:
        orient_positive = True

    amplitude = abs(weight)
    scores: list[float] = []
    for pred, width in zip(array("predict"), widths):
        p = float(pred)
        if not orient_positive:
            p = 1.0 - p
        if weight < 0:
            p = -p
        # In an open interval, avoid 0 * infinity poisoning an unused term.
        if amplitude == 1.0:
            score = p
        elif amplitude == 0.0:
            score = 1.0 - width
        else:
            score = (1.0 - amplitude) * (1.0 - width) + amplitude * p
        scores.append(score)
    return canonical_rank_indices(scores, widths, top_k=top_k)
