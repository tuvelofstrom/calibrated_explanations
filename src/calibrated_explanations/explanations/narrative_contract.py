"""Passive, deterministic narrative facts from the documented explanation JSON.

No prose generation, provider dispatch or additional prediction is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from ..utils.exceptions import ValidationError


def _canonical(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("Narrative source must be finite JSON data") from exc


def _check_interval(value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    if not {"low", "predict", "high"}.issubset(value):
        return
    low, middle, high = (value[k] for k in ("low", "predict", "high"))
    if any(isinstance(v, bool) for v in (low, middle, high)):
        raise ValidationError("Boolean prediction bound")
    if any(isinstance(v, (list, tuple)) for v in (low, middle, high)):
        if not all(isinstance(v, (list, tuple)) for v in (low, middle, high)):
            raise ValidationError("Inconsistent interval dimensions")
        if not (len(low) == len(middle) == len(high)):
            raise ValidationError("Inconsistent interval dimensions")
        for triple in zip(low, middle, high):
            _check_interval(dict(zip(("low", "predict", "high"), triple)))
        return
    if all(isinstance(v, (int, float)) for v in (low, middle, high)):
        if not (math.isfinite(low) and math.isfinite(middle) and math.isfinite(high)):
            raise ValidationError("Non-finite prediction bound")
        if not low <= middle <= high:
            raise ValidationError("Prediction violates interval bounds")


@dataclass(frozen=True)
class NarrativeSpec:
    """Versioned, content-addressed semantic snapshot, independent of formatting."""

    version: str
    source_id: str
    facts_json: str

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable source facts as a fresh JSON object."""
        return json.loads(self.facts_json)


def build_narrative_spec(payload: Mapping[str, Any]) -> NarrativeSpec:
    """Snapshot source-bound facts from the documented ADR-005 JSON payload.

    This function preserves established facts and their original order; it does
    not infer absent class, threshold, reject or calibration semantics. Inputs
    should be generated via documented explanation serialization.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError("Explanation payload must be a mapping")
    required = {"task", "explanation_type", "prediction", "rules"}
    if not required.issubset(payload):
        raise ValidationError("Missing required explanation fields")
    if payload["explanation_type"] not in {"factual", "alternative", "fast"}:
        raise ValidationError("Unsupported explanation type")
    prediction = payload["prediction"]
    rules = payload["rules"]
    if not isinstance(prediction, Mapping) or not isinstance(rules, list):
        raise ValidationError("Invalid explanation prediction or rules")
    _check_interval(prediction)
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ValidationError("Invalid explanation rule")
        _check_interval(rule.get("rule_prediction"))
        _check_interval(rule.get("rule_weight"))

    meta = payload.get("metadata")
    context = meta.get("narrative_context") if isinstance(meta, Mapping) else None
    if context is not None and not isinstance(context, Mapping):
        raise ValidationError("Invalid narrative context")
    facts = {
        "source_schema_version": payload.get("schema_version", "1.0.0"),
        "task": payload["task"],
        "explanation_type": payload["explanation_type"],
        "prediction": prediction,
        "rules": rules,
        "narrative_context": context,
    }
    canonical = _canonical(facts)
    return NarrativeSpec(
        version="1",
        source_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        facts_json=canonical,
    )
