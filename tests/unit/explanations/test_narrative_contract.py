"""Minimal narrative facts preserve the existing explanation contract."""

from __future__ import annotations

import pytest

from calibrated_explanations.explanations import build_narrative_spec
from calibrated_explanations.utils.exceptions import ValidationError


def _payload():
    return {
        "schema_version": "1.0.0", "task": "classification",
        "explanation_type": "factual",
        "prediction": {"predict": 0.7, "low": 0.6, "high": 0.8},
        "rules": [
            {"rule": "first", "rule_weight": {"predict": 0.2, "low": 0.1, "high": 0.3}},
            {"rule": "second", "rule_weight": {"predict": -0.1, "low": -0.2, "high": 0.0}},
        ],
        "metadata": {"narrative_context": {"class_labels": ["no", "yes"]}},
    }


def test_build_is_deterministic_with_stable_source_references():
    source = _payload()
    first = build_narrative_spec(source)
    second = build_narrative_spec(dict(reversed(list(source.items()))))
    assert first == second
    assert first.to_dict()["rules"] == source["rules"]
    assert first.to_dict()["prediction"] == source["prediction"]
    assert first.to_dict()["narrative_context"]["class_labels"] == ["no", "yes"]


def test_absent_context_is_valid_and_does_not_create_new_semantics():
    source = _payload()
    source.pop("metadata")
    assert build_narrative_spec(source).to_dict()["narrative_context"] is None


@pytest.mark.parametrize("bounds", [
    {"predict": 0.7, "low": 0.8, "high": 0.9},
    {"predict": 0.7, "low": 0.1, "high": float("nan")},
    {"predict": 0.7, "low": 0.1, "high": float("inf")},
])
def test_invalid_intervals_fail_closed(bounds):
    source = _payload()
    source["prediction"] = bounds
    with pytest.raises(ValidationError):
        build_narrative_spec(source)


def test_bad_context_fails():
    source = _payload()
    source["metadata"]["narrative_context"] = "guessed"
    with pytest.raises(ValidationError):
        build_narrative_spec(source)
