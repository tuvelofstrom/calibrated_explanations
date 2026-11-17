"""Tests for the plotting payload builder used by MultiClassCalibratedExplanations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations


class SimpleExp:
    def __init__(self, idx: int, cls: int):
        self.index = idx
        self.class_id = cls

    def get_rules(self) -> Dict[str, Any]:
        # minimal factual dict shape expected by the payload builder
        return {
            "base_predict": [0.6, 0.6],
            "base_predict_low": [0.55, 0.55],
            "base_predict_high": [0.65, 0.65],
            "weight": [0.1, 0.2],
            "weight_low": [0.05, 0.15],
            "weight_high": [0.15, 0.25],
            "predict": [0.6, 0.4],
            "predict_low": [0.55, 0.35],
            "predict_high": [0.65, 0.45],
            "value": ["a", "b"],
            "rule": ["r1", "r2"],
            "feature": [0, 1],
            "feature_value": [None, None],
            "is_conjunctive": [False, False],
            "classes": [self.class_id, self.class_id],
        }

    def _rank_features(self, *args, **kwargs):
        # simple rank implementation for tests: return indices up to num_to_show
        num = (
            kwargs.get("num_to_show") if "num_to_show" in kwargs else (len(args[0]) if args else 0)
        )
        try:
            num = int(num)
        except Exception:
            num = 0
        return list(range(min(num, len(args[0]) if args else 0)))


def test_plot_factual_payload_via_public_api(monkeypatch):
    pytest.importorskip("matplotlib")
    # Build a small multiclass collection with a single instance containing two class explanations
    x = np.array([[1.0]])
    explanations = [
        {0: SimpleExp(0, 0), 1: SimpleExp(0, 1)},
    ]

    class DummyExplainer:
        def __init__(self, num_features):
            self.num_features = num_features

    expl = DummyExplainer(x.shape[1])
    coll = MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=2, explanations=explanations
    )

    calls = []

    def fake_plot_probabilistic(*args, **kwargs):
        calls.append((args, kwargs))

    # Monkeypatch the module-level wrapper used by MultiClass plotting
    import calibrated_explanations.explanations.explanations as impl

    monkeypatch.setattr(impl, "_plot_probabilistic_dict", fake_plot_probabilistic)

    coll.plot_factual(index=None)

    # Should call the plot wrapper once (one instance)
    assert len(calls) == 1
    args, kw = calls[0]
    # positional args correspond to: class_explanations_list, factual_values, predicts, feature_weights_list, features_list_to_plot, filter_top, colors, column_names_list
    assert len(args) >= 8
    class_explanations_list = args[0]
    _factual_values = args[1]
    _predicts = args[2]
    feature_weights_list = args[3]
    features_list_to_plot = args[4]
    filter_top = args[5]

    assert isinstance(class_explanations_list, list)
    assert len(feature_weights_list) == 2
    assert len(features_list_to_plot) == 2
    assert filter_top[0] == 2


def test_should_rank_rules_within_each_class_before_multiclass_presentation_grouping():
    """ADR-042 D5/D9: rank native rules for each class, never grouped class rows."""
    class TwoRuleExp(SimpleExp):
        def get_rules(self):
            rules = super().get_rules()
            rules["weight"] = [0.1, 0.8] if self.class_id == 0 else [0.9, 0.2]
            rules["weight_low"] = [w - 0.05 for w in rules["weight"]]
            rules["weight_high"] = [w + 0.05 for w in rules["weight"]]
            return rules

    x = np.array([[1.0]])
    collection = MultiClassCalibratedExplanations(
        type("Explainer", (), {"num_features": 1})(),
        x, bins=None, num_classes=2,
        explanations=[{0: TwoRuleExp(0, 0), 1: TwoRuleExp(0, 1)}],
    )
    result = collection._build_factual_plot_payload(
        i=0,
        class_explanations=collection.explanations[0],
        filename="",
        show=False,
        uncertainty=False,
        style_override=None,
        kwargs={},
        filter_top=1,
    )
    assert result is not None
    # Class 0 selects r2; class 1 selects r1. A grouped-column ranking
    # instead might select just the strongest class entry from each row.
    assert result["column_names_list"] == ["r2", "r1"]
    assert result["filter_top"] == [1, 1]
    assert result["features_list_to_plot"] == [[0], [0]]


def test_should_preserve_requested_top_k_between_multiclass_alternative_instances(monkeypatch):
    """The second instance must not receive the first one's list of counts."""
    pytest.importorskip("matplotlib")
    import calibrated_explanations.explanations.explanations as impl

    class AlternativeExp(SimpleExp):
        def __init__(self, idx, cls):
            super().__init__(idx, cls)
            self.prediction = {"predict": 0.6, "low": 0.55, "high": 0.65, "classes": cls}

        def get_rules(self):
            rules = super().get_rules()
            rules["predict"] = [0.70, 0.85]
            rules["predict_low"] = [0.66, 0.83]
            rules["predict_high"] = [0.74, 0.87]
            return rules

    collection = MultiClassCalibratedExplanations(
        type("Explainer", (), {"num_features": 1})(),
        np.array([[1.0], [2.0]]),
        bins=None,
        num_classes=2,
        explanations=[
            {0: AlternativeExp(0, 0), 1: AlternativeExp(0, 1)},
            {0: AlternativeExp(1, 0), 1: AlternativeExp(1, 1)},
        ],
    )
    calls = []
    monkeypatch.setattr(
        impl, "_plot_alternative_dict",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    collection.plot_alternative(filter_top=1, show=False, rnk_metric="feature_weight")
    assert len(calls) == 2
    for args, kwargs in calls:
        assert args[4] == [[1], [1]]
        assert kwargs["num_to_show_list"] == [1, 1]


def test_should_forward_ranking_parameters_for_individual_multiclass_plots():
    """Single-class routes must not drop ranking kwargs."""
    class Recorder(SimpleExp):
        def plot(self, **kwargs):
            self.seen = kwargs

    record = Recorder(0, 0)
    coll = MultiClassCalibratedExplanations(
        type("Explainer", (), {"num_features": 1})(),
        np.array([[1.0]]), bins=None, num_classes=1, explanations=[{0: record}],
    )
    coll.plot_alternative(
        index=0, class_idx=0, show=False, filter_top=1,
        rnk_metric="feature_weight", rnk_weight=-0.25,
    )
    assert record.seen["rnk_metric"] == "feature_weight"
    assert record.seen["rnk_weight"] == -0.25
    assert record.seen["filter_top"] == 1


def test_should_preserve_ranking_and_report_multiclass_narrative_format_fallback(caplog):
    """Fallback to text is visible and never changes the requested rank settings."""
    class TextOnly:
        def __init__(self):
            self.calls = []

        def to_narrative(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["output_format"] == "dict":
                raise ValueError("dict unavailable")
            return "rendered"

    caplog.set_level("INFO")
    text_only = TextOnly()
    coll = MultiClassCalibratedExplanations(
        type("Explainer", (), {"num_features": 1})(),
        np.array([[1.0]]), bins=None, num_classes=1,
        explanations=[{0: text_only}],
    )
    with pytest.warns(UserWarning, match="identical ranking parameters"):
        result = coll.to_narrative(
            output_format="dict", rnk_metric="ensured",
            rnk_weight=0.3, filter_top=2,
        )
    assert result == [{0: {"text": "rendered"}}]
    assert len(text_only.calls) == 2
    assert all(call["rnk_metric"] == "ensured" for call in text_only.calls)
    assert all(call["rnk_weight"] == 0.3 for call in text_only.calls)
    assert all(call["filter_top"] == 2 for call in text_only.calls)
    assert "retrying as text" in caplog.text


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "3"])
def test_should_reject_invalid_multiclass_factual_top_k(invalid):
    coll = MultiClassCalibratedExplanations(
        type("Explainer", (), {"num_features": 1})(),
        np.array([[1.0]]), bins=None, num_classes=1,
        explanations=[{0: SimpleExp(0, 0)}],
    )
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError):
        coll._build_factual_plot_payload(
            i=0, class_explanations=coll.explanations[0], filename="",
            show=False, uncertainty=False, style_override=None,
            kwargs={}, filter_top=invalid,
        )
