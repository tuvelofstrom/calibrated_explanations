"""ADR-042 ranking: deterministic total order and task-specific semantics."""

from __future__ import annotations

import math

import pytest

from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.utils.ranking import canonical_rank_indices, rank_rule_indices


def _facts():
    return {
        "weight": [0.2, -0.2, 0.0],
        "weight_low": [0.0, -0.3, -0.1],
        "weight_high": [0.4, -0.1, 0.1],
        "predict": [0.2, 0.75, 0.8],
        "predict_low": [0.05, 0.74, 0.7],
        "predict_high": [0.45, 0.76, 0.9],
    }


def test_should_rank_factual_weights_by_magnitude_then_narrow_width_then_native_index():
    assert rank_rule_indices(_facts(), explanation_type="factual") == [1, 0, 2]


def test_should_rank_alternatives_by_ensured_by_default():
    assert rank_rule_indices(
        _facts(), explanation_type="alternative", base_predict=0.6
    ) == [1, 2, 0]


def test_should_allow_feature_weight_opt_in_for_alternatives():
    assert rank_rule_indices(
        _facts(),
        explanation_type="alternative",
        base_predict=0.6,
        rnk_metric="feature_weight",
    ) == [1, 0, 2]


def test_should_interpret_uncertainty_metric_as_ensured_with_zero_weight():
    assert rank_rule_indices(
        _facts(),
        explanation_type="alternative",
        base_predict=0.6,
        rnk_metric="uncertainty",
    ) == [1, 2, 0]


def test_should_put_nan_primary_last_and_nonfinite_width_last_in_ties():
    assert canonical_rank_indices(
        [math.nan, math.inf, -math.inf, 0.5, 0.5],
        [0.0, math.inf, 0.0, math.nan, 0.1],
    ) == [1, 4, 3, 2, 0]


def test_should_break_complete_ties_by_original_rule_index():
    assert canonical_rank_indices([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]) == [0, 1, 2]


def test_should_keep_zero_weight_rules_without_filtering():
    assert rank_rule_indices(
        {"weight": [0.0], "weight_low": [0.0], "weight_high": [0.0]},
        explanation_type="fast",
    ) == [0]


@pytest.mark.parametrize("k,expected", [(None, [1, 2, 0]), (0, []), (1, [1]), (9, [1, 2, 0])])
def test_should_select_prefix_when_top_k_is_valid(k, expected):
    assert canonical_rank_indices([0.1, 0.9, 0.4], top_k=k) == expected


@pytest.mark.parametrize("k", [-1, 1.5, True, "2"])
def test_should_reject_invalid_top_k(k):
    with pytest.raises(ValidationError):
        canonical_rank_indices([1.0], top_k=k)


def test_should_rank_unbounded_alternative_after_finite_one():
    rules = {
        "weight": [0.2, 0.3],
        "predict": [0.8, 0.9],
        "predict_low": [0.0, 0.8],
        "predict_high": [math.inf, 1.0],
    }
    assert rank_rule_indices(
        rules, explanation_type="alternative", base_predict=0.7
    ) == [1, 0]


def test_should_rank_plain_regression_on_raw_prediction_without_probability_inversion():
    rules = {
        "weight": [1.0, 2.0],
        "predict": [3.0, 5.0],
        "predict_low": [2.0, 4.0],
        "predict_high": [4.0, 6.0],
    }
    assert rank_rule_indices(
        rules, explanation_type="alternative", probability=False
    ) == [1, 0]


def test_should_allow_negative_ensured_weight_for_counter_alternatives():
    rules = _facts()
    assert rank_rule_indices(
        rules,
        explanation_type="alternative",
        base_predict=0.6,
        rnk_weight=-0.5,
    ) == [0, 1, 2]


def test_should_validate_factual_metric_and_alternative_weight():
    with pytest.raises(ValidationError):
        rank_rule_indices(_facts(), explanation_type="factual", rnk_metric="ensured")
    with pytest.raises(ValidationError):
        rank_rule_indices(
            _facts(), explanation_type="alternative", base_predict=0.6, rnk_weight=math.nan
        )
