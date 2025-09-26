"""Dependency-free baseline implementations for M3 (no plugins).

Currently provided:
- Surrogate shallow tree on model predictions (classification): `surrogate_tree`
- Stump-on-proba (depth-1 tree on calibrated probabilities): `stump_on_proba`

Each module exposes a `fit_baseline(...)` function returning a fitted baseline
object, and an `explain_rules(...)` function that yields per-instance rule
records aligned with CE’s rule schema where feasible.
"""

from .surrogate_tree import fit_baseline as fit_surrogate_tree, explain_rules as explain_surrogate_rules  # noqa: F401
from .stump_on_proba import fit_baseline as fit_stump, explain_rules as explain_stump_rules  # noqa: F401

__all__ = [
    "fit_surrogate_tree",
    "explain_surrogate_rules",
    "fit_stump",
    "explain_stump_rules",
]

