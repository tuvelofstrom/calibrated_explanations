from __future__ import annotations

from typing import Any, Dict

import numpy as np

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def explain_with_rules(
    learner: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_eval: np.ndarray,
    mode: str,
    difficulty_estimator=None,
    **kwargs,
) -> Dict[str, Any]:
    """Hook into calibrated_explanations to obtain explanations/rules for analysis.
    Returns a dict with explanations and any metadata needed.
    """
    ce = CalibratedExplainer(
        learner=learner,
        X_cal=X_cal,
        y_cal=y_cal,
        mode=mode,
        difficulty_estimator=difficulty_estimator,
        **kwargs,
    )
    # For now, request global explanations for X_eval points with default settings.
    # Users can extend to conditional CE via kwargs (bins/discretizers in CE API).
    exps = ce.explain(X_eval)
    return {"explanations": exps, "explainer": ce}

