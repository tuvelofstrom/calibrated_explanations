from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def split_conformal_interval(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Basic split conformal prediction intervals using absolute residuals.

    Returns dict with 'lower', 'upper', and 'center' (point preds).
    """
    y_cal_pred = model.predict(X_cal)
    resid = np.abs(y_cal - y_cal_pred)
    q = np.quantile(resid, 1 - alpha)
    y_te_pred = model.predict(X_test)
    lower = y_te_pred - q
    upper = y_te_pred + q
    return {"lower": lower, "upper": upper, "center": y_te_pred, "q": q}


