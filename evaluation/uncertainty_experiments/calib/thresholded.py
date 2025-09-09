from __future__ import annotations

import numpy as np
from typing import Dict


def exceedance_from_intervals(center: np.ndarray, lower: np.ndarray, upper: np.ndarray, t: float) -> np.ndarray:
    """Crude mapping from PI to exceedance probability P(Y>t).
    Placeholder until CPS utilities are integrated.
    """
    # Heuristic: 0 if t>=upper, 1 if t<=lower, else interpolate within [lower,upper]
    p = np.where(t >= upper, 0.0, np.where(t <= lower, 1.0, (upper - t) / (upper - lower + 1e-12)))
    return p

