from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Tuple


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


def cross_conformal_interval(
    make_model: Callable[[], any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    n_folds: int = 5,
) -> Dict[str, np.ndarray]:
    """Cross-conformal intervals via K-fold calibration residuals.

    Trains K models on train + (K-1) cal folds, collects residuals on held-out fold,
    then fits a final model on train+cal for the center. Uses global quantile of residuals.
    """
    n = len(X_cal)
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    resids = []
    for k in range(n_folds):
        hold = folds[k]
        keep = np.setdiff1d(idx, hold)
        m = make_model()
        X_fit = np.vstack([X_train, X_cal[keep]])
        y_fit = np.hstack([y_train, y_cal[keep]])
        m.fit(X_fit, y_fit)
        y_pred_hold = m.predict(X_cal[hold])
        resids.append(np.abs(y_cal[hold] - y_pred_hold))
    resids = np.concatenate(resids) if len(resids) else np.array([0.0])
    q = np.quantile(resids, 1 - alpha)
    # final model for center
    m_final = make_model()
    m_final.fit(np.vstack([X_train, X_cal]), np.hstack([y_train, y_cal]))
    center = m_final.predict(X_test)
    return {"lower": center - q, "upper": center + q, "center": center, "q": q}


def jackknife_plus_interval(
    make_model: Callable[[], any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    n_folds: int = 5,
) -> Dict[str, np.ndarray]:
    """Approximate Jackknife+ intervals using K-fold models.

    For fold k, train on train + other folds; compute residuals r_i on fold k points and predictions
    mu_k(x) for test x. Combine across all cal points:
    lower(x) = quantile_i [ mu_{fold(i)}(x) - r_i ] at alpha
    upper(x) = quantile_i [ mu_{fold(i)}(x) + r_i ] at 1-alpha
    """
    n = len(X_cal)
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    fold_models = []
    fold_residuals = []
    for k in range(n_folds):
        hold = folds[k]
        keep = np.setdiff1d(idx, hold)
        m = make_model()
        X_fit = np.vstack([X_train, X_cal[keep]])
        y_fit = np.hstack([y_train, y_cal[keep]])
        m.fit(X_fit, y_fit)
        y_pred_hold = m.predict(X_cal[hold])
        r = np.abs(y_cal[hold] - y_pred_hold)
        fold_models.append(m)
        fold_residuals.append((hold, r))

    # For each test point, compute mu_k(x)
    mus = [m.predict(X_test) for m in fold_models]  # list of [N_test]
    mus = np.stack(mus, axis=0)  # [K, N_test]

    # Build arrays of paired mu and r per cal point
    mu_minus_r = []
    mu_plus_r = []
    for k, (hold, r) in enumerate(fold_residuals):
        # Broadcast r over test points using corresponding mu_k(x)
        mu_k = mus[k]  # [N_test]
        mu_minus_r.append(mu_k[None, :] - r[:, None])  # [n_hold, N_test]
        mu_plus_r.append(mu_k[None, :] + r[:, None])
    if len(mu_minus_r) == 0:
        center = np.mean(mus, axis=0)
        return {"lower": center, "upper": center, "center": center, "q": 0.0}
    mu_minus_r = np.concatenate(mu_minus_r, axis=0)
    mu_plus_r = np.concatenate(mu_plus_r, axis=0)

    lower = np.quantile(mu_minus_r, alpha, axis=0)
    upper = np.quantile(mu_plus_r, 1 - alpha, axis=0)
    center = np.mean(mus, axis=0)
    return {"lower": lower, "upper": upper, "center": center}

