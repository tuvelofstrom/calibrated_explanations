from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path when running as a script
import sys
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]  # .../repo
_REPO_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_REPO_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from evaluation.uncertainty_experiments.data_gen.regression import generate_regression
from evaluation.uncertainty_experiments.data_gen.classification import make_binary_gaussian
from evaluation.uncertainty_experiments.models.wrappers import make_model
from evaluation.uncertainty_experiments.calib.classification_venn_abers import (
    fit_venn_abers,
    predict_calibrated,
)
from evaluation.uncertainty_experiments.calib.regression_cp import split_conformal_interval
from evaluation.uncertainty_experiments.calib.thresholded import exceedance_from_intervals
from evaluation.uncertainty_experiments.uncertainty.estimators import (
    ensemble_disagreement_classification,
    ensemble_disagreement_regression,
    knn_inverse_density,
    nonconformity_regression,
)
from evaluation.uncertainty_experiments.difficulty.de import CompositeDE
from evaluation.uncertainty_experiments.metrics.metrics import (
    ece,
    brier_score,
    coverage_width,
)


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _to_py(o: Any) -> Any:
    """Recursively convert numpy types/arrays into JSON-serializable Python types."""
    try:
        import numpy as _np  # local import to avoid hard dep in type hints
    except Exception:  # pragma: no cover
        _np = None  # type: ignore

    if _np is not None:
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, (_np.floating, _np.integer, _np.bool_)):
            return o.item()
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    return o


def run_regression(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = generate_regression(
        n_train=cfg["data"]["n_train"],
        n_cal=cfg["data"]["n_cal"],
        n_test=cfg["data"]["n_test"],
        dims=cfg["data"].get("dims", 2),
        holes=cfg["data"].get("holes"),
        shift=cfg["data"].get("shift", 0.0),
        hetero=cfg["data"].get("hetero"),
        seed=cfg.get("seed", 42),
    )

    model = make_model("regression", cfg["model"]["type"], cfg["model"]["params"]).fit(
        data["X_train"], data["y_train"]
    )

    # Uncertainty components
    ens_samples_cal = model.predict_many(data["X_cal"], n_samples=30, seed=cfg.get("seed", 42))
    ens_samples_te = model.predict_many(data["X_test"], n_samples=30, seed=cfg.get("seed", 42))
    disagreement_cal = ensemble_disagreement_regression(ens_samples_cal)

    yte_pred = model.predict(data["X_test"])  # for nonconformity baseline, use cal residuals
    yca_pred = model.predict(data["X_cal"])
    nonconf = nonconformity_regression(data["y_cal"], yca_pred)

    inv_density = knn_inverse_density(data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))

    # Composite DE on calibration points, then NN transfer to test
    de_cal = CompositeDE(
        weights=cfg["uncertainty"].get("de_weights", {}),
        X_ref=data["X_cal"],
        components={
            "nonconformity": nonconf,
            "disagreement": disagreement_cal,
            "inv_density": inv_density,
        },
    )
    de_test = de_cal.apply(data["X_test"])  # nearest-neighbor transfer

    # Split CP intervals
    cp = split_conformal_interval(
        model.model, data["X_cal"], data["y_cal"], data["X_test"], alpha=cfg["calibration"]["regression"].get("alpha", 0.1)
    )
    cov = coverage_width(cp["lower"], cp["upper"], data["y_test"])

    return {
        "task": "regression",
        "metrics": {**cov},
        "artifacts": {
            "y_test": data["y_test"],
            "pred_center": cp["center"],
            "pi_lower": cp["lower"],
            "pi_upper": cp["upper"],
            "de_test": de_test,
            "sigma_test": data["sigma_test"],
        },
    }


def run_classification(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = make_binary_gaussian(
        n_train=cfg["data"]["n_train"],
        n_cal=cfg["data"]["n_cal"],
        n_test=cfg["data"]["n_test"],
        dims=cfg["data"].get("dims", 2),
        overlap_scale=1.0,
        holes=cfg["data"].get("holes"),
        shift=cfg["data"].get("shift", 0.0),
        seed=cfg.get("seed", 42),
    )

    model = make_model("classification", cfg["model"]["type"], cfg["model"]["params"]).fit(
        data["X_train"], data["y_train"]
    )

    # Venn–Abers calibration on calibration set
    va = fit_venn_abers(model.model, data["X_cal"], data["y_cal"])  # align with CE
    probs = predict_calibrated(va, data["X_test"])  # calibrated probabilities

    # Uncertainty components
    ens_samples_cal = model.predict_many(data["X_cal"], n_samples=30, seed=cfg.get("seed", 42))
    ens_samples_te = model.predict_many(data["X_test"], n_samples=30, seed=cfg.get("seed", 42))
    disagreement_cal = ensemble_disagreement_classification(ens_samples_cal)

    # Density proxy on cal set then NN transfer
    inv_density = knn_inverse_density(data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))
    de_cal = CompositeDE(
        weights=cfg["uncertainty"].get("de_weights", {}),
        X_ref=data["X_cal"],
        components={
            "disagreement": disagreement_cal,
            "inv_density": inv_density,
        },
    )
    de_test = de_cal.apply(data["X_test"])  # nearest-neighbor transfer

    metrics = {
        "ece": ece(probs, data["y_test"]),
        "brier": brier_score(probs if probs.ndim > 1 else probs.reshape(-1, 1), data["y_test"]),
    }

    return {
        "task": "classification",
        "metrics": metrics,
        "artifacts": {
            "y_test": data["y_test"],
            "probs": probs,
            "de_test": de_test,
            "eta_test": (data["eta_test"] if "eta_test" in data else None),
        },
    }


def run_thresholded_regression(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Thresholded regression: estimate P(Y>t) for one or many thresholds t.

    Uses split-CP intervals for a simple exceedance mapping. CPS wiring can replace this later.
    """
    data = generate_regression(
        n_train=cfg["data"]["n_train"],
        n_cal=cfg["data"]["n_cal"],
        n_test=cfg["data"]["n_test"],
        dims=cfg["data"].get("dims", 2),
        holes=cfg["data"].get("holes"),
        shift=cfg["data"].get("shift", 0.0),
        hetero=cfg["data"].get("hetero"),
        seed=cfg.get("seed", 42),
    )

    model = make_model("regression", cfg["model"]["type"], cfg["model"]["params"]).fit(
        data["X_train"], data["y_train"]
    )

    # Intervals via split-CP
    alpha = cfg.get("calibration", {}).get("regression", {}).get("alpha", 0.1)
    cp = split_conformal_interval(model.model, data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha)

    # Threshold grid
    t_values = (
        cfg.get("calibration", {}).get("thresholded", {}).get("t_values")
    )
    if not t_values:
        # default to yt percentiles as sensible thresholds
        qs = [0.2, 0.5, 0.8]
        t_values = [float(np.quantile(data["y_test"], q)) for q in qs]

    # Uncertainty components for DE (re-using regression path)
    ens_samples_cal = model.predict_many(data["X_cal"], n_samples=30, seed=cfg.get("seed", 42))
    disagreement_cal = ensemble_disagreement_regression(ens_samples_cal)
    yca_pred = model.predict(data["X_cal"])
    nonconf = nonconformity_regression(data["y_cal"], yca_pred)
    inv_density = knn_inverse_density(data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))
    de_cal = CompositeDE(
        weights=cfg["uncertainty"].get("de_weights", {}),
        X_ref=data["X_cal"],
        components={
            "nonconformity": nonconf,
            "disagreement": disagreement_cal,
            "inv_density": inv_density,
        },
    )
    de_test = de_cal.apply(data["X_test"])  # nearest-neighbor transfer

    # Evaluate reliability per threshold
    per_t_metrics: Dict[str, Dict[str, float]] = {}
    per_t_probs: Dict[str, Any] = {}
    for t in t_values:
        p = exceedance_from_intervals(cp["center"], cp["lower"], cp["upper"], float(t))
        y_bin = (data["y_test"] > float(t)).astype(int)
        per_t_metrics[f"t={t:.4g}"] = {
            "ece": ece(p.reshape(-1, 1), y_bin),  # binary ECE over scalar prob
            "brier": brier_score(p.reshape(-1, 1), y_bin),
        }
        per_t_probs[f"t={t:.4g}"] = p

    return {
        "task": "thresh_reg",
        "metrics": {"per_threshold": per_t_metrics},
        "artifacts": {
            "y_test": data["y_test"],
            "pred_center": cp["center"],
            "pi_lower": cp["lower"],
            "pi_upper": cp["upper"],
            "de_test": de_test,
            "sigma_test": data["sigma_test"],
            "p_exceed": per_t_probs,
            "t_values": t_values,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run calibrated uncertainty experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        text = f.read()
        if args.config.endswith(".json"):
            cfg = json.loads(text)
        else:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML not installed. Use a .json config or install pyyaml."
                )
            cfg = yaml.safe_load(text)

    np.random.seed(cfg.get("seed", 42))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.get("output_dir", "evaluation/uncertainty_experiments/artifacts"))
    ensure_dir(out_dir)
    run_id = f"run_{cfg.get('task','task')}_{ts}"
    run_dir = out_dir / run_id
    ensure_dir(run_dir)

    task = cfg["task"].lower()
    if task == "regression":
        res = run_regression(cfg)
    elif task == "classification":
        res = run_classification(cfg)
    elif task in {"thresh_reg", "thresholded", "thresholded_regression"}:
        res = run_thresholded_regression(cfg)
    else:
        raise NotImplementedError(f"Unknown task '{task}'")

    # Save metrics and artifacts
    if cfg.get("logging", {}).get("save_metrics", True):
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_to_py(res["metrics"]), f, indent=2)
    if cfg.get("logging", {}).get("save_predictions", True):
        with open(run_dir / "artifacts.json", "w", encoding="utf-8") as f:
            json.dump(_to_py(res["artifacts"]), f)  # keep compact

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
