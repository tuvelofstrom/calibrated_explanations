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
from evaluation.uncertainty_experiments.calib.regression_cp import (
    split_conformal_interval,
    cross_conformal_interval,
    jackknife_plus_interval,
)
from evaluation.uncertainty_experiments.calib.thresholded import exceedance_from_intervals
from evaluation.uncertainty_experiments.uncertainty.estimators import (
    ensemble_disagreement_classification,
    ensemble_disagreement_regression,
    knn_inverse_density,
    knn_inverse_density_ref,
    nonconformity_regression,
)
from evaluation.uncertainty_experiments.difficulty.de import CompositeDE
from evaluation.uncertainty_experiments.metrics.metrics import (
    ece,
    brier_score,
    coverage_width,
)
from evaluation.uncertainty_experiments.rules.hooks import (
    extract_factual_rule_records_for_classification,
    extract_factual_rule_records_for_thresholded_regression,
    extract_alternative_rule_records_for_classification,
)
from evaluation.uncertainty_experiments.baselines import (
    fit_surrogate_tree,
    explain_surrogate_rules,
    fit_stump,
    explain_stump_rules,
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
    inv_density_test = knn_inverse_density_ref(data["X_test"], data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))

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

    # Conformal intervals (method switch)
    reg_cal = cfg.get("calibration", {}).get("regression", {})
    method = (reg_cal.get("method") or "split_cp").lower()
    alpha = reg_cal.get("alpha", 0.1)
    n_folds = int(reg_cal.get("n_folds", 5))

    def _make_model():
        # fresh model factory with same params
        return make_model("regression", cfg["model"]["type"], cfg["model"]["params"]).model

    if method in ("split", "split_cp"):
        cp = split_conformal_interval(model.model, data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha)
    elif method in ("cross", "cross_cp"):
        cp = cross_conformal_interval(_make_model, data["X_train"], data["y_train"], data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha, n_folds=n_folds)
    elif method in ("jackknife+", "jackknife_plus", "jk+"):
        cp = jackknife_plus_interval(_make_model, data["X_train"], data["y_train"], data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha, n_folds=n_folds)
    else:
        raise ValueError(f"Unknown regression calibration method: {method}")
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
            "inv_density_test": inv_density_test,
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

    # Probability estimates (ablation: ce_uncalibrated -> raw predict_proba)
    ablation = (cfg.get("ablation") or "none").lower()
    if ablation == "ce_uncalibrated":
        proba_raw = model.model.predict_proba(data["X_test"])  # type: ignore[attr-defined]
        probs = proba_raw[:, 1] if proba_raw.ndim == 2 and proba_raw.shape[1] > 1 else proba_raw.reshape(-1)
    else:
        va = fit_venn_abers(model.model, data["X_cal"], data["y_cal"])  # align with CE
        probs = predict_calibrated(va, data["X_test"])  # calibrated probabilities

    # Uncertainty components
    ens_samples_cal = model.predict_many(data["X_cal"], n_samples=30, seed=cfg.get("seed", 42))
    ens_samples_te = model.predict_many(data["X_test"], n_samples=30, seed=cfg.get("seed", 42))
    disagreement_cal = ensemble_disagreement_classification(ens_samples_cal)

    # Density proxy on cal set then NN transfer
    inv_density = knn_inverse_density(data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))
    inv_density_test = knn_inverse_density_ref(data["X_test"], data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))
    # Compose DE according to ablation
    de_weights = dict(cfg.get("uncertainty", {}).get("de_weights", {}))
    if ablation == "gate_density_only":
        de_weights = {"inv_density": 1.0, "disagreement": 0.0}
    elif ablation == "gate_disagreement_only":
        de_weights = {"inv_density": 0.0, "disagreement": 1.0}
    de_cal = CompositeDE(
        weights=de_weights,
        X_ref=data["X_cal"],
        components={
            "disagreement": disagreement_cal,
            "inv_density": inv_density,
        },
    )
    de_test = None if ablation == "ce_no_de" else de_cal.apply(data["X_test"])  # nearest-neighbor transfer

    metrics = {
        "ece": ece(probs, data["y_test"]),
        "brier": brier_score(probs if probs.ndim > 1 else probs.reshape(-1, 1), data["y_test"]),
    }

    res: Dict[str, Any] = {
        "task": "classification",
        "metrics": metrics,
        "artifacts": {
            "y_test": data["y_test"],
            "probs": probs,
            "de_test": de_test,
            "eta_test": (data["eta_test"] if "eta_test" in data else None),
            "inv_density_test": inv_density_test,
        },
    }

    # Optional: dump local CE factual/alternative rules for this run
    if cfg.get("logging", {}).get("dump_rules", False):
        try:
            # Build CE factual explanations on the same learner
            from evaluation.uncertainty_experiments.models.wrappers import ModelWrapper

            mw = model  # ModelWrapper
            # Derive feature names if available from learner
            feature_names = None
            if hasattr(mw.model, "n_features_in_"):
                feature_names = [str(i) for i in range(getattr(mw.model, "n_features_in_", 0))]

            rules_records = extract_factual_rule_records_for_classification(
                learner=mw.model,
                X_cal=data["X_cal"],
                y_cal=data["y_cal"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                de_test=de_test,
                inv_density_test=inv_density_test,
                eta_test=(data["eta_test"] if "eta_test" in data else None),
                feature_names=feature_names,
                seed=cfg.get("seed", 42),
            )
            # Optionally append alternative explanations from CE
            if cfg.get("logging", {}).get("dump_alternatives", False):
                alt_records = extract_alternative_rule_records_for_classification(
                    learner=mw.model,
                    X_cal=data["X_cal"],
                    y_cal=data["y_cal"],
                    X_test=data["X_test"],
                    y_test=data["y_test"],
                    feature_names=feature_names,
                    seed=cfg.get("seed", 42),
                )
                rules_records.extend(alt_records)
            res["rules_records"] = rules_records
        except Exception as exc:  # pragma: no cover - defensive
            # Surface a soft failure; main flow should not break on rules plumbing
            res["rules_error"] = str(exc)

    # Optional: dump baseline rules for classification (dependency-free baselines)
    if cfg.get("logging", {}).get("dump_baselines", False) and cfg.get("baselines"):
        try:
            baseline_records = _maybe_build_baselines_classification(cfg, model, data)
            if baseline_records:
                res["baseline_records"] = baseline_records
        except Exception:
            pass

    return res


def _attach_covariates(records: List[Dict[str, Any]], de: Any, invd: Any, eta: Any) -> List[Dict[str, Any]]:
    out = []
    import numpy as _np
    for rec in records:
        pid = int(rec.get("point_id", -1))
        if pid >= 0:
            rec = dict(rec)
            if de is not None:
                try:
                    rec["de"] = float(de[pid])
                except Exception:
                    pass
            if invd is not None:
                try:
                    rec["inv_density"] = float(invd[pid])
                except Exception:
                    pass
            if eta is not None:
                try:
                    rec["eta"] = float(eta[pid])
                except Exception:
                    pass
        out.append(rec)
    return out


def _maybe_build_baselines_classification(cfg: Dict[str, Any], model: Any, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    baselines = cfg.get("baselines", []) or []
    if not baselines:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    # Fit targets on train using underlying model probabilities
    # Feature names best-effort
    feature_names = None
    if hasattr(model.model, "n_features_in_"):
        feature_names = [str(i) for i in range(getattr(model.model, "n_features_in_", 0))]
    for name in baselines:
        try:
            if name == "surrogate_tree":
                b = fit_surrogate_tree(model.model, data["X_train"], data["y_train"], seed=cfg.get("seed", 42))
                recs = explain_surrogate_rules(b, data["X_cal"], data["y_cal"], data["X_test"], data["y_test"], feature_names=feature_names)
            elif name == "stump_on_proba":
                b = fit_stump(model.model, data["X_train"], data["y_train"], seed=cfg.get("seed", 42))
                recs = explain_stump_rules(b, data["X_cal"], data["y_cal"], data["X_test"], data["y_test"], feature_names=feature_names)
            else:
                continue
            # attach covariates
            recs = _attach_covariates(recs, de=data.get("de_test"), invd=data.get("inv_density_test"), eta=data.get("eta_test"))
            out[name] = recs
        except Exception:
            continue
    return out


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

    # Intervals via chosen CP
    reg_cal = cfg.get("calibration", {}).get("regression", {})
    method = (reg_cal.get("method") or "split_cp").lower()
    alpha = reg_cal.get("alpha", 0.1)
    n_folds = int(reg_cal.get("n_folds", 5))

    def _make_model():
        return make_model("regression", cfg["model"]["type"], cfg["model"]["params"]).model

    if method in ("split", "split_cp"):
        cp = split_conformal_interval(model.model, data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha)
    elif method in ("cross", "cross_cp"):
        cp = cross_conformal_interval(_make_model, data["X_train"], data["y_train"], data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha, n_folds=n_folds)
    elif method in ("jackknife+", "jackknife_plus", "jk+"):
        cp = jackknife_plus_interval(_make_model, data["X_train"], data["y_train"], data["X_cal"], data["y_cal"], data["X_test"], alpha=alpha, n_folds=n_folds)
    else:
        raise ValueError(f"Unknown regression calibration method: {method}")

    # Threshold grid
    def _resolve_t_values(raw):
        if raw is None or (isinstance(raw, str) and raw.lower() == "auto"):
            qs = [0.2, 0.5, 0.8]
            return [float(np.quantile(data["y_test"], q)) for q in qs]
        if isinstance(raw, (int, float)):
            return [float(raw)]
        if isinstance(raw, str):
            try:
                return [float(raw)]
            except Exception:
                # allow comma-separated values "-0.5,0.0,0.5"
                parts = [p.strip() for p in raw.split(",")]
                return [float(p) for p in parts if p]
        if isinstance(raw, (list, tuple)):
            out = []
            for v in raw:
                if isinstance(v, (int, float)):
                    out.append(float(v))
                else:
                    out.append(float(str(v)))
            return out
        # Fallback to auto
        qs = [0.2, 0.5, 0.8]
        return [float(np.quantile(data["y_test"], q)) for q in qs]

    t_values = _resolve_t_values(cfg.get("calibration", {}).get("thresholded", {}).get("t_values"))

    # Uncertainty components for DE (re-using regression path)
    ens_samples_cal = model.predict_many(data["X_cal"], n_samples=30, seed=cfg.get("seed", 42))
    disagreement_cal = ensemble_disagreement_regression(ens_samples_cal)
    yca_pred = model.predict(data["X_cal"])
    nonconf = nonconformity_regression(data["y_cal"], yca_pred)
    inv_density = knn_inverse_density(data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20))
    ablation = (cfg.get("ablation") or "none").lower()
    de_weights = dict(cfg.get("uncertainty", {}).get("de_weights", {}))
    if ablation == "gate_density_only":
        de_weights = {"inv_density": 1.0, "disagreement": 0.0, "nonconformity": 0.0}
    elif ablation == "gate_disagreement_only":
        de_weights = {"inv_density": 0.0, "disagreement": 1.0, "nonconformity": 0.0}
    de_cal = CompositeDE(
        weights=de_weights,
        X_ref=data["X_cal"],
        components={
            "nonconformity": nonconf,
            "disagreement": disagreement_cal,
            "inv_density": inv_density,
        },
    )
    de_test = None if ablation == "ce_no_de" else de_cal.apply(data["X_test"])  # nearest-neighbor transfer

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

    # Prepare base result (we may augment with rules_records below)
    base_res: Dict[str, Any] = {
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
            "inv_density_test": knn_inverse_density_ref(data["X_test"], data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20)),
        },
    }

    # Optional: dump CE factual rules with per-threshold probabilities
    if cfg.get("logging", {}).get("dump_rules", False):
        try:
            from evaluation.uncertainty_experiments.models.wrappers import ModelWrapper

            mw = model  # ModelWrapper
            feature_names = None
            if hasattr(mw.model, "n_features_in_"):
                feature_names = [str(i) for i in range(getattr(mw.model, "n_features_in_", 0))]

            rules_records = extract_factual_rule_records_for_thresholded_regression(
                learner=mw.model,
                X_cal=data["X_cal"],
                y_cal=data["y_cal"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                t_values=t_values,
                de_test=de_test,
                inv_density_test=knn_inverse_density_ref(data["X_test"], data["X_cal"], k=cfg["uncertainty"].get("knn_k", 20)),
                sigma_test=data["sigma_test"],
                feature_names=feature_names,
                seed=cfg.get("seed", 42),
            )
            # Attach to result for upstream persistence
            base_res["rules_records"] = rules_records
        except Exception as exc:  # pragma: no cover
            # Fall back silently; main artifacts are preserved
            pass

    return base_res


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
    # Optional: rules.jsonl for M2.1 when available
    if cfg.get("logging", {}).get("dump_rules", False) and "rules_records" in res:
        rules_path = run_dir / "rules.jsonl"
        run_id = run_id  # keep name explicit
        with open(rules_path, "w", encoding="utf-8") as f:
            for rec in res["rules_records"]:
                full = {"run_id": run_id, **rec}
                f.write(json.dumps(_to_py(full)) + "\n")
    # Save resolved run config for aggregation
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_to_py(cfg), f, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
