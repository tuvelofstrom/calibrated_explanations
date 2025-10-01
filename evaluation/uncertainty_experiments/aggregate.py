from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Set, Tuple

try:  # pragma: no cover - optional dependency and circular import safety
    from evaluation.uncertainty_experiments.grid_runner import _read_config as _read_grid_config  # type: ignore
except Exception:  # pragma: no cover
    _read_grid_config = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
import math

# Optional parallelism for per-run heavy steps
try:  # pragma: no cover - optional dependency
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    delayed = None  # type: ignore


def _scan_runs(root: Path) -> List[Path]:
    return [p for p in root.glob("run_*_*") if p.is_dir()]


def _filter_runs_by_experiment(runs: List[Path], experiments: Optional[Set[str]]) -> List[Path]:
    if not experiments:
        return runs
    filtered: List[Path] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        exp_name = cfg.get("experiment")
        if exp_name in experiments:
            filtered.append(rdir)
    return filtered


def _experiments_from_spec(spec: Dict[str, Any]) -> Set[str]:
    experiments: Set[str] = set()
    value = spec.get("experiment")
    if isinstance(value, str) and value:
        experiments.add(value)
    extra = spec.get("experiments")
    if isinstance(extra, str) and extra:
        experiments.add(extra)
    elif isinstance(extra, list):
        experiments.update({str(v) for v in extra if isinstance(v, str) and v})
    return experiments


def _read_spec(path: Path) -> Dict[str, Any]:
    if _read_grid_config is not None:
        return _read_grid_config(os.fspath(path))
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if str(path).endswith(".json"):
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read non-JSON specs")
    return yaml.safe_load(text)


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_ci95(xs: List[float]) -> Tuple[float, float]:
    import math

    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) <= 1:
        return (m, float("nan"))
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    se = math.sqrt(var) / (len(xs) ** 0.5)
    return (m, 1.96 * se)


def _rank_corr(xs: List[float], ys: List[float]) -> float:
    # Spearman via Pearson on ranks
    import numpy as np

    if len(xs) != len(ys) or len(xs) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(np.array(xs))).astype(float)
    ry = np.argsort(np.argsort(np.array(ys))).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def aggregate(
    root: Path,
    out: Path,
    n_jobs: int = 1,
    *,
    jaccard_k: int = 3,
    rule_failure_tau: float = 0.7,
    experiments: Optional[Set[str]] = None,
) -> None:
    # Ensure output directory exists and normalize path to avoid Windows path quirks
    out.mkdir(parents=True, exist_ok=True)
    try:
        out = out.resolve()
    except Exception:
        pass

    def _open_w(p: Path):
        """Open a text file for CSV writing robustly across platforms.

        Ensures parent directory exists. Falls back to str(path) or
        omitting newline on rare Windows invalid-argument issues.
        """
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            return open(p, "w", newline="", encoding="utf-8")
        except OSError as e:
            # Retry with str path, and without newline if needed
            if getattr(e, "errno", None) == 22:
                try:
                    return open(os.fspath(p), "w", encoding="utf-8")
                except Exception:
                    pass
            raise
    runs = _filter_runs_by_experiment(_scan_runs(root), experiments)

    # Context columns extracted from per-run config
    def _ctx(cfg: Dict[str, Any]) -> Dict[str, Any]:
        d = cfg.get("data", {})
        cal = cfg.get("calibration", {})
        model = cfg.get("model", {})
        return {
            "experiment": cfg.get("experiment"),
            "task": cfg.get("task"),
            "seed": cfg.get("seed"),
            "dims": d.get("dims"),
            "n_train": d.get("n_train"),
            "n_cal": d.get("n_cal"),
            "n_test": d.get("n_test"),
            "shift": d.get("shift"),
            "holes_label": d.get("holes_label"),
            "model": model.get("type"),
            "model_params": json.dumps(model.get("params", {}), sort_keys=True),
            "calib_classification": (cal.get("classification", {}) or {}).get("method"),
            "calib_regression": (cal.get("regression", {}) or {}).get("method"),
            "alpha": (cal.get("regression", {}) or {}).get("alpha"),
            "calib_thresholded": (cal.get("thresholded", {}) or {}).get("method"),
            "t_values": json.dumps((cal.get("thresholded", {}) or {}).get("t_values")),
            "knn_k": (cfg.get("uncertainty", {}) or {}).get("knn_k"),
            "de_weights": json.dumps((cfg.get("uncertainty", {}) or {}).get("de_weights"), sort_keys=True),
            "ablation": cfg.get("ablation"),
            "baselines": json.dumps(cfg.get("baselines", []), sort_keys=True),
        }

    # 1) Regression: conditional coverage by sigma bins
    reg_rows: List[Dict[str, Any]] = []
    # Iterate all runs and filter to regression (fix: previously looped classification runs)
    processed = 0
    total_runs = len(runs)
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            m = _load_json(rdir / "metrics.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "regression":
            continue
        import numpy as np

        sigma = np.array(a.get("sigma_test", []), dtype=float)
        y = np.array(a.get("y_test", []), dtype=float)
        lo = np.array(a.get("pi_lower", []), dtype=float)
        hi = np.array(a.get("pi_upper", []), dtype=float)
        if len(sigma) == 0:
            continue
        # Bin sigma into tertiles
        cuts = np.quantile(sigma, [0.33, 0.66])
        bins = np.digitize(sigma, cuts)
        for b in [0, 1, 2]:
            mask = bins == b
            if not mask.any():
                continue
            covered = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
            width = (hi[mask] - lo[mask])
            reg_rows.append({**_ctx(cfg), "sigma_bin": b, "coverage": float(covered.mean()), "avg_width": float(width.mean())})
    # Write coverage_by_sigma.csv
    # Always write CSV (with header only if empty)
    with open(out / "coverage_by_sigma.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "sigma_bin","coverage","avg_width",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if reg_rows:
            w.writerows(reg_rows)

    # 1b) CE-first: base calibration stratified by top-1 |w| (classification factual rules)
    def _bin_indices(p: List[float], n_bins: int = 10) -> Tuple[List[int], List[float]]:
        import numpy as np

        arr = np.asarray(p, dtype=float)
        # clip into [0,1]
        arr = np.clip(arr, 0.0, 1.0)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        b = np.digitize(arr, edges, right=True)  # 0..n_bins
        b[b == 0] = 1
        b[b > n_bins] = n_bins
        return b.tolist(), edges.tolist()

    ce_rel_rows: List[Dict[str, Any]] = []
    ce_ece_rows: List[Dict[str, Any]] = []
    ce_w_unc_rows: List[Dict[str, Any]] = []
    ce_w_unc_eta_rows: List[Dict[str, Any]] = []
    ce_dir_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        rules_path = rdir / "rules.jsonl"
        if not rules_path.exists():
            continue
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        # cfg already checked; continue
        # Load rule records
        recs: List[Dict[str, Any]] = []
        with open(rules_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
        if not recs:
            processed += 1
            if processed % 25 == 0 or processed == total_runs:
                print(f"[CE metrics] scanned {processed}/{total_runs} runs for rules.jsonl")
            continue
        # Filter factual + availability of base prediction and y_true
        recs = [r for r in recs if r.get("explanation_type") == "factual" and (r.get("p_pred") is not None) and (r.get("y_true") is not None)]
        if not recs:
            continue
        import numpy as np
        # Build per-point top-1 |w|
        df = recs
        # Group by point_id
        by_pt: Dict[int, Dict[str, Any]] = {}
        for r in df:
            pid = int(r.get("point_id", -1))
            if pid < 0:
                continue
            wabs = abs(float(r.get("w", 0.0)))
            cur = by_pt.get(pid)
            if cur is None or wabs > cur["wabs"]:
                by_pt[pid] = {
                    "wabs": wabs,
                    "w_width": float(r.get("w_width", float("nan"))),
                    "p_pred": float(r.get("p_pred", float("nan"))),
                    "y_true": int(r.get("y_true", 0)),
                    "y_pred": int(r.get("y_pred", 0)),
                    "w": float(r.get("w", 0.0)),
                    "inv_density": (float(r.get("inv_density")) if r.get("inv_density") is not None else float("nan")),
                    "eta": (float(r.get("eta")) if r.get("eta") is not None else float("nan")),
                }
        if not by_pt:
            processed += 1
            if processed % 25 == 0 or processed == total_runs:
                print(f"[CE metrics] scanned {processed}/{total_runs} runs for rules.jsonl")
            continue
        import numpy as np
        pts = list(by_pt.values())
        p = np.array([v["p_pred"] for v in pts], dtype=float)
        y = np.array([v["y_true"] for v in pts], dtype=int)
        wabs = np.array([v["wabs"] for v in pts], dtype=float)
        wwidth = np.array([v["w_width"] for v in pts], dtype=float)
        invd = np.array([v["inv_density"] for v in pts], dtype=float)
        eta = np.array([v["eta"] for v in pts], dtype=float)

        # Tertiles for |w|, density, eta
        def tertiles(arr: np.ndarray) -> np.ndarray:
            x = arr.copy()
            if np.all(np.isnan(x)):
                return np.zeros_like(x)
            # replace nans with mean for binning
            mu = np.nanmean(x)
            x[np.isnan(x)] = mu
            cuts = np.quantile(x, [0.33, 0.66])
            return np.digitize(x, cuts)

        wb = tertiles(wabs)
        db = tertiles(invd)
        eb = tertiles(eta)

        # Reliability by |w| tertiles (base prediction)
        for wi in [0, 1, 2]:
            mask_w = wb == wi
            if not np.any(mask_w):
                continue
            bins, edges = _bin_indices(p[mask_w].tolist(), n_bins=10)
            bins = np.array(bins)
            total = int(np.sum(mask_w))
            ece_sum = 0.0
            for b in range(1, 11):
                m = bins == b
                if not np.any(m):
                    continue
                p_mean = float(np.nanmean(p[mask_w][m]))
                acc_mean = float(np.mean(y[mask_w][m]))
                n = int(np.sum(m))
                se = math.sqrt(acc_mean * (1 - acc_mean) / max(n, 1)) if n > 1 else 0.0
                ce_rel_rows.append({**_ctx(cfg), "run_dir": rdir.name, "weight_bin": wi, "bin": b, "bin_edge_lo": edges[b-1], "bin_edge_hi": edges[b], "p_mean": p_mean, "acc_mean": acc_mean, "count": n, "acc_ci95": 1.96 * se})
                ece_sum += abs(acc_mean - p_mean) * (n / max(total, 1))
            ce_ece_rows.append({**_ctx(cfg), "run_dir": rdir.name, "weight_bin": wi, "ece": ece_sum})

        # Weight uncertainty vs density tertiles (top-1 rule)
        for di in [0, 1, 2]:
            md = db == di
            if not np.any(md):
                continue
            vals = wwidth[md]
            mu = float(np.nanmean(vals))
            # CI via normal approx on non-nan
            vv = vals[~np.isnan(vals)]
            se = float(np.std(vv, ddof=1) / math.sqrt(max(1, len(vv)))) if len(vv) > 1 else 0.0
            ce_w_unc_rows.append({**_ctx(cfg), "run_dir": rdir.name, "density_bin": di, "mean_w_width": mu, "ci95": 1.96 * se})

        # Weight uncertainty vs eta tertiles (top-1 rule)
        for ei in [0, 1, 2]:
            me = eb == ei
            if not np.any(me):
                continue
            vals = wwidth[me]
            mu = float(np.nanmean(vals))
            vv = vals[~np.isnan(vals)]
            se = float(np.std(vv, ddof=1) / math.sqrt(max(1, len(vv)))) if len(vv) > 1 else 0.0
            ce_w_unc_eta_rows.append({**_ctx(cfg), "run_dir": rdir.name, "eta_bin": ei, "mean_w_width": mu, "ci95": 1.96 * se})

        # Rule direction consistency (supporting predicted class) by density and eta tertiles
        sgn = np.sign(np.array([v["w"] for v in pts]))
        ypred = np.array([v["y_pred"] for v in pts])
        support = ((ypred == 1) & (sgn > 0)) | ((ypred == 0) & (sgn < 0))
        support = support.astype(float)
        for di in [0, 1, 2]:
            md = db == di
            if not np.any(md):
                continue
            rate = float(np.mean(support[md]))
            se = math.sqrt(rate * (1 - rate) / max(1, int(np.sum(md)))) if int(np.sum(md)) > 1 else 0.0
            ce_dir_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "density", "bin_id": di, "support_rate": rate, "ci95": 1.96 * se})
        for ei in [0, 1, 2]:
            me = eb == ei
            if not np.any(me):
                continue
            rate = float(np.mean(support[me]))
            se = math.sqrt(rate * (1 - rate) / max(1, int(np.sum(me)))) if int(np.sum(me)) > 1 else 0.0
            ce_dir_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "eta", "bin_id": ei, "support_rate": rate, "ci95": 1.96 * se})
        processed += 1
        if processed % 25 == 0 or processed == total_runs:
            print(f"[CE metrics] processed {processed}/{total_runs} runs for CE reliability")

    # Write CE reliability by weight
    with open(out / "ce_reliability_by_weight.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","weight_bin","bin","bin_edge_lo","bin_edge_hi","p_mean","acc_mean","count","acc_ci95",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ce_rel_rows:
            w.writerows(ce_rel_rows)

    with _open_w(out / "ce_ece_by_weight.csv") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","weight_bin","ece",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ce_ece_rows:
            w.writerows(ce_ece_rows)

    with open(out / "ce_weight_uncertainty_by_density.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","density_bin","mean_w_width","ci95",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ce_w_unc_rows:
            w.writerows(ce_w_unc_rows)

    with open(out / "ce_weight_uncertainty_by_eta.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","eta_bin","mean_w_width","ci95",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ce_w_unc_eta_rows:
            w.writerows(ce_w_unc_eta_rows)

    with open(out / "ce_rule_direction_consistency.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","bin_type","bin_id","support_rate","ci95",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ce_dir_rows:
            w.writerows(ce_dir_rows)

    # 1c) Effect-centric evaluation (classification factual) using stored rule_predict (model-consistent)
    eff_cov_rows: List[Dict[str, Any]] = []
    eff_mag_rows: List[Dict[str, Any]] = []
    eff_sign_rows: List[Dict[str, Any]] = []
    eff_rank_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        rules_path = rdir / "rules.jsonl"
        if not rules_path.exists():
            continue
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        recs: List[Dict[str, Any]] = []
        with open(rules_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("explanation_type") != "factual":
                    continue
                try:
                    wv = float(rec.get("w"))
                    wl = float(rec.get("w_low"))
                    wh = float(rec.get("w_high"))
                    rp = float(rec.get("rule_predict"))
                    pb = float(rec.get("p_pred"))
                except Exception:
                    continue
                recs.append({
                    "w": wv,
                    "w_low": wl,
                    "w_high": wh,
                    "delta": (rp - pb),
                    "inv_density": rec.get("inv_density"),
                    "eta": rec.get("eta"),
                })
        if not recs:
            continue
        import numpy as np
        w = np.array([r["w"] for r in recs], dtype=float)
        wl = np.array([r["w_low"] for r in recs], dtype=float)
        wh = np.array([r["w_high"] for r in recs], dtype=float)
        delta = np.array([r["delta"] for r in recs], dtype=float)
        invd = np.array([float(r["inv_density"]) if r["inv_density"] is not None else np.nan for r in recs], dtype=float)
        eta_arr = np.array([float(r["eta"]) if r["eta"] is not None else np.nan for r in recs], dtype=float)

        absw = np.abs(w)
        absd = np.abs(delta)
        covered = ((delta >= wl) & (delta <= wh)).astype(float)

        def tertiles(arr: np.ndarray) -> np.ndarray:
            x = arr.copy()
            if np.all(np.isnan(x)):
                return np.zeros_like(x)
            mu = np.nanmean(x)
            x[np.isnan(x)] = mu
            cuts = np.quantile(x, [0.33, 0.66])
            return np.digitize(x, cuts)

        db = tertiles(invd)
        eb = tertiles(eta_arr)
        wb = tertiles(absw)

        def _cov(mask: np.ndarray) -> Tuple[float, float, int]:
            if not np.any(mask):
                return (float("nan"), float("nan"), 0)
            vals = covered[mask]
            rate = float(np.mean(vals))
            n = int(np.sum(mask))
            se = (rate * (1 - rate) / max(1, n)) ** 0.5 if n > 1 else 0.0
            return (rate, 1.96 * se, n)

        for bi in [0, 1, 2]:
            r, ci, n = _cov(db == bi)
            eff_cov_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "density", "bin_id": bi, "coverage": r, "ci95": ci, "n": n})
        for bi in [0, 1, 2]:
            r, ci, n = _cov(eb == bi)
            eff_cov_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "eta", "bin_id": bi, "coverage": r, "ci95": ci, "n": n})
        for bi in [0, 1, 2]:
            r, ci, n = _cov(wb == bi)
            eff_cov_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "absw", "bin_id": bi, "coverage": r, "ci95": ci, "n": n})

        # Magnitude calibration by |w| tertiles
        for bi in [0, 1, 2]:
            m = wb == bi
            if not np.any(m):
                continue
            vals = absd[m]
            mu = float(np.nanmean(vals))
            se = float(np.nanstd(vals, ddof=1) / max(1, np.sqrt(np.sum(~np.isnan(vals))))) if np.sum(~np.isnan(vals)) > 1 else 0.0
            eff_mag_rows.append({**_ctx(cfg), "run_dir": rdir.name, "absw_bin": bi, "mean_abs_delta": mu, "ci95": 1.96 * se})

        # Sign consistency by density/eta
        sgn_ok = (np.sign(delta) == np.sign(w)).astype(float)
        for bi in [0, 1, 2]:
            m = db == bi
            if np.any(m):
                rate = float(np.mean(sgn_ok[m]))
                n = int(np.sum(m))
                se = (rate * (1 - rate) / max(1, n)) ** 0.5 if n > 1 else 0.0
                eff_sign_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "density", "bin_id": bi, "rate": rate, "ci95": 1.96 * se, "n": n})
        for bi in [0, 1, 2]:
            m = eb == bi
            if np.any(m):
                rate = float(np.mean(sgn_ok[m]))
                n = int(np.sum(m))
                se = (rate * (1 - rate) / max(1, n)) ** 0.5 if n > 1 else 0.0
                eff_sign_rows.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": "eta", "bin_id": bi, "rate": rate, "ci95": 1.96 * se, "n": n})

        eff_rank_rows.append({**_ctx(cfg), "run_dir": rdir.name, "spearman_absw_absdelta": _rank_corr(absw.tolist(), absd.tolist())})

    with open(out / "effect_interval_coverage.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","bin_type","bin_id","coverage","ci95","n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if eff_cov_rows:
            w.writerows(eff_cov_rows)

    with open(out / "effect_magnitude_calibration.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","absw_bin","mean_abs_delta","ci95",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if eff_mag_rows:
            w.writerows(eff_mag_rows)

    with open(out / "effect_sign_consistency.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","bin_type","bin_id","rate","ci95","n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if eff_sign_rows:
            w.writerows(eff_sign_rows)

    with open(out / "effect_rank_correlation.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","spearman_absw_absdelta",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if eff_rank_rows:
            w.writerows(eff_rank_rows)

    # 1d) Strict coverage via light fresh sampling (rebuild data/model and CE; match stored intervals)
    strict_rows: List[Dict[str, Any]] = []
    try:
        from evaluation.uncertainty_experiments.data_gen.classification import make_binary_gaussian as _mk
        from evaluation.uncertainty_experiments.models.wrappers import make_model as _mk_model
        from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer as _W
    except Exception:
        _mk = None  # type: ignore
        _mk_model = None  # type: ignore
        _W = None  # type: ignore

    if _mk is not None and _mk_model is not None and _W is not None:
        for rdir in runs:
            try:
                cfg = _load_json(rdir / "config.json")
                art = _load_json(rdir / "artifacts.json")
            except Exception:
                continue
            if cfg.get("task") != "classification":
                continue
            rules_path = rdir / "rules.jsonl"
            if not rules_path.exists():
                continue
            # Load stored intervals keyed by (point_id, antecedent_str)
            stored: Dict[Tuple[int, str], Tuple[float, float]] = {}
            with open(rules_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("explanation_type") != "factual":
                        continue
                    pid = rec.get("point_id")
                    rule = rec.get("antecedent_str")
                    if pid is None or rule is None:
                        continue
                    try:
                        wl = float(rec.get("w_low"))
                        wh = float(rec.get("w_high"))
                    except Exception:
                        continue
                    stored[(int(pid), str(rule))] = (wl, wh)
            if not stored:
                continue
            # Re-generate data and rebuild model/CE (same data seed; CE seed offset for fresh sampling)
            data = _mk(
                n_train=cfg["data"]["n_train"],
                n_cal=cfg["data"]["n_cal"],
                n_test=cfg["data"]["n_test"],
                dims=cfg["data"].get("dims", 2),
                holes=cfg["data"].get("holes"),
                shift=cfg["data"].get("shift", 0.0),
                seed=cfg.get("seed", 42),
            )
            mw = _mk_model("classification", cfg["model"]["type"], cfg["model"]["params"]).fit(data["X_train"], data["y_train"])  # type: ignore[attr-defined]
            W = _W(mw.model)
            # Use a different seed to trigger independent sampling in CE
            W.calibrate(data["X_cal"], data["y_cal"], seed=int(cfg.get("seed", 42)) + 777)
            ce_new = W.explain_factual(data["X_test"])  # type: ignore
            # Get bins for density/eta from stored artifacts
            import numpy as np
            invd = np.array(art.get("inv_density_test", []), dtype=float)
            eta = np.array(art.get("eta_test", []), dtype=float) if art.get("eta_test") is not None else np.full(len(data["X_test"]), np.nan)
            def tertiles(arr: np.ndarray) -> np.ndarray:
                x = arr.copy()
                if np.all(np.isnan(x)):
                    return np.zeros_like(x)
                mu = np.nanmean(x)
                x[np.isnan(x)] = mu
                cuts = np.quantile(x, [0.33, 0.66])
                return np.digitize(x, cuts)
            db = tertiles(invd) if invd.size else np.zeros(len(data["X_test"]))
            eb = tertiles(eta) if eta.size else np.zeros(len(data["X_test"]))
            # Sample limit
            SAMPLE_POINTS = int((cfg.get("faithfulness", {}) or {}).get("sample_points", 200))
            n = len(ce_new.explanations)
            idxs = list(range(min(n, SAMPLE_POINTS)))
            for pid in idxs:
                exp = ce_new.explanations[pid]
                rules = exp._get_rules()  # noqa: SLF001
                if not rules.get("rule"):
                    continue
                # rank by |weight|
                try:
                    import numpy as _np
                    order = list(_np.argsort(_np.abs(_np.array(rules.get("weight", [])))))
                    order.reverse()
                except Exception:
                    order = list(range(len(rules.get("rule", []))))
                idx = order[0]
                antecedent = str(rules["rule"][idx]).strip()
                key = (pid, antecedent)
                if key not in stored:
                    continue
                wl, wh = stored[key]
                try:
                    rp = float(rules["predict"][idx])
                    pb = float(exp.prediction.get("predict"))
                    delta = rp - pb
                except Exception:
                    continue
                bin_d = int(db[pid]) if invd.size else 0
                bin_e = int(eb[pid]) if eta.size else 0
                inside = (delta >= wl) and (delta <= wh)
                strict_rows.append({**_ctx(cfg), "run_dir": rdir.name, "point_id": pid, "bin_type": "density", "bin_id": bin_d, "inside": int(inside)})
                strict_rows.append({**_ctx(cfg), "run_dir": rdir.name, "point_id": pid, "bin_type": "eta", "bin_id": bin_e, "inside": int(inside)})

    # Aggregate strict coverage
    strict_cov_rows: List[Dict[str, Any]] = []
    if strict_rows:
        by_key: Dict[Tuple[str, str, int], List[int]] = {}
        for r in strict_rows:
            k = (r.get("run_dir"), r.get("bin_type"), int(r.get("bin_id")))
            by_key.setdefault(k, []).append(int(r.get("inside", 0)))
        for (rd, bt, bi), vals in by_key.items():
            import numpy as np
            rate = float(np.mean(vals))
            n = len(vals)
            se = (rate * (1 - rate) / max(1, n)) ** 0.5 if n > 1 else 0.0
            # Retrieve a representative cfg for context
            cfg = None
            for rdir in runs:
                if (rdir.name == rd):
                    try:
                        cfg = _load_json(rdir / "config.json")
                    except Exception:
                        cfg = None
                    break
            row_ctx = _ctx(cfg) if cfg else {}
            row_ctx.update({"run_dir": rd, "bin_type": bt, "bin_id": int(bi), "coverage": rate, "ci95": 1.96 * se, "n": n})
            strict_cov_rows.append(row_ctx)

    with open(out / "effect_interval_coverage_strict.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","bin_type","bin_id","coverage","ci95","n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if strict_cov_rows:
            w.writerows(strict_cov_rows)

    # 4) Stability across seeds (top-1 rule exact match), stratified by density/eta
    # Group runs by config excluding seed
    def _group_key(cfg: Dict[str, Any]) -> str:
        g = dict(cfg)
        g.pop("seed", None)
        return json.dumps(g, sort_keys=True, default=str)

    groups: Dict[str, List[Path]] = {}
    cfgs: Dict[str, Dict[str, Any]] = {}
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        k = _group_key(cfg)
        groups.setdefault(k, []).append(rdir)
        cfgs[k] = cfg

    stability_rows: List[Dict[str, Any]] = []
    for gk, rlist in groups.items():
        if len(rlist) < 2:
            continue
        # Build per-run point->top rule_id and covariates
        per_run: List[Dict[int, Dict[str, Any]]]= []
        for rdir in rlist:
            rules_path = rdir / "rules.jsonl"
            if not rules_path.exists():
                continue
            by_pt: Dict[int, Dict[str, Any]] = {}
            with open(rules_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("explanation_type") != "factual":
                        continue
                    pid = int(rec.get("point_id", -1))
                    if pid < 0:
                        continue
                    wabs = abs(float(rec.get("w", 0.0)))
                    cur = by_pt.get(pid)
                    if cur is None or wabs > cur.get("wabs", -1):
                        by_pt[pid] = {
                            "wabs": wabs,
                            "rule_id": rec.get("rule_id"),
                            "inv_density": rec.get("inv_density"),
                            "eta": rec.get("eta"),
                        }
            if by_pt:
                per_run.append(by_pt)
        if len(per_run) < 2:
            continue
        # Intersect common points
        common_ids = set(per_run[0].keys())
        for m in per_run[1:]:
            common_ids &= set(m.keys())
        if not common_ids:
            continue
        import numpy as np
        # Use covariates from first run for binning
        invd = np.array([per_run[0][pid].get("inv_density", np.nan) for pid in sorted(common_ids)], dtype=float)
        eta_arr = np.array([per_run[0][pid].get("eta", np.nan) for pid in sorted(common_ids)], dtype=float)
        def tertiles(arr: np.ndarray) -> np.ndarray:
            x = arr.copy()
            if np.all(np.isnan(x)):
                return np.zeros_like(x)
            mu = np.nanmean(x)
            x[np.isnan(x)] = mu
            cuts = np.quantile(x, [0.33, 0.66])
            return np.digitize(x, cuts)
        db = tertiles(invd)
        eb = tertiles(eta_arr)
        # Compute exact-match across runs per point
        rule_matrix: List[List[str]] = []
        ids_sorted = sorted(common_ids)
        for m in per_run:
            rule_matrix.append([m[pid]["rule_id"] for pid in ids_sorted])
        rule_matrix = np.array(rule_matrix)
        # exact if all equal across axis 0 for each point
        exact = np.all(rule_matrix == rule_matrix[0:1, :], axis=0).astype(float)
        def _agg(mask: np.ndarray) -> Dict[str, float]:
            if not np.any(mask):
                return {"rate": float("nan"), "ci95": float("nan"), "n": 0}
            vals = exact[mask]
            rate = float(np.mean(vals))
            n = int(np.sum(mask))
            se = (rate * (1 - rate) / max(1, n)) ** 0.5 if n > 1 else 0.0
            return {"rate": rate, "ci95": 1.96 * se, "n": n}
        # Aggregate by density bins
        for bi in [0, 1, 2]:
            res = _agg(db == bi)
            stability_rows.append({**_ctx(cfgs[gk]), "run_group_id": hash(gk), "n_runs": len(per_run), "bin_type": "density", "bin_id": bi, "exact_match_rate": res["rate"], "ci95": res["ci95"], "n_points": res["n"]})
        # Aggregate by eta bins
        for bi in [0, 1, 2]:
            res = _agg(eb == bi)
            stability_rows.append({**_ctx(cfgs[gk]), "run_group_id": hash(gk), "n_runs": len(per_run), "bin_type": "eta", "bin_id": bi, "exact_match_rate": res["rate"], "ci95": res["ci95"], "n_points": res["n"]})

    with open(out / "rule_stability.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_group_id","n_runs","bin_type","bin_id","exact_match_rate","ci95","n_points",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if stability_rows:
            w.writerows(stability_rows)

    # 4a) Stability Jaccard@k across seeds (top-k rule sets), stratified by density/eta
    jacc_rows: List[Dict[str, Any]] = []
    K = int(jaccard_k)
    for gk, rlist in groups.items():
        if len(rlist) < 2:
            continue
        # Build per-run point->topK rule_ids and covariates
        per_run_topk: List[Dict[int, Dict[str, Any]]] = []
        for rdir in rlist:
            rules_path = rdir / "rules.jsonl"
            if not rules_path.exists():
                continue
            by_pt: Dict[int, Dict[str, Any]] = {}
            # Collect rules per point
            temp: Dict[int, List[Dict[str, Any]]] = {}
            with open(rules_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("explanation_type") != "factual":
                        continue
                    pid = int(rec.get("point_id", -1))
                    if pid < 0:
                        continue
                    temp.setdefault(pid, []).append(rec)
            if not temp:
                continue
            for pid, lst in temp.items():
                # rank by absolute weight descending
                try:
                    lst_sorted = sorted(lst, key=lambda r: abs(float(r.get("w", 0.0))), reverse=True)
                except Exception:
                    lst_sorted = lst
                topk = [r.get("rule_id") for r in lst_sorted[:K]]
                if not topk:
                    continue
                # take covariates from the strongest rule record
                invd = lst_sorted[0].get("inv_density")
                eta_v = lst_sorted[0].get("eta")
                by_pt[pid] = {"topk": topk, "inv_density": invd, "eta": eta_v}
            if by_pt:
                per_run_topk.append(by_pt)
        if len(per_run_topk) < 2:
            continue
        # Common points
        common_ids = set(per_run_topk[0].keys())
        for m in per_run_topk[1:]:
            common_ids &= set(m.keys())
        if not common_ids:
            continue
        ids_sorted = sorted(common_ids)
        import numpy as np
        # Use covariates from first run for binning
        invd = np.array([per_run_topk[0][pid].get("inv_density", np.nan) for pid in ids_sorted], dtype=float)
        eta_arr = np.array([per_run_topk[0][pid].get("eta", np.nan) for pid in ids_sorted], dtype=float)
        def tertiles(arr: np.ndarray) -> np.ndarray:
            x = arr.copy()
            if np.all(np.isnan(x)):
                return np.zeros_like(x)
            mu = np.nanmean(x)
            x[np.isnan(x)] = mu
            cuts = np.quantile(x, [0.33, 0.66])
            return np.digitize(x, cuts)
        db = tertiles(invd)
        eb = tertiles(eta_arr)
        # Compute average pairwise Jaccard per point
        def avg_pairwise_jaccard(idx_mask: np.ndarray) -> float:
            from itertools import combinations
            ids = [pid for i, pid in enumerate(ids_sorted) if idx_mask[i]]
            if len(ids) == 0:
                return float('nan')
            acc = []
            pairs = list(combinations(range(len(per_run_topk)), 2))
            if not pairs:
                return float('nan')
            for pid in ids:
                sets = [set(m[pid]["topk"]) for m in per_run_topk if pid in m]
                for (a, b) in pairs:
                    try:
                        s1 = set(per_run_topk[a][pid]["topk"]) ; s2 = set(per_run_topk[b][pid]["topk"]) 
                        inter = len(s1 & s2) ; uni = len(s1 | s2)
                        acc.append((inter / uni) if uni > 0 else 0.0)
                    except Exception:
                        continue
            if not acc:
                return float('nan')
            return float(sum(acc) / len(acc))
        # Aggregate by bins
        for bi in [0, 1, 2]:
            val = avg_pairwise_jaccard(db == bi)
            jacc_rows.append({**_ctx(cfgs[gk]), "run_group_id": hash(gk), "n_runs": len(per_run_topk), "bin_type": "density", "bin_id": bi, "jaccard_k": K, "avg_jaccard": val})
        for bi in [0, 1, 2]:
            val = avg_pairwise_jaccard(eb == bi)
            jacc_rows.append({**_ctx(cfgs[gk]), "run_group_id": hash(gk), "n_runs": len(per_run_topk), "bin_type": "eta", "bin_id": bi, "jaccard_k": K, "avg_jaccard": val})

    with open(out / "rule_stability_jaccard.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_group_id","n_runs","bin_type","bin_id","jaccard_k","avg_jaccard",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if jacc_rows:
            w.writerows(jacc_rows)

    # Compute stability deltas per group (high-low)
    stability_delta_by_group: Dict[int, Dict[str, float]] = {}
    for bt in ("density", "eta"):
        # Organize by group and bin
        by_group: Dict[int, Dict[int, float]] = {}
        for r in stability_rows:
            if r.get("bin_type") != bt:
                continue
            gid = int(r.get("run_group_id"))
            bin_id = int(r.get("bin_id"))
            rate = float(r.get("exact_match_rate"))
            by_group.setdefault(gid, {})[bin_id] = rate
        for gid, bins in by_group.items():
            lo = bins.get(0, float("nan"))
            hi = bins.get(2, float("nan"))
            if gid not in stability_delta_by_group:
                stability_delta_by_group[gid] = {}
            stability_delta_by_group[gid][f"stability_delta_{bt}"] = (hi - lo) if (not math.isnan(hi) and not math.isnan(lo)) else float("nan")

    # 5) Faithfulness (Rule Causal Sensitivity@k)
    faith_rows: List[Dict[str, Any]] = []
    try:
        # Lazy imports for generation and CE
        from evaluation.uncertainty_experiments.data_gen.classification import make_binary_gaussian as _make_cls
        from evaluation.uncertainty_experiments.models.wrappers import make_model as _make_model
        from evaluation.uncertainty_experiments.rules.hooks import extract_factual_rule_records_for_classification as _extract_rules
        from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer as _W
    except Exception:
        _make_cls = None  # type: ignore
        _make_model = None  # type: ignore
        _extract_rules = None  # type: ignore
        _W = None  # type: ignore

    if _make_cls is not None and _make_model is not None and _W is not None:
        # Build classification run list
        cls_runs: List[Path] = []
        for rdir in runs:
            try:
                cfg = _load_json(rdir / "config.json")
            except Exception:
                continue
            if cfg.get("task") == "classification":
                cls_runs.append(rdir)

        # Per-run computation (optionally parallel)
        def _faith_for_run(rdir: Path) -> List[Dict[str, Any]]:
            rows_local: List[Dict[str, Any]] = []
            try:
                cfg = _load_json(rdir / "config.json")
            except Exception:
                return rows_local
            # Per-run config knobs
            faith_cfg = cfg.get("faithfulness", {}) or {}
            K = int(faith_cfg.get("k", 8))
            SAMPLE_POINTS = int(faith_cfg.get("sample_points", 200))
            # Re-generate data using config
            data = _make_cls(
                n_train=cfg["data"]["n_train"],
                n_cal=cfg["data"]["n_cal"],
                n_test=cfg["data"]["n_test"],
                dims=cfg["data"].get("dims", 2),
                holes=cfg["data"].get("holes"),
                shift=cfg["data"].get("shift", 0.0),
                seed=cfg.get("seed", 42),
            )
            mw = _make_model("classification", cfg["model"]["type"], cfg["model"]["params"])  # wrapper
            mw.fit(data["X_train"], data["y_train"])
            w = _W(mw.model).calibrate(data["X_cal"], data["y_cal"], seed=cfg.get("seed", 42))
            ce = w.explain_factual(data["X_test"])  # explanations to parse rules

            # Base probabilities per instance
            try:
                base_proba = w.explainer.predict_proba(data["X_test"])  # type: ignore[attr-defined]
            except Exception:
                base_proba = None
            if base_proba is None:
                return
            import numpy as np
            p1 = base_proba[:, 1] if base_proba.ndim == 2 and base_proba.shape[1] > 1 else np.asarray(base_proba).reshape(-1)

            # Select subset of points for runtime
            N = len(data["X_test"])
            idx_all = np.arange(N)
            if N > SAMPLE_POINTS:
                rng = np.random.default_rng(int(cfg.get("seed", 0)))
                idx = np.sort(rng.choice(N, size=SAMPLE_POINTS, replace=False))
            else:
                idx = idx_all

            # Helper: parse antecedent string "feature < t" or "feature > t" or "feature = cat"
            def _parse_rule(rule: str) -> Tuple[str, str, str]:
                if "<" in rule:
                    parts = rule.split("<")
                    return parts[0].strip(), "<", parts[1].strip()
                if ">" in rule:
                    parts = rule.split(">")
                    return parts[0].strip(), ">", parts[1].strip()
                if "=" in rule:
                    parts = rule.split("=")
                    return parts[0].strip(), "=", parts[1].strip()
                return rule, "?", ""

            X_cal = data["X_cal"]
            X_test = data["X_test"].copy()
            # Feature names best-effort
            feat_names = [str(i) for i in range(X_test.shape[1])]
            try:
                feat_names = list(w.explainer.feature_names)  # type: ignore[attr-defined]
            except Exception:
                pass
            name_to_idx = {n: i for i, n in enumerate(feat_names)}

            # Compute per-instance top-1 rule and jitter deltas
            deltas: List[float] = []  # overall, not used in output aggregation directly
            # density/eta for stratification
            invd = None
            eta = None
            try:
                a = _load_json(rdir / "artifacts.json")
                invd = np.array(a.get("inv_density_test", []), dtype=float)
                eta = np.array(a.get("eta_test", []), dtype=float) if a.get("eta_test") is not None else None
            except Exception:
                pass
            bins_density = None
            bins_eta = None
            if invd is not None and invd.size == len(X_test):
                cuts = np.quantile(invd, [0.33, 0.66])
                bins_density = np.digitize(invd, cuts)
            if eta is not None and len(eta) == len(X_test):
                cuts = np.quantile(eta, [0.33, 0.66])
                bins_eta = np.digitize(eta, cuts)

            # For plotting, accumulate per-bin deltas
            delta_density_in: Dict[int, List[float]] = {0: [], 1: [], 2: []}
            delta_density_out: Dict[int, List[float]] = {0: [], 1: [], 2: []}
            delta_eta_in: Dict[int, List[float]] = {0: [], 1: [], 2: []}
            delta_eta_out: Dict[int, List[float]] = {0: [], 1: [], 2: []}

            for i in idx:
                exp = ce.explanations[i]
                rules = exp._get_rules()  # noqa: SLF001
                if not rules.get("rule"):
                    continue
                # top-1 by |w|
                weights = np.asarray(rules["weight"]) if "weight" in rules else None
                if weights is None:
                    continue
                top_idx = int(np.argmax(np.abs(weights)))
                rule_str = str(rules["rule"][top_idx])
                fname, op, thr_str = _parse_rule(rule_str)
                if fname not in name_to_idx:
                    continue
                f = name_to_idx[fname]
                base_p = float(p1[i])
                rng = np.random.default_rng(int(cfg.get("seed", 0)) + i)
                mean_in = None
                mean_out = None
                if op == "=":
                    # Categorical: in-region (same category) has zero delta; out-of-region: switch to other cats from cal
                    try:
                        cur_val = X_test[i, f]
                        others = X_cal[X_cal[:, f] != cur_val, f]
                        if others.size > 0:
                            take_out = others if others.size <= K else rng.choice(others, size=K, replace=False)
                            outs: List[float] = []
                            for v in take_out:
                                x = X_test[i, :].copy()
                                x[f] = v
                                proba = w.explainer.predict_proba(x.reshape(1, -1))  # type: ignore[attr-defined]
                                pv = proba[0, 1] if proba.ndim == 2 and proba.shape[1] > 1 else float(np.asarray(proba).reshape(-1)[0])
                                outs.append(abs(pv - base_p))
                            mean_out = float(np.mean(outs)) if outs else None
                        mean_in = 0.0
                    except Exception:
                        pass
                else:
                    # Numeric: parse threshold and sample both in-region and out-of-region sides
                    try:
                        thr = float(thr_str)
                    except Exception:
                        continue
                    if op == "<":
                        cand_in = X_cal[X_cal[:, f] < thr, f]
                        cand_out = X_cal[X_cal[:, f] >= thr, f]
                    else:
                        cand_in = X_cal[X_cal[:, f] > thr, f]
                        cand_out = X_cal[X_cal[:, f] <= thr, f]
                    if cand_in.size > 0:
                        take_in = cand_in if cand_in.size <= K else rng.choice(cand_in, size=K, replace=False)
                        ins: List[float] = []
                        for v in take_in:
                            x = X_test[i, :].copy()
                            x[f] = v
                            proba = w.explainer.predict_proba(x.reshape(1, -1))  # type: ignore[attr-defined]
                            pv = proba[0, 1] if proba.ndim == 2 and proba.shape[1] > 1 else float(np.asarray(proba).reshape(-1)[0])
                            ins.append(abs(pv - base_p))
                        mean_in = float(np.mean(ins)) if ins else None
                    if cand_out.size > 0:
                        take_out = cand_out if cand_out.size <= K else rng.choice(cand_out, size=K, replace=False)
                        outs: List[float] = []
                        for v in take_out:
                            x = X_test[i, :].copy()
                            x[f] = v
                            proba = w.explainer.predict_proba(x.reshape(1, -1))  # type: ignore[attr-defined]
                            pv = proba[0, 1] if proba.ndim == 2 and proba.shape[1] > 1 else float(np.asarray(proba).reshape(-1)[0])
                            outs.append(abs(pv - base_p))
                        mean_out = float(np.mean(outs)) if outs else None
                # Accumulate by bins
                if bins_density is not None:
                    bi = int(bins_density[i])
                    if mean_in is not None:
                        delta_density_in[bi].append(mean_in)
                    if mean_out is not None:
                        delta_density_out[bi].append(mean_out)
                if bins_eta is not None:
                    ei = int(bins_eta[i])
                    if mean_in is not None:
                        delta_eta_in[ei].append(mean_in)
                    if mean_out is not None:
                        delta_eta_out[ei].append(mean_out)

            # Aggregate into rows per run
            def _emit(rows_map: Dict[int, List[float]], btype: str, region: str) -> None:
                for bi in [0, 1, 2]:
                    vals = rows_map.get(bi, [])
                    if not vals:
                        continue
                    m = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / math.sqrt(max(1, len(vals)))) if len(vals) > 1 else 0.0
                    rows_local.append({**_ctx(cfg), "run_dir": rdir.name, "bin_type": btype, "region": region, "bin_id": bi, "mean_delta": m, "ci95": 1.96 * se, "n": len(vals)})

            if bins_density is not None:
                _emit(delta_density_in, "density", "in")
                _emit(delta_density_out, "density", "out")
            if bins_eta is not None:
                _emit(delta_eta_in, "eta", "in")
                _emit(delta_eta_out, "eta", "out")

            return rows_local

        # Execute
        if Parallel is not None and len(cls_runs) > 1:
            results = Parallel(n_jobs=-1, backend="loky", prefer="processes")(
                delayed(_faith_for_run)(rdir) for rdir in cls_runs
            )
            for rows_local in results:
                faith_rows.extend(rows_local)
        else:
            for rdir in cls_runs:
                faith_rows.extend(_faith_for_run(rdir))

    with open(out / "rule_faithfulness.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_dir","bin_type","region","bin_id","mean_delta","ci95","n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if faith_rows:
            w.writerows(faith_rows)

    # (Moved) Uncertainty Sensitivity Indices computed after utility AUC is available

    # 2) Classification: ECE by n_cal
    ece_rows: Dict[Tuple[Any, Any, Any, Any, Any, Any], List[float]] = {}
    key_ctx: Dict[Tuple[Any, Any, Any, Any, Any, Any], Dict[str, Any]] = {}
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            m = _load_json(rdir / "metrics.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        c = _ctx(cfg)
        key = (c["model"], c["dims"], c["n_cal"], c["shift"], c["holes_label"], c["calib_classification"])
        ece_rows.setdefault(key, []).append(float(m.get("ece", float("nan"))))
        key_ctx[key] = c
    ece_agg_rows: List[Dict[str, Any]] = []
    for key, vals in ece_rows.items():
        c = dict(key_ctx[key])
        c.pop("seed", None)  # aggregated over seeds
        mean, ci = _mean_ci95([v for v in vals if v == v])
        ece_agg_rows.append({**c, "mean_ece": mean, "ci95": ci, "n": len(vals)})
    with open(out / "ece_by_ncal.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "mean_ece","ci95","n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ece_agg_rows:
            w.writerows(ece_agg_rows)

    # 2b) Classification: ECE by density tertiles
    ece_by_dens_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        import numpy as np
        invd = np.array(a.get("inv_density_test", []), dtype=float)
        probs = a.get("probs")
        y = a.get("y_test")
        if invd.size == 0 or probs is None or y is None:
            continue
        invd = (invd - invd.mean()) / (invd.std() + 1e-12)
        cuts = np.quantile(invd, [0.33, 0.66])
        bins = np.digitize(invd, cuts)
        probs_np = np.array(probs)
        y_np = np.array(y)
        from evaluation.uncertainty_experiments.metrics.metrics import ece as _ece
        for b in [0, 1, 2]:
            mask = bins == b
            if not mask.any():
                continue
            e = _ece(probs_np[mask], y_np[mask])
            ece_by_dens_rows.append({**_ctx(cfg), "density_bin": b, "ece": e})
    with open(out / "ece_by_density.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "density_bin","ece",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if ece_by_dens_rows:
            w.writerows(ece_by_dens_rows)

    # 3) Thresholded regression reliability across t
    th_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            m = _load_json(rdir / "metrics.json")
        except Exception:
            continue
        if cfg.get("task") not in ("thresh_reg", "thresholded", "thresholded_regression"):
            continue
        per_t = m.get("per_threshold", {})
        for tkey, md in per_t.items():
            th_rows.append({**_ctx(cfg), "threshold": tkey, "ece": md.get("ece"), "brier": md.get("brier")})
    with open(out / "thresh_reg_reliability.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "threshold","ece","brier",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if th_rows:
            w.writerows(th_rows)

    # Baseline rule-level reliability (classification): read baseline_rules_*.jsonl
    bl_rule_rows: List[Dict[str, Any]] = []
    bl_ece_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        # For each baseline file
        for path in rdir.glob("baseline_rules_*.jsonl"):
            baseline_name = path.stem.replace("baseline_rules_", "")
            recs: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("y_true") is None:
                        continue
                    # Use leaf probability as rule probability proxy
                    p = rec.get("p_pred_base")
                    try:
                        pv = float(p)
                    except Exception:
                        continue
                    recs.append({"p": pv, "y": int(rec.get("y_true"))})
            if not recs:
                continue
            ps = [r["p"] for r in recs]
            ys = [r["y"] for r in recs]
            bins, edges = _bin_indices(ps, n_bins=10)
            total = max(1, len(ps))
            ece_sum = 0.0
            for b in range(1, 11):
                idx = [i for i, bb in enumerate(bins) if bb == b]
                if not idx:
                    continue
                p_mean = sum(ps[i] for i in idx) / len(idx)
                acc_mean = sum(ys[i] for i in idx) / len(idx)
                n = len(idx)
                se = math.sqrt(acc_mean * (1 - acc_mean) / max(n, 1)) if n > 1 else 0.0
                bl_rule_rows.append({**_ctx(cfg), "run_dir": rdir.name, "baseline": baseline_name, "bin": b, "bin_edge_lo": edges[b-1], "bin_edge_hi": edges[b], "p_mean": p_mean, "acc_mean": acc_mean, "count": n, "acc_ci95": 1.96 * se})
                ece_sum += abs(acc_mean - p_mean) * (n / total)
            bl_ece_rows.append({**_ctx(cfg), "run_dir": rdir.name, "baseline": baseline_name, "ece": ece_sum})

    with open(out / "baseline_rule_reliability_by_bin.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_dir","baseline","bin","bin_edge_lo","bin_edge_hi","p_mean","acc_mean","count","acc_ci95",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if bl_rule_rows:
            w.writerows(bl_rule_rows)

    with open(out / "baseline_rule_ece.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_dir","baseline","ece",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if bl_ece_rows:
            w.writerows(bl_ece_rows)

    # 3a) Thresholded regression: Rule-level ECE across t from rules.jsonl
    th_rule_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        rules_path = rdir / "rules.jsonl"
        if not rules_path.exists():
            # Fallback handled below (recompute on the fly if needed)
            pass
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        if cfg.get("task") not in ("thresh_reg", "thresholded", "thresholded_regression"):
            continue
        # Load rules; require p_rule_t dict and y_true
        recs: List[Dict[str, Any]] = []
        if rules_path.exists():
            with open(rules_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("explanation_type") != "factual":
                        continue
                    if rec.get("y_true") is None:
                        continue
                    if not isinstance(rec.get("p_rule_t"), dict):
                        continue
                    recs.append(rec)
        # If missing on-disk rules for thresholded regression, rebuild lightweight per-rule records on the fly
        if not recs:
            try:
                from evaluation.uncertainty_experiments.data_gen.regression import generate_regression as _mk_reg  # type: ignore
                from evaluation.uncertainty_experiments.models.wrappers import make_model as _mk_model_reg  # type: ignore
                from evaluation.uncertainty_experiments.rules.hooks import (
                    extract_factual_rule_records_for_thresholded_regression as _extract_th_rules,  # type: ignore
                )
            except Exception:
                _mk_reg = None  # type: ignore
                _mk_model_reg = None  # type: ignore
                _extract_th_rules = None  # type: ignore
            if _mk_reg is not None and _mk_model_reg is not None and _extract_th_rules is not None:
                try:
                    # Rebuild synthetic data consistent with this run's config
                    d = cfg.get("data", {}) or {}
                    hetero = d.get("hetero") if isinstance(d.get("hetero"), dict) else None
                    data = _mk_reg(
                        n_train=int(d.get("n_train", 1000)),
                        n_cal=int(d.get("n_cal", 100)),
                        n_test=int(d.get("n_test", 1000)),
                        dims=int(d.get("dims", 2)),
                        holes=d.get("holes"),
                        shift=float(d.get("shift", 0.0) or 0.0),
                        hetero=hetero,
                        seed=int(cfg.get("seed", 42)),
                    )
                    # Build and fit learner
                    model_cfg = cfg.get("model", {}) or {}
                    mw = _mk_model_reg("regression", model_cfg.get("type", "dt"), model_cfg.get("params", {}) or {})
                    mw.fit(data["X_train"], data["y_train"])  # type: ignore[attr-defined]
                    # Determine thresholds from metrics.json if available
                    t_keys: List[str] = []
                    try:
                        m = _load_json(rdir / "metrics.json")
                        pt = m.get("per_threshold", {}) if isinstance(m, dict) else {}
                        t_keys = list(pt.keys())
                    except Exception:
                        t_keys = []
                    if not t_keys:
                        # fallback to config t_values list (may be 'auto')
                        t_cfg = ((cfg.get("calibration", {}) or {}).get("thresholded", {}) or {}).get("t_values")
                        if isinstance(t_cfg, (list, tuple)):
                            t_keys = [f"t={float(v):.4g}" for v in t_cfg]
                    # Parse to floats
                    t_values: List[float] = []
                    for k in t_keys:
                        try:
                            t_values.append(float(str(k).split("=")[-1]))
                        except Exception:
                            continue
                    if t_values:
                        # Extract per-rule records with per-threshold probabilities
                        recs = _extract_th_rules(
                            mw.model,
                            data["X_cal"],
                            data["y_cal"],
                            data["X_test"],
                            data["y_test"],
                            t_values,
                            de_test=None,
                            inv_density_test=None,
                            sigma_test=data.get("sigma_test"),
                            feature_names=None,
                            seed=int(cfg.get("seed", 42)),
                        )
                except Exception:
                    recs = []
        if not recs:
            continue
        # Collect all t keys
        keys: List[str] = sorted(set([k for r in recs for k in (r.get("p_rule_t") or {}).keys()]))
        import numpy as np
        y_vals = np.array([float(r.get("y_true")) for r in recs], dtype=float)
        for key in keys:
            # Build p and labels
            ps: List[float] = []
            ys: List[int] = []
            try:
                t_float = float(str(key).split("=")[-1])
            except Exception:
                # Try to parse as raw float string
                try:
                    t_float = float(key)
                except Exception:
                    continue
            for r in recs:
                pmap = r.get("p_rule_t") or {}
                if key not in pmap:
                    continue
                p = pmap.get(key)
                try:
                    pv = float(p)
                except Exception:
                    continue
                ps.append(pv)
                ys.append(int(float(r.get("y_true")) > t_float))
            if not ps:
                continue
            bins, edges = _bin_indices(ps, n_bins=10)
            total = max(1, len(ps))
            ece_sum = 0.0
            for b in range(1, 11):
                m = [i for i, bb in enumerate(bins) if bb == b]
                if not m:
                    continue
                p_mean = sum(ps[i] for i in m) / len(m)
                acc_mean = sum(ys[i] for i in m) / len(m)
                n = len(m)
                se = math.sqrt(acc_mean * (1 - acc_mean) / max(n, 1)) if n > 1 else 0.0
                th_rule_rows.append({**_ctx(cfg), "run_dir": rdir.name, "threshold": key, "bin": b, "bin_edge_lo": edges[b-1], "bin_edge_hi": edges[b], "p_mean": p_mean, "acc_mean": acc_mean, "count": n, "acc_ci95": 1.96 * se})
                ece_sum += abs(acc_mean - p_mean) * (n / total)
            # Also emit a compact ECE row
            th_rule_rows.append({**_ctx(cfg), "run_dir": rdir.name, "threshold": key, "bin": 0, "bin_edge_lo": 0.0, "bin_edge_hi": 1.0, "p_mean": float('nan'), "acc_mean": float('nan'), "count": total, "acc_ci95": float('nan'), "rule_ece": ece_sum})

    with open(out / "thresh_reg_rule_reliability_by_bin.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_dir","threshold","bin","bin_edge_lo","bin_edge_hi","p_mean","acc_mean","count","acc_ci95","rule_ece",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if th_rule_rows:
            w.writerows(th_rule_rows)

    # 3b) Regression: conditional coverage by density tertiles
    reg_dens_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "regression":
            continue
        import numpy as np
        invd = np.array(a.get("inv_density_test", []), dtype=float)
        y = np.array(a.get("y_test", []), dtype=float)
        lo = np.array(a.get("pi_lower", []), dtype=float)
        hi = np.array(a.get("pi_upper", []), dtype=float)
        if invd.size == 0:
            continue
        cuts = np.quantile(invd, [0.33, 0.66])
        bins = np.digitize(invd, cuts)
        for b in [0, 1, 2]:
            mask = bins == b
            if not mask.any():
                continue
            covered = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
            width = (hi[mask] - lo[mask])
            reg_dens_rows.append({**_ctx(cfg), "density_bin": b, "coverage": float(covered.mean()), "avg_width": float(width.mean())})
    with open(out / "coverage_by_density.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "density_bin","coverage","avg_width",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if reg_dens_rows:
            w.writerows(reg_dens_rows)

    # 4) Correlations: DE vs sigma (regression)
    corr_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "regression":
            continue
        import numpy as np
        de = np.array(a.get("de_test", []), dtype=float).tolist()
        sigma = np.array(a.get("sigma_test", []), dtype=float).tolist()
        if len(de) == 0 or len(sigma) == 0:
            continue
        corr = _rank_corr(de, sigma)
        corr_rows.append({**_ctx(cfg), "spearman_de_sigma": corr})
    with open(out / "correlations_de_sigma.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "spearman_de_sigma",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if corr_rows:
            w.writerows(corr_rows)

    # 5) Risk–coverage (classification): abstain by epistemic vs aleatoric indicators
    rc_rows: List[Dict[str, Any]] = []
    cover_grid = [i / 10.0 for i in range(1, 11)]  # 0.1..1.0
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        import numpy as np
        probs = np.array(a.get("probs", []))
        y = np.array(a.get("y_test", []))
        de = np.array(a.get("de_test", [])) if "de_test" in a else None
        invd = np.array(a.get("inv_density_test", [])) if "inv_density_test" in a else None
        eta = np.array(a.get("eta_test", [])) if "eta_test" in a and a.get("eta_test") is not None else None
        if probs.size == 0 or y.size == 0:
            continue
        # Binary: extract p1 if needed
        if probs.ndim == 2 and probs.shape[1] > 1:
            p1 = probs[:, 1]
        else:
            p1 = probs.reshape(-1)
        y_pred = (p1 >= 0.5).astype(int)
        err = (y_pred != y).astype(float)

        def rc_from_signal(signal: np.ndarray, label: str):
            order = np.argsort(signal)  # keep lowest signal (lowest uncertainty)
            n = len(order)
            for c in cover_grid:
                k = max(1, int(round(c * n)))
                keep = order[:k]
                risk = float(err[keep].mean())
                rc_rows.append({**_ctx(cfg), "coverage": c, "signal": label, "risk": risk})

        if invd is not None and invd.size == len(y):
            rc_from_signal(invd, "epistemic_density")
        if de is not None and de.size == len(y):
            rc_from_signal(de, "epistemic_de")
        if eta is not None and eta.size == len(y):
            rc_from_signal(eta, "aleatoric_eta")

    with open(out / "risk_coverage_classification.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "coverage","signal","risk",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if rc_rows:
            w.writerows(rc_rows)

    # 6) Regression: conditional coverage by sigma and density (joint bins)
    reg_joint_rows: List[Dict[str, Any]] = []
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "regression":
            continue
        import numpy as np
        sigma = np.array(a.get("sigma_test", []), dtype=float)
        invd = np.array(a.get("inv_density_test", []), dtype=float)
        y = np.array(a.get("y_test", []), dtype=float)
        lo = np.array(a.get("pi_lower", []), dtype=float)
        hi = np.array(a.get("pi_upper", []), dtype=float)
        if sigma.size == 0 or invd.size == 0:
            continue
        s_cuts = np.quantile(sigma, [0.33, 0.66])
        d_cuts = np.quantile(invd, [0.33, 0.66])
        sb = np.digitize(sigma, s_cuts)
        db = np.digitize(invd, d_cuts)
        for si in [0, 1, 2]:
            for di in [0, 1, 2]:
                mask = (sb == si) & (db == di)
                if not mask.any():
                    continue
                covered = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
                width = (hi[mask] - lo[mask])
                reg_joint_rows.append({**_ctx(cfg), "sigma_bin": si, "density_bin": di, "coverage": float(covered.mean()), "avg_width": float(width.mean())})
    with open(out / "coverage_by_sigma_density.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "sigma_bin","density_bin","coverage","avg_width",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if reg_joint_rows:
            w.writerows(reg_joint_rows)

    # 7) Selective Explanation Utility (DE-gated) for classification and thresholded regression
    util_rows: List[Dict[str, Any]] = []
    util_auc_rows: List[Dict[str, Any]] = []
    cov_grid = [i / 20.0 for i in range(1, 21)]  # 0.05..1.0
    default_cost = 0.5

    # Classification
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        import numpy as np
        probs = np.array(a.get("probs", []))
        y = np.array(a.get("y_test", []))
        if probs.size == 0 or y.size == 0:
            continue
        # Binary p1
        p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.reshape(-1)
        y_pred = (p1 >= 0.5).astype(int)
        correct = (y_pred == y).astype(float)
        n = len(y)
        # Signals
        de = a.get("de_test")
        eta = a.get("eta_test")
        invd = a.get("inv_density_test")
        signals: List[Tuple[str, Any, bool]] = []
        if de is not None:
            signals.append(("epistemic_de", np.array(de), True))  # lower is better
        if invd is not None:
            signals.append(("epistemic_density", np.array(invd), True))
        if eta is not None:
            signals.append(("aleatoric_eta", np.array(eta), True))
        # Random
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        signals.append(("random", rng.random(n), True))

        for label, sig, asc in signals:
            order = np.argsort(sig) if asc else np.argsort(-sig)
            utils_curve: List[float] = []
            for c in cov_grid:
                k = max(1, int(round(c * n)))
                shown = np.zeros(n, dtype=bool)
                shown[order[:k]] = True
                u = np.zeros(n, dtype=float)
                u[shown & (correct == 1.0)] = 1.0
                u[shown & (correct == 0.0)] = -default_cost
                U = float(np.mean(u))
                util_rows.append({**_ctx(cfg), "run_dir": rdir.name, "task": "classification", "signal": label, "coverage": c, "utility": U})
                utils_curve.append(U)
            # AUC via trapezoidal rule
            auc = 0.0
            prev_c, prev_u = 0.0, utils_curve[0]
            for i, c in enumerate(cov_grid):
                u = utils_curve[i]
                auc += 0.5 * (u + prev_u) * (c - prev_c)
                prev_c, prev_u = c, u
            util_auc_rows.append({**_ctx(cfg), "run_dir": rdir.name, "task": "classification", "signal": label, "auc": auc})

    # Thresholded regression (compute per t and also average)
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
            a = _load_json(rdir / "artifacts.json")
        except Exception:
            continue
        if cfg.get("task") not in ("thresh_reg", "thresholded", "thresholded_regression"):
            continue
        import numpy as np
        y = np.array(a.get("y_test", []))
        de = a.get("de_test")
        invd = a.get("inv_density_test")
        sigma = a.get("sigma_test")
        p_exceed = a.get("p_exceed", {})
        t_values = a.get("t_values", [])
        if y.size == 0 or not p_exceed:
            continue
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        for t in t_values:
            key = f"t={float(t):.4g}"
            p = np.array(p_exceed.get(key) or p_exceed.get(str(key)) or p_exceed.get(t) or p_exceed.get(str(t)))
            if p is None or len(p) != len(y):
                continue
            y_bin = (y > float(t)).astype(int)
            y_pred = (p >= 0.5).astype(int)
            correct = (y_pred == y_bin).astype(float)
            n = len(y)
            signals: List[Tuple[str, Any, bool]] = []
            if de is not None:
                signals.append(("epistemic_de", np.array(de), True))
            if invd is not None:
                signals.append(("epistemic_density", np.array(invd), True))
            if sigma is not None:
                signals.append(("aleatoric_sigma", np.array(sigma), True))
            signals.append(("random", rng.random(n), True))
            for label, sig, asc in signals:
                order = np.argsort(sig) if asc else np.argsort(-sig)
                utils_curve: List[float] = []
                for c in cov_grid:
                    k = max(1, int(round(c * n)))
                    shown = np.zeros(n, dtype=bool)
                    shown[order[:k]] = True
                    u = np.zeros(n, dtype=float)
                    u[shown & (correct == 1.0)] = 1.0
                    u[shown & (correct == 0.0)] = -default_cost
                    U = float(np.mean(u))
                    util_rows.append({**_ctx(cfg), "run_dir": rdir.name, "task": "thresh_reg", "threshold": key, "signal": label, "coverage": c, "utility": U})
                    utils_curve.append(U)
                auc = 0.0
                prev_c, prev_u = 0.0, utils_curve[0]
                for i, c in enumerate(cov_grid):
                    u = utils_curve[i]
                    auc += 0.5 * (u + prev_u) * (c - prev_c)
                    prev_c, prev_u = c, u
                util_auc_rows.append({**_ctx(cfg), "run_dir": rdir.name, "task": "thresh_reg", "threshold": key, "signal": label, "auc": auc})

    # Write selective_utility_curves.csv
    with open(out / "selective_utility_curves.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","task","threshold","signal","coverage","utility",
        ]
        # note: duplicate 'task' key above is harmless in header; DictWriter uses last occurrence
        fieldnames = list(dict.fromkeys(fieldnames))
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if util_rows:
            w.writerows(util_rows)

    # Write selective_utility_auc.csv
    with open(out / "selective_utility_auc.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","task","threshold","signal","auc",
        ]
        fieldnames = list(dict.fromkeys(fieldnames))
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if util_auc_rows:
            w.writerows(util_auc_rows)

    # 7) Uncertainty Sensitivity Indices (summary per run)
    sens_rows: List[Dict[str, Any]] = []
    # Build maps for quick lookup
    # ECE by weight per run_dir
    ece_map: Dict[str, Dict[int, float]] = {}
    for r in ce_ece_rows:
        rd = r.get("run_dir")
        e = float(r.get("ece"))
        wb = int(r.get("weight_bin"))
        ece_map.setdefault(rd, {})[wb] = e
    # Faithfulness per run_dir/bin_type/region/bin
    faith_map: Dict[str, Dict[str, Dict[str, Dict[int, float]]]] = {}
    for r in faith_rows:
        rd = r.get("run_dir")
        bt = r.get("bin_type")
        region = r.get("region", "in")
        bi = int(r.get("bin_id"))
        m = float(r.get("mean_delta"))
        faith_map.setdefault(rd, {}).setdefault(bt, {}).setdefault(region, {})[bi] = m
    # Utility AUC by run_dir/signal
    auc_map: Dict[str, Dict[str, float]] = {}
    for r in util_auc_rows:
        if r.get("task") != "classification":
            continue
        rd = r.get("run_dir")
        sig = r.get("signal")
        a = float(r.get("auc"))
        auc_map.setdefault(rd, {})[sig] = a

    # Stability delta per run_dir via group
    def _group_key(cfg: Dict[str, Any]) -> str:
        g = dict(cfg)
        g.pop("seed", None)
        return json.dumps(g, sort_keys=True, default=str)

    # Iterate runs and emit summary row
    for rdir in runs:
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        rd_name = rdir.name
        # Context
        row = {**_ctx(cfg), "run_dir": rd_name}
        # Δ ECE (weight) = high - low
        em = ece_map.get(rd_name, {})
        e_low = em.get(0, float("nan"))
        e_high = em.get(2, float("nan"))
        row["delta_ece_weight_high_low"] = (e_high - e_low) if (not math.isnan(e_low) and not math.isnan(e_high)) else float("nan")
        # Faithfulness slopes (density/eta, in/out)
        fm = faith_map.get(rd_name, {})
        for bt in ("density", "eta"):
            for region in ("in", "out"):
                bins = fm.get(bt, {}).get(region, {})
                lo = bins.get(0, float("nan"))
                hi = bins.get(2, float("nan"))
                row[f"faith_delta_{bt}_{region}"] = (hi - lo) if (not math.isnan(lo) and not math.isnan(hi)) else float("nan")
        # Utility AUC advantages (classification): DE vs eta/random
        am = auc_map.get(rd_name, {})
        de = am.get("epistemic_de", float("nan"))
        eta_auc = am.get("aleatoric_eta", float("nan"))
        rnd = am.get("random", float("nan"))
        row["auc_adv_de_vs_eta"] = (de - eta_auc) if (not math.isnan(de) and not math.isnan(eta_auc)) else float("nan")
        row["auc_adv_de_vs_random"] = (de - rnd) if (not math.isnan(de) and not math.isnan(rnd)) else float("nan")
        # Stability deltas via group ID
        gk = _group_key(cfg)
        gid = hash(gk)
        deltas = stability_delta_by_group.get(gid, {})
        row["stability_delta_density"] = deltas.get("stability_delta_density", float("nan"))
        row["stability_delta_eta"] = deltas.get("stability_delta_eta", float("nan"))

        sens_rows.append(row)

    with open(out / "uncertainty_sensitivity_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","ablation",
            "run_dir","delta_ece_weight_high_low","faith_delta_density_in","faith_delta_density_out","faith_delta_eta_in","faith_delta_eta_out",
            "auc_adv_de_vs_eta","auc_adv_de_vs_random","stability_delta_density","stability_delta_eta",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if sens_rows:
            w.writerows(sens_rows)

    # 8) Rule-failure detection (classification): AUROC/AP for detecting low-precision rules via DE/density/eta
    from math import isnan
    def _auroc(scores: List[float], labels: List[int]) -> float:
        import numpy as np
        if len(scores) == 0 or len(scores) != len(labels):
            return float('nan')
        # sort by descending score
        order = np.argsort(-np.array(scores, dtype=float))
        y = np.array(labels, dtype=int)[order]
        # positives are failures (1), negatives are non-failures (0)
        P = np.sum(y == 1)
        N = np.sum(y == 0)
        if P == 0 or N == 0:
            return float('nan')
        tp = 0.0
        fp = 0.0
        prev_s = None
        auc = 0.0
        tpr_prev, fpr_prev = 0.0, 0.0
        for i, idx in enumerate(order):
            yi = y[i]
            if yi == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / P
            fpr = fp / N
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
            tpr_prev, fpr_prev = tpr, fpr
        return float(auc)
    def _avg_prec(scores: List[float], labels: List[int]) -> float:
        import numpy as np
        if len(scores) == 0 or len(scores) != len(labels):
            return float('nan')
        order = np.argsort(-np.array(scores, dtype=float))
        y = np.array(labels, dtype=int)[order]
        P = np.sum(y == 1)
        if P == 0:
            return float('nan')
        tp = 0.0
        precs = []
        recs = []
        for i in range(len(y)):
            if y[i] == 1:
                tp += 1
            prec = tp / (i + 1)
            rec = tp / P
            precs.append(prec)
            recs.append(rec)
        # AP as Riemann sum over recall steps (monotone by construction)
        ap = 0.0
        prev_r, prev_p = 0.0, 1.0
        for p, r in zip(precs, recs):
            ap += p * max(0.0, r - prev_r)
            prev_r, prev_p = r, p
        return float(ap)

    rf_rows_auc: List[Dict[str, Any]] = []
    rf_rows_ap: List[Dict[str, Any]] = []
    TAU = float(rule_failure_tau)
    for rdir in runs:
        rules_path = rdir / "rules.jsonl"
        if not rules_path.exists():
            continue
        try:
            cfg = _load_json(rdir / "config.json")
        except Exception:
            continue
        if cfg.get("task") != "classification":
            continue
        # Load factual rules
        by_rule: Dict[str, Dict[str, Any]] = {}
        by_pt_top: Dict[int, Dict[str, Any]] = {}
        with open(rules_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("explanation_type") != "factual":
                    continue
                # group by rule_id for empirical precision among supporting instances
                try:
                    wv = float(rec.get("w", 0.0))
                except Exception:
                    wv = 0.0
                sgn = 1.0 if wv > 0 else (-1.0 if wv < 0 else 0.0)
                y_pred = rec.get("y_pred")
                y_true = rec.get("y_true")
                if y_pred is None or y_true is None:
                    continue
                supports = (y_pred == 1 and sgn > 0) or (y_pred == 0 and sgn < 0)
                rid = rec.get("rule_id")
                if supports:
                    by_rule.setdefault(rid, {"correct": 0, "total": 0})
                    by_rule[rid]["total"] += 1
                    by_rule[rid]["correct"] += 1 if int(y_true) == int(y_pred) else 0
                # capture top-1 per point (by |w|)
                pid = int(rec.get("point_id", -1))
                if pid >= 0:
                    cur = by_pt_top.get(pid)
                    wabs = abs(wv)
                    if cur is None or wabs > cur.get("wabs", -1):
                        by_pt_top[pid] = {
                            "rule_id": rid,
                            "wabs": wabs,
                            "de": rec.get("de"),
                            "inv_density": rec.get("inv_density"),
                            "eta": rec.get("eta"),
                        }
        if not by_pt_top:
            continue
        # Compute empirical precision per rule
        rule_prec: Dict[str, float] = {}
        for rid, d in by_rule.items():
            if d["total"] > 0:
                rule_prec[rid] = float(d["correct"] / d["total"])
        # Build labels (failure if rule precision < TAU) and scores
        labels: List[int] = []
        de_scores: List[float] = []
        dens_scores: List[float] = []
        eta_scores: List[float] = []
        import numpy as np
        rng = np.random.default_rng(0)
        rnd_scores: List[float] = []
        for pid, info in by_pt_top.items():
            rid = info.get("rule_id")
            prec = rule_prec.get(rid, float('nan'))
            if not (prec == prec):
                continue
            labels.append(1 if prec < TAU else 0)
            de_scores.append(float(info.get("de")) if info.get("de") is not None else float('nan'))
            dens_scores.append(float(info.get("inv_density")) if info.get("inv_density") is not None else float('nan'))
            eta_scores.append(float(info.get("eta")) if info.get("eta") is not None else float('nan'))
            rnd_scores.append(float(rng.random()))
        # Filter out any NaNs consistently per signal
        def _clean(scores: List[float], labels: List[int]) -> (List[float], List[int]):
            xs, ys = [], []
            for s, y in zip(scores, labels):
                if s == s:
                    xs.append(float(s))
                    ys.append(int(y))
            return xs, ys
        for label, scores in ("epistemic_de", de_scores), ("epistemic_density", dens_scores), ("aleatoric_eta", eta_scores), ("random", rnd_scores):
            xs, ys = _clean(scores, labels)
            auc = _auroc(xs, ys)
            ap = _avg_prec(xs, ys)
            rf_rows_auc.append({**_ctx(cfg), "run_dir": rdir.name, "signal": label, "auroc": auc, "tau": TAU})
            rf_rows_ap.append({**_ctx(cfg), "run_dir": rdir.name, "signal": label, "ap": ap, "tau": TAU})

    with open(out / "rule_failure_auroc.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_dir","signal","auroc","tau",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if rf_rows_auc:
            w.writerows(rf_rows_auc)

    with open(out / "rule_failure_ap.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","seed","dims","n_train","n_cal","n_test","shift","holes_label",
            "model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "run_dir","signal","ap","tau",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if rf_rows_ap:
            w.writerows(rf_rows_ap)

    # 7b) Slopes vs shift (δ) and holes size (s) for key metrics
    def _num_hole(label: Any) -> float:
        m = {"none": 0.0, "small": 0.5, "large": 1.0}
        try:
            return m[str(label)]
        except Exception:
            return float('nan')
    def _slope(xs: List[float], ys: List[float]) -> float:
        import numpy as np
        x = np.array([v for v in xs if v == v], dtype=float)
        y = np.array([ys[i] for i, v in enumerate(xs) if v == v and ys[i] == ys[i]], dtype=float)
        if len(x) < 2 or len(y) < 2:
            return float('nan')
        try:
            m, b = np.polyfit(x, y, 1)
            return float(m)
        except Exception:
            return float('nan')

    slope_rows: List[Dict[str, Any]] = []
    # Rule-ECE slope (classification factual)
    # Group by config excluding seed and shift/holes and explanation_type
    def _key_without(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        outk = dict(d)
        for k in keys:
            outk.pop(k, None)
        return outk
    # Prepare rows grouped by (context minus seed/shift/holes, explanation_type)
    by_group: Dict[str, Dict[str, Any]] = {}
    vals_by_group_shift: Dict[str, Dict[float, List[float]]] = {}
    vals_by_group_holes: Dict[str, Dict[float, List[float]]] = {}
    # Use baseline rule ECE rows (classification factual) for slope computation
    for r in bl_ece_rows:
        c = _key_without({k: r.get(k) for k in ("experiment","task","dims","n_train","n_cal","n_test","shift","holes_label","model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","explanation_type")}, ["shift","holes_label"])  # context sans shift/holes
        et = r.get("explanation_type")
        gkey = json.dumps({**c, "explanation_type": et}, sort_keys=True)
        by_group[gkey] = {**c, "explanation_type": et}
        sh = float(_load_json((Path(root) / r.get("run_dir") / "config.json")).get("data", {}).get("shift", float('nan'))) if isinstance(root, Path) else float('nan')
        # shift and holes from the row itself
        sh = float(r.get("shift")) if r.get("shift") is not None else float('nan')
        hl = _num_hole(r.get("holes_label"))
        e = float(r.get("ece"))
        vals_by_group_shift.setdefault(gkey, {}).setdefault(sh, []).append(e)
        vals_by_group_holes.setdefault(gkey, {}).setdefault(hl, []).append(e)
    for gkey, ctx in by_group.items():
        # factual only preferred, but include any
        sh_map = vals_by_group_shift.get(gkey, {})
        hl_map = vals_by_group_holes.get(gkey, {})
        sh_x = sorted([k for k in sh_map.keys() if k == k])
        sh_y = [float(sum(sh_map[s])/max(1,len(sh_map[s]))) for s in sh_x]
        hl_x = sorted([k for k in hl_map.keys() if k == k])
        hl_y = [float(sum(hl_map[s])/max(1,len(hl_map[s]))) for s in hl_x]
        slope_rows.append({**ctx, "metric": "rule_ece", "slope_shift": _slope(sh_x, sh_y), "slope_holes": _slope(hl_x, hl_y)})

    # Stability exact-match slope (average across bins per group) vs shift/holes
    # Build a map of (context minus seed/shift/holes, bin_type) -> {shift: avg_rate}
    st_by_group: Dict[str, Dict[str, Any]] = {}
    st_shift: Dict[str, Dict[float, List[float]]] = {}
    st_holes: Dict[str, Dict[float, List[float]]] = {}
    for r in stability_rows:
        c = _key_without({k: r.get(k) for k in ("experiment","task","dims","n_train","n_cal","n_test","shift","holes_label","model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights","bin_type")}, ["shift","holes_label"])  # sans shift/holes
        bt = r.get("bin_type")
        gkey = json.dumps({**c, "bin_type": bt}, sort_keys=True)
        st_by_group[gkey] = {**c, "bin_type": bt}
        sh = float(r.get("shift")) if r.get("shift") is not None else float('nan')
        hl = _num_hole(r.get("holes_label"))
        st_shift.setdefault(gkey, {}).setdefault(sh, []).append(float(r.get("exact_match_rate")))
        st_holes.setdefault(gkey, {}).setdefault(hl, []).append(float(r.get("exact_match_rate")))
    for gkey, ctx in st_by_group.items():
        sh_x = sorted([k for k in st_shift.get(gkey, {}).keys() if k == k])
        sh_y = [float(sum(st_shift[gkey][s]) / max(1, len(st_shift[gkey][s]))) for s in sh_x]
        hl_x = sorted([k for k in st_holes.get(gkey, {}).keys() if k == k])
        hl_y = [float(sum(st_holes[gkey][s]) / max(1, len(st_holes[gkey][s]))) for s in hl_x]
        slope_rows.append({**ctx, "metric": "stability_exact", "slope_shift": _slope(sh_x, sh_y), "slope_holes": _slope(hl_x, hl_y)})

    # Utility AUC (classification, DE signal) slope vs shift/holes
    util_group: Dict[str, Dict[str, Any]] = {}
    util_shift: Dict[str, Dict[float, List[float]]] = {}
    util_holes: Dict[str, Dict[float, List[float]]] = {}
    for r in util_auc_rows:
        if r.get("task") != "classification" or r.get("signal") != "epistemic_de":
            continue
        c = _key_without({k: r.get(k) for k in ("experiment","task","dims","n_train","n_cal","n_test","shift","holes_label","model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights")}, ["shift","holes_label"])  # sans shift/holes
        gkey = json.dumps(c, sort_keys=True)
        util_group[gkey] = c
        sh = float(r.get("shift")) if r.get("shift") is not None else float('nan')
        hl = _num_hole(r.get("holes_label"))
        a = float(r.get("auc"))
        util_shift.setdefault(gkey, {}).setdefault(sh, []).append(a)
        util_holes.setdefault(gkey, {}).setdefault(hl, []).append(a)
    for gkey, ctx in util_group.items():
        sh_x = sorted([k for k in util_shift.get(gkey, {}).keys() if k == k])
        sh_y = [float(sum(util_shift[gkey][s]) / max(1, len(util_shift[gkey][s]))) for s in sh_x]
        hl_x = sorted([k for k in util_holes.get(gkey, {}).keys() if k == k])
        hl_y = [float(sum(util_holes[gkey][s]) / max(1, len(util_holes[gkey][s]))) for s in hl_x]
        slope_rows.append({**ctx, "metric": "util_auc_de", "slope_shift": _slope(sh_x, sh_y), "slope_holes": _slope(hl_x, hl_y)})

    with open(out / "uncertainty_sensitivity_slopes.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment","task","dims","n_train","n_cal","n_test","model","model_params","calib_classification","calib_regression","alpha","calib_thresholded","t_values","knn_k","de_weights",
            "ablation",
            "explanation_type","bin_type","metric","slope_shift","slope_holes",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if slope_rows:
            w.writerows(slope_rows)

    # Write an index.json with summary counts
    summary = {
        "runs_found": len(runs),
        "rows": {
            "coverage_by_sigma": len(reg_rows),
            "ce_reliability_by_weight": len(ce_rel_rows),
            "ce_ece_by_weight": len(ce_ece_rows),
            "ce_weight_uncertainty_by_density": len(ce_w_unc_rows),
            "ce_rule_direction_consistency": len(ce_dir_rows),
            "effect_interval_coverage": len(eff_cov_rows),
            "effect_magnitude_calibration": len(eff_mag_rows),
            "effect_sign_consistency": len(eff_sign_rows),
            "effect_rank_correlation": len(eff_rank_rows),
            "effect_interval_coverage_strict": len(strict_cov_rows),
            "rule_stability": len(stability_rows),
            "rule_faithfulness": len(faith_rows),
            "uncertainty_sensitivity_summary": len(sens_rows),
            "ece_by_ncal": len(ece_agg_rows),
            "thresh_reg_reliability": len(th_rows),
            "thresh_reg_rule_reliability_by_bin": len(th_rule_rows),
            "ece_by_density": len(ece_by_dens_rows),
            "coverage_by_density": len(reg_dens_rows),
            "correlations_de_sigma": len(corr_rows),
            "risk_coverage_classification": len(rc_rows),
            "coverage_by_sigma_density": len(reg_joint_rows),
            "selective_utility_curves": len(util_rows),
            "selective_utility_auc": len(util_auc_rows),
        },
    }
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Aggregate experiment runs into paper-ready CSVs")
    ap.add_argument("--root", required=True, help="Artifacts root (where run_* folders are)")
    ap.add_argument("--out", help="Output directory for derived CSVs")
    ap.add_argument("--config", help="Grid spec used to derive experiment metadata (YAML/JSON)")
    ap.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        help="Experiment identifier to include. Repeat or pass comma-separated values for multiple entries.",
    )
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for expensive per-run steps (faithfulness)")
    ap.add_argument("--jaccard-k", type=int, default=3, help="Top-k size for Jaccard stability (default: 3)")
    ap.add_argument("--rule-failure-tau", type=float, default=0.7, help="Threshold τ for rule failure (precision<τ) (default: 0.7)")
    args = ap.parse_args()

    root = Path(args.root)
    experiments: Set[str] = set()
    if args.experiments:
        for item in args.experiments:
            if not item:
                continue
            experiments.update({part.strip() for part in item.split(",") if part.strip()})

    spec: Optional[Dict[str, Any]] = None
    spec_path: Optional[Path] = None
    if args.config:
        spec_path = Path(args.config)
        spec = _read_spec(spec_path)
        experiments.update(_experiments_from_spec(spec))

    if args.out is None:
        if args.config is None:
            ap.error("--out is required unless --config is provided")
        derived_root = Path("evaluation/uncertainty_experiments/derived")
        if experiments:
            if len(experiments) == 1:
                suffix = next(iter(experiments))
            else:
                suffix = "multi_experiment"
        else:
            suffix = (spec_path.stem if spec_path is not None else "grid")
        out = derived_root / suffix
    else:
        out = Path(args.out)

    n_jobs = int(args.n_jobs or 1)
    if n_jobs == 1:
        if spec:
            try:
                cfg_nj = int(spec.get("n_jobs", 1) or 1)
                n_jobs = max(1, cfg_nj)
            except Exception:
                pass
        if n_jobs == 1:
            # Try to default to grid spec n_jobs from index_*.json
            try:
                idx_files = list(root.glob("index_*.json"))
                if idx_files:
                    idx = _load_json(idx_files[0])
                    spec_idx = idx.get("spec", {}) if isinstance(idx, dict) else {}
                    cfg_nj = int(spec_idx.get("n_jobs", 1) or 1)
                    n_jobs = max(1, cfg_nj)
            except Exception:
                n_jobs = 1

    experiments_filter = experiments or None
    if experiments_filter:
        print(f"Restricting aggregation to experiments: {sorted(experiments_filter)}")

    aggregate(
        root,
        out,
        n_jobs=n_jobs,
        jaccard_k=int(args.jaccard_k or 3),
        rule_failure_tau=float(args.rule_failure_tau or 0.7),
        experiments=experiments_filter,
    )
    print(f"Derived CSVs written to {out}")

if __name__ == "__main__":
    main()
