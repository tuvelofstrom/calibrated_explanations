from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib

# Ensure repo root/src on path when invoked directly
import sys
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
_REPO_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_REPO_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from evaluation.uncertainty_experiments.runner import (
    run_regression,
    run_classification,
    run_thresholded_regression,
    ensure_dir,
    _to_py,
)
from joblib import Parallel, delayed


def _read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        if path.endswith(".json"):
            return json.loads(text)
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Use a .json config or install pyyaml.")
        return yaml.safe_load(text)


def _holes_preset(name: str) -> List[List[float]] | None:
    if name == "none":
        return None
    if name == "small":
        return [[-0.5, -0.5, -0.1, 0.1]]
    if name == "large":
        return [[-0.8, -0.8, 0.0, 0.2], [0.2, -0.2, 0.8, 0.2]]
    raise ValueError(f"Unknown holes preset: {name}")


def _expand(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = spec.get("tasks", [spec.get("task", "regression")])
    seeds = spec.get("seeds", [spec.get("seed", 42)])
    output_dir = spec.get("output_dir", "evaluation/uncertainty_experiments/artifacts")

    data_spec = spec.get("data", {})
    dims_list = data_spec.get("dims", [2])
    n_train_list = data_spec.get("n_train", [1000])
    n_cal_list = data_spec.get("n_cal", [200])
    n_test_list = data_spec.get("n_test", [1000])
    shift_list = data_spec.get("shift", [0.0])
    holes_list = data_spec.get("holes", ["none"])  # preset names
    hetero = data_spec.get("hetero")

    models = spec.get("models", [{"type": "rf", "params": {"n_estimators": 100}}])
    calib = spec.get("calibration", {})
    unc = spec.get("uncertainty", {})
    logging = spec.get("logging", {"save_predictions": True, "save_metrics": True})

    runs: List[Dict[str, Any]] = []
    ablations = spec.get("ablations", ["none"])
    baselines = spec.get("baselines", [])
    for task, seed, dims, n_train, n_cal, n_test, shift, holes_name in itertools.product(
        tasks, seeds, dims_list, n_train_list, n_cal_list, n_test_list, shift_list, holes_list
    ):
        for model in models:
            cfg: Dict[str, Any] = {
                "experiment": spec.get("experiment", "grid"),
                "task": task,
                "seed": seed,
                "output_dir": output_dir,
                "data": {
                    "n_train": n_train,
                    "n_cal": n_cal,
                    "n_test": n_test,
                    "dims": dims,
                    "holes": _holes_preset(holes_name),
                    "holes_label": holes_name,
                    "shift": shift,
                },
                "model": model,
                "uncertainty": unc,
                "ablations": ablations,
                "baselines": baselines,
                "logging": logging,
            }
            if hetero is not None:
                cfg["data"]["hetero"] = hetero

            # Calibration per task
            # Expand over calibration settings if lists are provided
            if task == "classification":
                cls_methods = calib.get("classification", ["venn_abers"])
                cls_methods = cls_methods if isinstance(cls_methods, list) else [cls_methods]
                for cm in cls_methods:
                    for abl in (ablations if ablations else ["none"]):
                        cfg2 = json.loads(json.dumps(cfg))
                        cfg2["calibration"] = {"classification": {"method": cm}}
                        cfg2["ablation"] = abl
                        runs.append(cfg2)
            elif task == "regression":
                reg_methods = calib.get("regression", ["split_cp"])
                reg_methods = reg_methods if isinstance(reg_methods, list) else [reg_methods]
                for rm in reg_methods:
                    for abl in (ablations if ablations else ["none"]):
                        cfg2 = json.loads(json.dumps(cfg))
                        cfg2["calibration"] = {"regression": {"method": rm, "alpha": calib.get("alpha", 0.1), "n_folds": calib.get("n_folds", 5)}}
                        cfg2["ablation"] = abl
                        runs.append(cfg2)
            else:  # thresholded
                reg_methods = calib.get("regression", ["split_cp"])
                reg_methods = reg_methods if isinstance(reg_methods, list) else [reg_methods]
                for rm in reg_methods:
                    for abl in (ablations if ablations else ["none"]):
                        cfg2 = json.loads(json.dumps(cfg))
                        cfg2["calibration"] = {
                            "regression": {"method": rm, "alpha": calib.get("alpha", 0.1), "n_folds": calib.get("n_folds", 5)},
                            "thresholded": {"method": calib.get("thresholded", {}).get("method", "interval_map"), "t_values": calib.get("thresholded", {}).get("t_values", "auto")},
                        }
                        cfg2["ablation"] = abl
                        runs.append(cfg2)
    return runs


def _run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    task = cfg["task"].lower()
    if task == "regression":
        return run_regression(cfg)
    if task == "classification":
        return run_classification(cfg)
    if task in {"thresh_reg", "thresholded", "thresholded_regression"}:
        return run_thresholded_regression(cfg)
    raise NotImplementedError(f"Unknown task '{task}'")


def _slug(cfg: Dict[str, Any]) -> str:
    base = (
        f"{cfg['task']}_seed{cfg['seed']}_dims{cfg['data']['dims']}"
        f"_ncal{cfg['data']['n_cal']}_model{cfg['model']['type']}"
    )
    # Include a short hash of the full config to avoid collisions across shifts/holes/params
    h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def _run_and_persist(cfg: Dict[str, Any], out_dir: Path) -> Tuple[str, bool, str]:
    """Run a single config and persist outputs. Returns (slug, ok, msg)."""
    slug = _slug(cfg)
    run_dir = out_dir / f"run_{slug}"
    ensure_dir(run_dir)
    try:
        res = _run_one(cfg)
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_to_py(res.get("metrics", {})), f, indent=2)
        with open(run_dir / "artifacts.json", "w", encoding="utf-8") as f:
            json.dump(_to_py(res.get("artifacts", {})), f)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(_to_py(cfg), f, indent=2)
        # Optional: persist rules.jsonl if requested and available
        if cfg.get("logging", {}).get("dump_rules", False) and "rules_records" in res:
            rules_path = run_dir / "rules.jsonl"
            run_id = f"run_{_slug(cfg)}"
            with open(rules_path, "w", encoding="utf-8") as f:
                for rec in res["rules_records"]:
                    # Attach run_id for provenance; write as compact JSON lines
                    full = {"run_id": run_id, **rec}
                    f.write(json.dumps(_to_py(full)) + "\n")
        # Optional: persist baselines
        if cfg.get("logging", {}).get("dump_baselines", False) and "baseline_records" in res:
            run_id = f"run_{_slug(cfg)}"
            for bname, recs in res["baseline_records"].items():
                outp = run_dir / f"baseline_rules_{bname}.jsonl"
                with open(outp, "w", encoding="utf-8") as f:
                    for rec in recs:
                        full = {"run_id": run_id, "baseline": bname, **rec}
                        f.write(json.dumps(_to_py(full)) + "\n")
        with open(run_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump({"ok": True}, f)
        return (slug, True, "ok")
    except Exception as e:  # pragma: no cover
        with open(run_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": str(e)}, f)
        return (slug, False, str(e))


def main():
    ap = argparse.ArgumentParser(description="Grid executor for uncertainty experiments")
    ap.add_argument("--config", required=True, help="Path to experiment spec YAML/JSON")
    ap.add_argument("--n-jobs", type=int, default=None, help="Reserved; runs sequentially if None")
    ap.add_argument("--resume", action="store_true", help="Skip runs with existing metrics.json")
    args = ap.parse_args()

    spec = _read_config(args.config)
    runs = _expand(spec)

    out_dir = Path(spec.get("output_dir", "evaluation/uncertainty_experiments/artifacts"))
    ensure_dir(out_dir)

    # Persist run index
    run_index_path = out_dir / f"index_{spec.get('experiment','grid')}.json"
    with open(run_index_path, "w", encoding="utf-8") as f:
        json.dump(_to_py({"spec": spec, "n": len(runs)}), f, indent=2)

    # Prepare run list with resume filtering
    to_run: List[Dict[str, Any]] = []
    skipped = 0
    for cfg in runs:
        slug = _slug(cfg)
        run_dir = out_dir / f"run_{slug}"
        if args.resume and (run_dir / "metrics.json").exists():
            print(f"[skip] {slug} (metrics.json exists)")
            skipped += 1
            continue
        to_run.append(cfg)

    n_jobs = args.n_jobs or spec.get("n_jobs", 1) or 1
    print(f"Planned runs: {len(to_run)} (skipped {skipped}); n_jobs={n_jobs}")

    if n_jobs == 1 or len(to_run) <= 1:
        completed = 0
        for cfg in to_run:
            slug, ok, msg = _run_and_persist(cfg, out_dir)
            completed += 1
            print(f"[{completed}/{len(to_run)}] {slug} -> {'ok' if ok else 'FAIL'} {msg if not ok else ''}")
    else:
        # Parallel execution
        results = Parallel(n_jobs=int(n_jobs), backend="loky", prefer="processes")(
            delayed(_run_and_persist)(cfg, out_dir) for cfg in to_run
        )
        # Log summary
        ok_cnt = sum(1 for _, ok, _ in results if ok)
        fail_cnt = len(results) - ok_cnt
        print(f"Completed: {ok_cnt} OK, {fail_cnt} FAIL")


if __name__ == "__main__":
    main()
