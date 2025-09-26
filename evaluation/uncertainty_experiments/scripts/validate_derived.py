from __future__ import annotations

"""Validate derived CSVs for required columns and non-empty content.

Usage:
    python evaluation/uncertainty_experiments/scripts/validate_derived.py \
        --derived evaluation/uncertainty_experiments/derived
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List


REQUIREMENTS: Dict[str, List[str]] = {
    "ce_reliability_by_weight.csv": [
        "experiment","task","seed","n_cal","model","run_dir",
        "weight_bin","bin","p_mean","acc_mean","count",
    ],
    "ce_ece_by_weight.csv": [
        "experiment","task","seed","n_cal","model","run_dir","weight_bin","ece",
    ],
    "ce_weight_uncertainty_by_density.csv": [
        "experiment","task","seed","n_cal","model","run_dir","density_bin","mean_w_width",
    ],
    "ce_rule_direction_consistency.csv": [
        "experiment","task","seed","n_cal","model","run_dir","bin_type","bin_id","support_rate",
    ],
    "rule_stability.csv": [
        "experiment","task","model","run_group_id","n_runs","bin_type","bin_id","exact_match_rate",
    ],
    "rule_faithfulness.csv": [
        "experiment","task","model","run_dir","bin_type","region","bin_id","mean_delta",
    ],
    "thresh_reg_reliability.csv": [
        "experiment","task","model","threshold","ece",
    ],
    "selective_utility_curves.csv": [
        "experiment","task","model","run_dir","signal","coverage","utility",
    ],
    "selective_utility_auc.csv": [
        "experiment","task","model","run_dir","signal","auc",
    ],
    "uncertainty_sensitivity_summary.csv": [
        "experiment","task","model","run_dir","delta_ece_weight_high_low",
    ],
}


def validate_csv(path: Path, required_cols: List[str]) -> List[str]:
    errs: List[str] = []
    if not path.exists():
        return [f"missing: {path.name}"]
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in required_cols if c not in cols]
        if missing:
            errs.append(f"{path.name}: missing columns: {missing}")
        count = 0
        for row in reader:
            count += 1
        if count == 0:
            errs.append(f"{path.name}: has header but no rows")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate derived CSVs")
    ap.add_argument("--derived", required=True, help="Path to derived directory")
    args = ap.parse_args()
    root = Path(args.derived)
    errs: List[str] = []
    for name, cols in REQUIREMENTS.items():
        errs.extend(validate_csv(root / name, cols))
    if errs:
        print("Validation FAILED:")
        for e in errs:
            print(" -", e)
    else:
        print("Validation OK: all required CSVs present with expected columns and non-empty.")


if __name__ == "__main__":
    main()

