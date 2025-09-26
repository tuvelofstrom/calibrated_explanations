# Uncertainty Experiments — Results Guide

This guide maps each research question to the derived CSV files and figures generated under `evaluation/uncertainty_experiments`. Paths are workspace‑relative and clickable.

## Where Artifacts Live
- Derived CSVs (for analysis): `evaluation/uncertainty_experiments/derived`
- Figures (ready‑to‑view PNGs): `evaluation/uncertainty_experiments/figures`
- Per‑run artifacts: `evaluation/uncertainty_experiments/artifacts/run_*`

Most CSV rows include context columns so you can slice/group results:
- Common context: `experiment, task, seed, dims, n_train, n_cal, n_test, shift, holes_label, model, model_params, calib_*`
- Uncertainty setup: `knn_k, de_weights`
- New: `ablation` (e.g., `none`, `ce_uncalibrated`, `ce_no_de`, `gate_density_only`, `gate_disagreement_only`)
- Optional: `baseline` name for baseline outputs

## Core Questions → Files/Figures

### 1) Are CE rule confidences calibrated? (classification)
- Per‑rule reliability (binned):
  - CSV: `evaluation/uncertainty_experiments/derived/rule_reliability_by_bin.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/rule_level_reliability.png`
  - What: Reliability diagram (predicted rule probability vs empirical accuracy), per explanation type (factual and, if available, alternative).
- Aggregate rule calibration (ECE):
  - CSV: `evaluation/uncertainty_experiments/derived/rule_ece.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/rule_ece_bars.png`
  - What: Lower is better; compares explanation types and ablations.

### 2) How do aleatoric vs epistemic uncertainty inflate feature‑effect uncertainty?
- Weight interval width vs epistemic density:
  - CSV: `evaluation/uncertainty_experiments/derived/ce_weight_uncertainty_by_density.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/ce_weight_uncertainty_by_density.png`
  - What: Mean top‑1 rule weight interval width by inverse‑density tertiles.
- Weight interval width vs aleatoric η:
  - CSV: `evaluation/uncertainty_experiments/derived/ce_weight_uncertainty_by_eta.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/ce_weight_uncertainty_eta.png`
  - What: Mean top‑1 rule weight interval width by η tertiles.

### 3) Are explanations stable across seeds, and how does stability degrade with uncertainty?
- Exact top‑1 match across seeds:
  - CSV: `evaluation/uncertainty_experiments/derived/rule_stability.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/rule_stability_density.png`, `evaluation/uncertainty_experiments/figures/rule_stability_eta.png`
  - What: Exact identity match rate vs density/η tertiles.
- Set‑level stability (Jaccard@k across seeds):
  - CSV: `evaluation/uncertainty_experiments/derived/rule_stability_jaccard.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/rule_stability_jaccard_density.png`, `evaluation/uncertainty_experiments/figures/rule_stability_jaccard_eta.png`
  - What: Average pairwise Jaccard overlap of top‑k rule sets; control k via aggregator flag `--jaccard-k`.

### 4) Can DE/density predict rule failure (actionable abstention)?
- Rule failure detection (classification):
  - CSVs: `evaluation/uncertainty_experiments/derived/rule_failure_auroc.csv`, `evaluation/uncertainty_experiments/derived/rule_failure_ap.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/rule_failure_auroc.png`, `evaluation/uncertainty_experiments/figures/rule_failure_ap.png`
  - What: AUROC/AP for detecting low‑precision rules (precision < τ). Control τ via `--rule-failure-tau`.

### 5) Do selective explanation policies improve utility under uncertainty?
- Utility–coverage curves and AUC (classification and thresholded):
  - CSVs: `evaluation/uncertainty_experiments/derived/selective_utility_curves.csv`, `evaluation/uncertainty_experiments/derived/selective_utility_auc.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/selective_utility_classification.png`, `evaluation/uncertainty_experiments/figures/selective_utility_thresh_reg.png`,
             `evaluation/uncertainty_experiments/figures/selective_utility_auc_classification.png`, `evaluation/uncertainty_experiments/figures/selective_utility_auc_thresh_reg.png`,
             `evaluation/uncertainty_experiments/figures/selective_utility_auc_distribution_classification.png`, `evaluation/uncertainty_experiments/figures/selective_utility_auc_distribution_thresh_reg.png`
  - What: Compare gating by `epistemic_de` / `epistemic_density` vs `aleatoric_eta` / `random`.

### 6) Are exceedance probabilities calibrated (thresholded regression)?
- Instance‑level reliability across thresholds t:
  - CSV: `evaluation/uncertainty_experiments/derived/thresh_reg_reliability.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/thresh_reg_ece_by_t.png`
- Rule‑level exceedance reliability across t:
  - CSV: `evaluation/uncertainty_experiments/derived/thresh_reg_rule_reliability_by_bin.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/thresh_reg_rule_ece_by_t.png`

### 7) How do shift (δ) and holes (s) affect calibration, stability, and utility?
- Slope indices (per metric):
  - CSV: `evaluation/uncertainty_experiments/derived/uncertainty_sensitivity_slopes.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/uncertainty_sensitivity_slopes_shift.png`, `evaluation/uncertainty_experiments/figures/uncertainty_sensitivity_slopes_holes.png`
  - What: Linear slopes vs δ and vs hole size for key metrics (rule_ece, stability_exact, util_auc_de).

### 8) Baselines (dependency‑free) — surrogate tree and stump
- Rule‑level reliability/ECE (baseline proxies):
  - CSVs: `evaluation/uncertainty_experiments/derived/baseline_rule_reliability_by_bin.csv`, `evaluation/uncertainty_experiments/derived/baseline_rule_ece.csv`
  - What: Reliability diagrams/ECE using baseline rule probabilities (leaf probabilities) as proxies.

## Prediction‑Level Sanity Checks
- Classification ECE by density: `evaluation/uncertainty_experiments/derived/ece_by_density.csv` → `evaluation/uncertainty_experiments/figures/classification_ece_by_density.png`
- Regression coverage vs σ / density / joint σ×density:
  - `evaluation/uncertainty_experiments/derived/coverage_by_sigma.csv` → `evaluation/uncertainty_experiments/figures/regression_coverage_sigma.png`
  - `evaluation/uncertainty_experiments/derived/coverage_by_density.csv` → `evaluation/uncertainty_experiments/figures/regression_coverage_density.png`
  - `evaluation/uncertainty_experiments/derived/coverage_by_sigma_density.csv` → `evaluation/uncertainty_experiments/figures/regression_coverage_sigma_density_heatmap.png`

## Columns You’ll Use Often
- `ablation`: slice to compare `none` vs `ce_uncalibrated`, `ce_no_de`, `gate_density_only`, `gate_disagreement_only`.
- `baseline`: present in baseline CSVs (`surrogate_tree`, `stump_on_proba`).
- `run_dir`: run identifier you can join across CSVs if needed.

## How To Re‑generate
- Aggregate: `python -m evaluation.uncertainty_experiments.aggregate --root evaluation/uncertainty_experiments/artifacts --out evaluation/uncertainty_experiments/derived --n-jobs 6 --jaccard-k 5 --rule-failure-tau 0.65`
- Figures: `python -m evaluation.uncertainty_experiments.viz.make_figures --derived evaluation/uncertainty_experiments/derived --figdir evaluation/uncertainty_experiments/figures`

## Notes & Caveats
- Alternatives: include in rule‑level reliability only if `rules.jsonl` contains `rule_predict` for `explanation_type='alternative'`.
- Rule failure labeling: τ controls class balance; if AUROC/AP are empty or NaN‑heavy, try a different `--rule-failure-tau` and/or require a minimum rule support.
- Stability: Jaccard@k depends on `--jaccard-k` and seed diversity; use both exact‑match and Jaccard@k for a fuller picture.

