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

### 1) Are CE predictions calibrated, and how does |w| relate? (classification)
- Reliability by top‑1 |w| tertiles:
  - CSV: `evaluation/uncertainty_experiments/derived/ce_reliability_by_weight.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/ce_reliability_by_weight.png`
  - What: Reliability diagram of base prediction p_pred within |w| tertiles. Uncertainty: stratification by explanation strength (|w|); not an uncertainty proxy itself.
  - How: For each run, pick the top-1 rule by |w| per point from `rules.jsonl`, bin points into tertiles by |w|, compute a 10-bin reliability diagram (p vs empirical accuracy) within each tertile, then average across runs.
  - Why: Tests whether stronger explanations (higher |w|) coincide with better-calibrated predictions, informing when CE explanations are most trustworthy.
  - Interpretation: Curves near the diagonal indicate good calibration; systematic deviations reveal over/under-confidence within |w| tertiles (e.g., high |w| should ideally be best calibrated).
- ECE by |w| tertiles:
  - CSV: `evaluation/uncertainty_experiments/derived/ce_ece_by_weight.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/ce_ece_by_weight.png`
  - What: Expected calibration error of p_pred within |w| tertiles. Uncertainty: stratification by explanation strength (|w|); not an uncertainty proxy itself.
  - How: Use the same tertiles by |w|. Within each tertile, compute ECE over 10 probability bins; report the mean ECE across runs.
  - Why: Summarizes calibration into a single number per |w| stratum to compare conditions and ablations.
  - Interpretation: Lower is better. If ECE decreases from low→high |w|, stronger explanations align with better-calibrated predictions; flat or inverted trends flag concerns.

### 2) How do aleatoric vs epistemic uncertainty inflate feature‑effect uncertainty?
- Weight interval width vs epistemic density:
  - CSV: `evaluation/uncertainty_experiments/derived/ce_weight_uncertainty_by_density.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/ce_weight_uncertainty_by_density.png`
  - What: Mean top‑1 rule weight interval width by inverse‑density tertiles. Uncertainty: epistemic via inverse‑density (lower density = higher epistemic uncertainty).
  - How: From `rules.jsonl`, take the top-1 rule per point and its weight interval width; bin points by tertiles of `inv_density` and average widths with 95% CI across runs.
  - Why: Links epistemic uncertainty (data scarcity) to uncertainty in rule effect magnitude.
  - Interpretation: Increasing widths from low→high inverse density suggest epistemic uncertainty inflates effect uncertainty; flat trends imply resilience to sparse regions.
- Weight interval width vs aleatoric η:
  - CSV: `evaluation/uncertainty_experiments/derived/ce_weight_uncertainty_by_eta.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/ce_weight_uncertainty_eta.png`
  - What: Mean top‑1 rule weight interval width by η tertiles. Uncertainty: aleatoric via η (higher η = more label noise).
  - How: Bin points by tertiles of `eta` from `rules.jsonl`; compute mean width and 95% CI per tertile and average across runs.
  - Why: Quantifies how aleatoric noise affects uncertainty on rule weights.
  - Interpretation: Wider intervals at higher η indicate more label noise leads to more uncertain rule effects; minimal change suggests robustness to aleatoric uncertainty.

### 3) Are explanations stable across seeds, and how does stability degrade with uncertainty?
- Exact top‑1 match across seeds:
  - CSV: `evaluation/uncertainty_experiments/derived/rule_stability.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/rule_stability_density.png`, `evaluation/uncertainty_experiments/figures/rule_stability_eta.png`
  - What: Exact identity match rate vs density (epistemic) and η (aleatoric) tertiles.
  - How: For each test point, compare the top-1 rule identity across seeds; compute exact-match rate within density/η tertiles and aggregate with 95% CI.
  - Why: Stability across seeds is essential for reliable explanations; uncertainty should explain when stability degrades.
  - Interpretation: Lower stability in high inverse-density or high η bins means explanations are less dependable where uncertainty is higher.
- Set‑level stability (Jaccard@k across seeds):
  - CSV: `evaluation/uncertainty_experiments/derived/rule_stability_jaccard.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/rule_stability_jaccard_density.png`, `evaluation/uncertainty_experiments/figures/rule_stability_jaccard_eta.png`
  - What: Average pairwise Jaccard overlap of top‑k rule sets; control k via aggregator flag `--jaccard-k`. Uncertainty: density (epistemic) and η (aleatoric) tertiles.
  - How: For each point and seed, take the top-k rules; compute pairwise Jaccard across seeds; average within density/η tertiles and then across runs.
  - Why: Captures set-level stability beyond exact top-1 identity, more sensitive to near-ties.
  - Interpretation: Higher Jaccard is better; declines with higher epistemic/aleatoric bins indicate brittleness of explanation sets under uncertainty.

### 4) Can DE/density predict rule failure (actionable abstention)?
- Rule failure detection (classification):
  - CSVs: `evaluation/uncertainty_experiments/derived/rule_failure_auroc.csv`, `evaluation/uncertainty_experiments/derived/rule_failure_ap.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/rule_failure_auroc.png`, `evaluation/uncertainty_experiments/figures/rule_failure_ap.png`
  - What: AUROC/AP for detecting low‑precision rules (precision < τ). Control τ via `--rule-failure-tau`. Uncertainty signals: epistemic (DE disagreement, inverse‑density), aleatoric (η); random is a control.
  - How: Label each rule as failure if its precision (per run) < τ. Score detection using per-point signals from the top-1 rule (`de`, `inv_density`, `eta`) and compute AUROC/AP across points; aggregate across runs.
  - Why: If uncertainty signals can flag risky rules, they enable practical abstention and triage policies.
  - Interpretation: Higher AUROC/AP indicates stronger detection; epistemic signals outperforming random supports actionable abstention. Compare across signals and sensitivity to τ.

### 5) Do selective explanation policies improve utility under uncertainty?
- Utility–coverage curves and AUC (classification and thresholded):
  - CSVs: `evaluation/uncertainty_experiments/derived/selective_utility_curves.csv`, `evaluation/uncertainty_experiments/derived/selective_utility_auc.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/selective_utility_classification.png`, `evaluation/uncertainty_experiments/figures/selective_utility_thresh_reg.png`,
             `evaluation/uncertainty_experiments/figures/selective_utility_auc_classification.png`, `evaluation/uncertainty_experiments/figures/selective_utility_auc_thresh_reg.png`,
             `evaluation/uncertainty_experiments/figures/selective_utility_auc_distribution_classification.png`, `evaluation/uncertainty_experiments/figures/selective_utility_auc_distribution_thresh_reg.png`
  - What: Compare gating by `epistemic_de` / `epistemic_density` (epistemic) vs `aleatoric_eta` (aleatoric) vs `random`.
  - How: Form policies that select a fraction of points based on descending uncertainty scores; compute rule-utility vs coverage curves and AUC per policy; report mean and distribution across runs.
  - Why: Tests whether uncertainty-aware gating improves overall utility by focusing explanations where they are most reliable/useful.
  - Interpretation: Higher AUC (or curves dominating others) indicates a better utility–coverage tradeoff; epistemic policies often help at low-to-mid coverage, random is a baseline.

### 6) Are exceedance probabilities calibrated (thresholded regression)?
- Instance‑level reliability across thresholds t:
  - CSV: `evaluation/uncertainty_experiments/derived/thresh_reg_reliability.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/thresh_reg_ece_by_t.png`
  - What: Calibration of exceedance probabilities p(Y>t) across t. Not stratified by uncertainty; complements CE figures.
  - How: For each threshold key `t=...`, compute 10-bin ECE over instance-level exceedance predictions; average per model across runs with 95% CI ribbons.
  - Why: Validates the probabilistic quality of exceedance predictions across thresholds, foundational for thresholded rule analyses.
  - Interpretation: Lower ECE and smooth trends across t indicate good calibration; spikes highlight problematic thresholds.
- Rule‑level exceedance reliability across t:
  - CSV: `evaluation/uncertainty_experiments/derived/thresh_reg_rule_reliability_by_bin.csv`
  - Figure: `evaluation/uncertainty_experiments/figures/thresh_reg_rule_ece_by_t.png`
  - What: Rule‑level exceedance calibration (ECE) aggregated per threshold t. Not stratified by uncertainty.
  - How: From compact rows (bin==0) in the CSV, collect `rule_ece` per t; plot mean and 95% CI across runs.
  - Why: Connects rule-based decisions to exceedance outcomes; checks calibration at the explanation level across thresholds.
  - Interpretation: Lower mean ECE across t is better; variation with t indicates threshold dependence of rule calibration.

### 7) How do shift (δ) and holes (s) affect calibration, stability, and utility?
- Slope indices (per metric):
  - CSV: `evaluation/uncertainty_experiments/derived/uncertainty_sensitivity_slopes.csv`
  - Figures: `evaluation/uncertainty_experiments/figures/uncertainty_sensitivity_slopes_shift.png`, `evaluation/uncertainty_experiments/figures/uncertainty_sensitivity_slopes_holes.png`
  - What: Linear slopes vs shift δ and hole size s for key metrics (rule_ece, stability_exact, util_auc_de). Uncertainty: both capture epistemic distributional shift/coverage gaps.
  - How: For each metric and grouping, fit a simple linear trend of metric vs δ and vs s across runs; emit per-metric slopes for shift and holes.
  - Why: Summarizes robustness in a single statistic; larger slopes imply stronger degradation under uncertainty shifts.
  - Interpretation: Slopes near zero indicate robustness; positive magnitude implies degradation (direction depends on metric definition). Compare metrics/models to spot vulnerabilities.

### 8) Baselines (dependency‑free) — surrogate tree and stump
- Rule‑level reliability/ECE (baseline proxies):
  - CSVs: `evaluation/uncertainty_experiments/derived/baseline_rule_reliability_by_bin.csv`, `evaluation/uncertainty_experiments/derived/baseline_rule_ece.csv`
  - What: Reliability diagrams/ECE using baseline rule probabilities (leaf probabilities) as proxies.
  - How: Train dependency-free baselines (surrogate tree, stump on proba); compute rule-level reliability/ECE analogously to CE metrics and aggregate across runs.
  - Why: Provides dependency-free checks to confirm qualitative trends are not artifacts of CE; benchmarks CE against simple proxies.
  - Interpretation: If baselines are worse-calibrated yet show similar trends, CE likely adds value; if baselines match CE, effects may stem from dataset structure.

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
- Aggregate (core grid only → `derived/ce_paper_core_v1`): `python -m evaluation.uncertainty_experiments.aggregate --root evaluation/uncertainty_experiments/artifacts --config evaluation/uncertainty_experiments/configs/core_experiment.yaml`
- Figures: `python -m evaluation.uncertainty_experiments.viz.make_figures --derived evaluation/uncertainty_experiments/derived --figdir evaluation/uncertainty_experiments/figures`

## Notes & Caveats
- Alternatives: If present, may appear in baseline CSVs; CE figures focus on factual rules’ |w|‑conditioned calibration.
- Rule failure labeling: τ controls class balance; if AUROC/AP are empty or NaN‑heavy, try a different `--rule-failure-tau` and/or require a minimum rule support.
- Stability: Jaccard@k depends on `--jaccard-k` and seed diversity; use both exact‑match and Jaccard@k for a fuller picture.

