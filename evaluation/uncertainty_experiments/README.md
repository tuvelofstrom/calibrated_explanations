Calibrated Uncertainty Experiments (Non‑Bayesian, Conformal/Venn–Abers)

Overview
- Purpose: End-to-end experimental sandbox for separating aleatoric vs epistemic effects using synthetic controls, non-Bayesian estimators, and conformal/Venn–Abers calibration across classification, regression, and thresholded regression.
- Location: `evaluation/uncertainty_experiments`
- Entry point: `python evaluation/uncertainty_experiments/runner.py --config evaluation/uncertainty_experiments/configs/smoke.yaml`

What’s included (v0, minimal runnable)
- Data generators: 2D/10D regression and binary classification with heteroscedastic noise `σ(x)` and feature-conditional label noise `η(x)`, plus train-support holes and shifts.
- Models: Sklearn DecisionTree/RandomForest wrappers with unified APIs.
- Calibration: 
  - Classification: Venn–Abers (via in-repo implementation) wrapper.
  - Regression: Split Conformal Intervals; Cross-Conformal; Jackknife+ (K-fold approximation). CPS placeholder wiring for exceedance.
- Uncertainty: Ensemble disagreement, kNN density, nonconformity residuals.
- Difficulty Estimator: Composite DE combining normalized components (weights in config) with ablation switches.
- Metrics: ECE/Brier (classification), coverage/width (regression), reliability utility functions.
- Runner: YAML-driven, seeds, artifacts saved under `evaluation/uncertainty_experiments/artifacts/<run_id>/`.

Planned extensions (hooks exist)
- Cross-/jackknife+ conformal; CPS-based exceedance `P(Y>t)` sweeps; ablation matrix and MLflow logging; rule stability and conditional CE analyses across uncertainty bins.

Batch execution and aggregation (added)
- Grid executor: `evaluation/uncertainty_experiments/grid_runner.py`
  - Spec: `evaluation/uncertainty_experiments/configs/exp_full.yaml`
  - Run: `python -m evaluation.uncertainty_experiments.grid_runner --config evaluation/uncertainty_experiments/configs/exp_full.yaml --n-jobs 4 --resume`
  - Parallelism: uses joblib with process backend; `--n-jobs` overrides spec `n_jobs`.
  - Calibration expansion: lists under `calibration.regression` expand into separate runs (e.g., split_cp, cross_cp, jackknife_plus).
- Aggregator: `evaluation/uncertainty_experiments/aggregate.py`
  - Run: `python -m evaluation.uncertainty_experiments.aggregate --root evaluation/uncertainty_experiments/artifacts --out evaluation/uncertainty_experiments/derived [--n-jobs 4]`
  - Outputs CSVs (core): 
    - coverage_by_sigma.csv, coverage_by_density.csv, coverage_by_sigma_density.csv (regression)
    - ce_reliability_by_weight.csv, ce_ece_by_weight.csv (CE base calibration stratified by |w|)
    - ce_weight_uncertainty_by_density.csv (mean w_width across density tertiles)
    - ce_rule_direction_consistency.csv (top‑1 rule sign supports prediction)
    - rule_stability.csv (exact match across seeds by density/η tertiles)
    - rule_faithfulness.csv (mean |Δp| under in‑/out‑of‑region jitter by density/η bins)
    - thresh_reg_reliability.csv (thresholded regression ECE vs t)
    - selective_utility_curves.csv, selective_utility_auc.csv
    - uncertainty_sensitivity_summary.csv (indices/deltas per run)

Figures
- Script: `evaluation/uncertainty_experiments/viz/make_figures.py`
- Run: `python -m evaluation.uncertainty_experiments.viz.make_figures --derived evaluation/uncertainty_experiments/derived --figdir evaluation/uncertainty_experiments/figures`
- Produces:
  - `ce_reliability_by_weight.png`: Base reliability curves stratified by top‑1 |w| tertiles (45° reference, CI over accuracy).
  - `ce_ece_by_weight.png`: ECE bars per |w| tertile (±95% CI).
  - `ce_weight_uncertainty_by_density.png`: Mean weight interval width across density tertiles (±95% CI).
  - `ce_rule_direction_consistency_density.png`, `ce_rule_direction_consistency_eta.png`: Fraction of instances where the top‑1 rule’s sign supports the predicted class (±95% CI).
  - `rule_stability_density.png`, `rule_stability_eta.png`: Exact‑match rate of top‑1 rule across seeds by density/η tertiles (±95% CI).
  - `rule_faithfulness_density.png`, `rule_faithfulness_eta.png`: Mean |Δp| under in‑/out‑of‑region jitter by density/η tertiles (grouped bars with CI).
  - `regression_coverage_sigma.png`: Coverage reliability vs inherent noise. Bars show empirical coverage in σ(x) tertiles with a nominal target line; line shows average interval width. Uniform coverage across σ(x) validates conformal calibration; increasing width with σ(x) visualizes aleatoric effects (intervals widen where noise is higher).
  - `regression_coverage_density.png`: Coverage vs epistemic support. Bars show coverage across density tertiles with width overlay. Under‑coverage in low‑density bins signals epistemic failures or distribution shift; stable coverage across bins indicates robustness.
  - `regression_coverage_sigma_density_heatmap.png`: Joint view of coverage across σ(x) × density tertiles (3×3). Helps isolate zones where calibration struggles (e.g., low‑density/high‑σ corners) and visually separate aleatoric (σ) from epistemic (density) effects.
  - `thresh_reg_ece_by_t.png`: Reliability for exceedance decisions across thresholds t. Shows ECE of P(Y>t) over t; flat, low curves indicate consistent calibration across decision thresholds (a CPS goal), while drift indicates poor mappings or heteroscedastic effects (supports threshold‑robustness claims).
  - `selective_utility_classification.png`, `selective_utility_thresh_reg.png`: Mean utility–coverage with 95% CI ribbons.
  - `selective_utility_auc_distribution_classification.png`, `selective_utility_auc_distribution_thresh_reg.png`: AUC distributions per signal (violin + mean/CI).

Quickstart
1) Requirements: repo `requirements.txt` already includes numpy, scikit-learn, venn-abers, crepes. If optional libs are missing, the runner degrades gracefully.
2) Smoke run (tiny, fast):
   - `python evaluation/uncertainty_experiments/runner.py --config evaluation/uncertainty_experiments/configs/smoke.yaml`
   - Outputs artifacts (JSON metrics, small CSVs) under `artifacts/<timestamp>.*`.
3) Full grid (small example):
   - `python -m evaluation.uncertainty_experiments.grid_runner --config evaluation/uncertainty_experiments/configs/exp_full.yaml`
   - Aggregate: `python -m evaluation.uncertainty_experiments.aggregate --root evaluation/uncertainty_experiments/artifacts --out evaluation/uncertainty_experiments/derived`

Repo Integration
- Uses `calibrated_explanations` internal Venn–Abers and `CalibratedExplainer` hooks for rules/conditional CE (see `rules/hooks.py`).
- No tests added; designed not to affect library packaging. Everything lives under `evaluation/`.

CE Rules Dump (M2.1)
- Enable per-instance local factual rules dump by adding to your config:
  - `logging: { save_predictions: true, save_metrics: true, dump_rules: true }`
- After a run, `rules.jsonl` appears under the run folder with one JSON record per (point_id, rule):
  - Core keys: `run_id, point_id, explanation_type='factual', rule_rank, rule_id, feature_index, feature_name, antecedent_str`.
  - Per‑rule effect (weight) with uncertainty: `w`, `w_low`, `w_high`, `w_width`, and optional `rule_predict`, `rule_predict_low`, `rule_predict_high`.
  - Per‑instance base prediction and labels: `p_pred`, `y_true`, `y_pred`.
  - Uncertainty covariates: `de`, `inv_density`, `eta`.
- Rule IDs are stable (md5 of canonical antecedent string).

Structure
- `data_gen/`: regression.py, classification.py
- `models/`: wrappers.py
- `calib/`: classification_venn_abers.py, regression_cp.py, thresholded.py
- `uncertainty/`: estimators.py
- `difficulty/`: de.py
- `metrics/`: metrics.py
- `rules/`: hooks.py
- `runner.py`: Typer/argparse CLI for grid configs
- `configs/`: `smoke.yaml` example config
- `viz/`: minimal stubs (extend later)

Validator
- Optional: validate derived CSVs have required columns and non‑empty values:
  - `python evaluation/uncertainty_experiments/scripts/validate_derived.py --derived evaluation/uncertainty_experiments/derived`

Notes
- This is a research sandbox. Keep it light to avoid repo bloat.
- Prefer adding new experiments/configs here rather than modifying core library code.
Parallelism
- The aggregator supports `--n-jobs` to parallelize expensive per-run steps (faithfulness sampling and CE metrics generation). Set to your CPU core count (e.g., 4–16) for speedups. Defaults to 1.

Updates (v2)
- Aggregator CLI adds:
  - `--jaccard-k`: top-k size for Jaccard stability (default 3).
  - `--rule-failure-tau`: threshold τ for labeling rule failures as precision<τ (default 0.7).
- New derived CSVs:
  - `rule_reliability_by_bin.csv`, `rule_ece.csv` (classification rule-level reliability; factual and alternative).
  - `thresh_reg_rule_reliability_by_bin.csv` (thresholded rule-level ECE vs t).
  - `ce_weight_uncertainty_by_eta.csv` (weight-interval width vs η tertiles).
  - `rule_stability_jaccard.csv` (avg pairwise Jaccard@k across seeds; k from `--jaccard-k`).
  - `rule_failure_auroc.csv`, `rule_failure_ap.csv` (DE/density/η detection of low-precision rules; τ from `--rule-failure-tau`).
  - `uncertainty_sensitivity_slopes.csv` (slopes vs shift δ and holes size s for key metrics).
- New figures (viz/make_figures.py):
  - `rule_level_reliability.png`, `rule_ece_bars.png` (classification rule-level calibration).
  - `thresh_reg_rule_ece_by_t.png` (thresholded rule-level ECE across thresholds).
  - `ce_weight_uncertainty_eta.png` (aleatoric impact on weight uncertainty).
  - `rule_stability_jaccard_density.png`, `rule_stability_jaccard_eta.png` (set-level stability).
  - `rule_failure_auroc.png`, `rule_failure_ap.png` (DE/density/η for rule failure detection).
  - `uncertainty_sensitivity_slopes_shift.png`, `uncertainty_sensitivity_slopes_holes.png` (robustness to shift and holes).
- Quick interpretation:
  - Rule-level reliability/ECE: closer to 45°/lower ECE indicates calibrated rule confidences; compare factual vs alternative.
  - Weight uncertainty vs η: higher η tertiles should widen intervals (aleatoric effect) independent of density.
  - Jaccard@k stability: drops in low-density bins reflect epistemic volatility of explanation sets.
  - Rule failure detection: DE/density should beat η/random in AUROC/AP, supporting DE as an actionable explanation gate.
