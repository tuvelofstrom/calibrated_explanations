Calibrated Uncertainty Experiments (Non-Bayesian, Conformal/Venn-Abers)

Overview
- Purpose: End-to-end sandbox for studying aleatoric vs epistemic behaviour of Calibrated Explanations (CE) across classification, regression, and thresholded regression.
- Location: `evaluation/uncertainty_experiments`
- Quick smoke test: `python evaluation/uncertainty_experiments/runner.py --config evaluation/uncertainty_experiments/configs/smoke.yaml`
- Planning docs: `SETUP_GUIDE.md` (configuration knob cheat sheet), `RESULTS_GUIDE.md` (metric lookup), `UNCERTAINTY_EVAL_ACTION_PLAN.md` (current experiment roadmap).

Key Components
- Data generators: 2D and 10D regression / binary classification with heteroscedastic noise sigma(x), feature-conditional label noise eta(x), optional support holes, and covariate shift.
- Models: sklearn decision tree and random forest wrappers with unified fit/predict and ensemble sampling helpers.
- Calibration:
  - Classification: in-repo Venn-Abers implementation (optional "none" ablation).
  - Regression: split conformal, cross-conformal, Jackknife+ (K-fold approximation).
  - Thresholded regression: interval_map exceedance probabilities (CPS placeholder for future work).
- Uncertainty signals: ensemble disagreement, kNN inverse density, nonconformity residuals; Composite Difficulty Estimator with configurable weights and ablations.
- Metrics pipeline: rule-level effect coverage/magnitude/sign, prediction calibration, weight uncertainty, rule failure detection, stability (exact and Jaccard), faithfulness probes, selective utility, and sensitivity slopes.
- Runner: YAML-driven single run execution; artifacts stored under `evaluation/uncertainty_experiments/artifacts/run_*`.

Batch Execution & Aggregation
- Grid executor: `evaluation/uncertainty_experiments/grid_runner.py`
  - Example: `python -m evaluation.uncertainty_experiments.grid_runner --config evaluation/uncertainty_experiments/configs/exp_full.yaml --n-jobs 6 --resume`
  - Expands list-valued fields (tasks, seeds, data sizes, calibration methods, ablations, baselines) into a Cartesian run grid.
- Aggregator: `evaluation/uncertainty_experiments/aggregate.py`
  - Example (filter to the core grid and auto-place outputs in `derived/ce_paper_core_v1`):
    `python -m evaluation.uncertainty_experiments.aggregate --root evaluation/uncertainty_experiments/artifacts --config evaluation/uncertainty_experiments/configs/core_experiment.yaml`
  - Pass `--out` to override the derived directory or `--experiment ce_paper_core_v1` to select runs without a spec file.
  - Emits CSVs including:
    - Regression coverage: `coverage_by_sigma.csv`, `coverage_by_density.csv`, `coverage_by_sigma_density.csv`.
    - CE calibration: `ce_reliability_by_weight.csv`, `ce_ece_by_weight.csv`.
    - Effect-centric metrics: `effect_interval_coverage.csv`, `effect_magnitude_calibration.csv`, `effect_sign_consistency.csv`, `effect_rank_correlation.csv`.
    - Weight uncertainty: `ce_weight_uncertainty_by_density.csv`, `ce_weight_uncertainty_by_eta.csv`.
    - Stability: `rule_stability.csv`, `rule_stability_jaccard.csv`.
    - Rule failure detection: `rule_failure_auroc.csv`, `rule_failure_ap.csv`.
    - Faithfulness: `rule_faithfulness.csv`.
    - Selective utility: `selective_utility_curves.csv`, `selective_utility_auc.csv`.
    - Thresholded reliability: `thresh_reg_reliability.csv`, `thresh_reg_rule_reliability_by_bin.csv`.
    - Sensitivity summaries: `uncertainty_sensitivity_summary.csv`, `uncertainty_sensitivity_slopes.csv`.
    - Baseline outputs: `baseline_rule_*` (surrogate tree and stump comparisons).
- Figure generation: `evaluation/uncertainty_experiments/viz/make_figures.py`
  - Example: `python -m evaluation.uncertainty_experiments.viz.make_figures --derived evaluation/uncertainty_experiments/derived --figdir evaluation/uncertainty_experiments/figures`
  - Produces PNGs for calibration, weight uncertainty, stability (exact/Jaccard), rule failure ROC-style bars, selective-utility curves, regression coverage, and thresholded reliability.

Experiment Specs
- Smoke: `configs/smoke.yaml` — minimal single-run sanity check.
- Full grid: `configs/exp_full.yaml` — legacy comprehensive sweep (expand as needed).
- Paper baseline: `configs/core_experiment.yaml` — targeted 2D runs for headline tables.
- Stress and ablations: `configs/complementary_experiment.yaml` — high-dim + ablation/baseline comparisons.
- Complete sweep: `configs/complete_experiment.yaml` — exhaustive matrix for extended compute windows.

Repro Checklist
1. Choose a config (see above) and launch via `grid_runner.py` or the single `runner.py`.
2. Aggregate results into CSVs with the aggregator command.
3. Regenerate figures.
4. Use `RESULTS_GUIDE.md` to map research questions to CSVs/figures; consult `SETUP_GUIDE.md` when adjusting parameters.
5. Track progress and open questions in `UNCERTAINTY_EVAL_ACTION_PLAN.md`.
