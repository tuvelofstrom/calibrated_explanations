# Calibrated Explanations (CE) - Uncertainty Evaluation Plan (v3)

**Purpose & Scope**
- Evaluate CE explanations under both aleatoric and epistemic regimes, covering rule calibration, stability, faithfulness, and selective utility.
- Primary tasks: binary classification and thresholded regression; regression remains for sanity checks and uncertainty separation.
- Controlled factors follow the uncertainty taxonomy in `SETUP_GUIDE.md`: aleatoric via `hetero.scale`, label noise proxies (`eta`), and coverage targets (`alpha`); epistemic via kNN density, support holes, dimensionality, and covariate shift.

**Key Questions**
- Calibration & Effects: How well do rule and prediction confidences align with empirical outcomes across aleatoric/epistemic slices?
- Robustness: How quickly do stability and faithfulness degrade under epistemic stress compared to pure aleatoric noise?
- Actionability: Does difficulty-aware gating improve selective utility and rule-failure detection versus baselines and ablations?
- Thresholded Reliability: Are exceedance probabilities calibrated across thresholds at both instance and rule levels?

**Status Summary**
- Completed: Core aggregation outputs now include effect-centric metrics, weight-uncertainty by density and eta, rule-failure AUROC/AP, Jaccard stability, and sensitivity slopes (see `RESULTS_GUIDE.md`).
- Completed: Baselines and CE ablations are wired with schema-aligned JSONL dumps.
- Pending decision: Legacy `rule_ece.csv` and `rule_reliability_by_bin.csv` remain as sanity checks; decide whether to retire them or label explicitly in reports.
- In progress: Figures for the new metrics still need to be generated or refreshed (`viz/make_figures.py`).

**Experiment Matrices**
- Core sweep (`configs/core_experiment.yaml`):
  - Focus: canonical 2D setting, `n_cal` in {100, 400}, `holes` in {none, small}, `shift` in {0.0, 0.2}.
  - Use seeds [0, 1, 2], DT depth 6 vs RF 200 trees, calibration {split_cp, jackknife_plus} with `alpha = 0.1`.
  - Targets: produce the paper's baseline tables for calibration, effect coverage, stability, selective utility, and rule failure detection.
- Complementary sweep (`configs/complementary_experiment.yaml`):
  - Stress tests: dims {2, 10}, `n_train` {1000, 5000}, `n_cal` {50, 200}, `shift` up to 0.3, hole preset {none, large}.
  - Ablations: {ce_uncalibrated, ce_no_de, gate_density_only, gate_disagreement_only}; DE ablations {none, no_density, nonconformity_only}; baselines enabled.
  - Targets: analyse modality comparisons, ablation effects, high-dimensional behaviour, and selective utility under stronger epistemic drift.

**Metrics & Analyses (Mapping to Derived Artifacts)**
- Prediction calibration: `ce_reliability_by_weight.csv`, `ce_ece_by_weight.csv`.
- Effect-centric evaluation: `effect_interval_coverage.csv`, `effect_magnitude_calibration.csv`, `effect_sign_consistency.csv`, `effect_rank_correlation.csv`.
- Weight uncertainty: `ce_weight_uncertainty_by_density.csv`, `ce_weight_uncertainty_by_eta.csv`.
- Stability: `rule_stability.csv`, `rule_stability_jaccard.csv`; sensitivity: `uncertainty_sensitivity_slopes.csv`.
- Rule failure detection: `rule_failure_auroc.csv`, `rule_failure_ap.csv` (compare DE, density, eta, random).
- Faithfulness sanity: `rule_faithfulness.csv` (inspect alongside density and eta bins).
- Selective utility: `selective_utility_curves.csv`, `selective_utility_auc.csv`.
- Thresholded reliability: `thresh_reg_reliability.csv`, `thresh_reg_rule_reliability_by_bin.csv` (name differs from the original plan).
- Baselines: `baseline_rule_*` CSVs for surrogate and stump comparisons.

**Outstanding Work**
1. Decide on the fate of legacy rule reliability CSVs; if retained, document their role as auxiliary checks in `RESULTS_GUIDE.md` and in the paper.
2. Regenerate figures once core and complementary grids complete; ensure new metrics have matched visuals (reliability, stability, weight uncertainty, selective utility, failure detection, thresholded calibration).
3. Perform statistical summaries (mean, CI, slopes) per uncertainty factor for each key metric; integrate into the draft paper narrative.
4. Draft result write-ups linking each question to the relevant CSV or figure and uncertainty factor (use `SETUP_GUIDE.md` sweep recommendations to justify axis choices).
5. For ablation and baseline comparisons, define table layouts summarising deltas relative to CE control runs.

**Execution Checklist**
- Run core grid: `python -m evaluation.uncertainty_experiments.grid_runner --config evaluation/uncertainty_experiments/configs/core_experiment.yaml --n-jobs 6 --resume`.
- Run complementary grid after validating core results; consider splitting by task if runtime is high.
- Aggregate after each grid: `python -m evaluation.uncertainty_experiments.aggregate --root evaluation/uncertainty_experiments/artifacts --out evaluation/uncertainty_experiments/derived --n-jobs 6 --jaccard-k 5 --rule-failure-tau 0.65`.
- Regenerate figures: `python -m evaluation.uncertainty_experiments.viz.make_figures --derived evaluation/uncertainty_experiments/derived --figdir evaluation/uncertainty_experiments/figures`.

**Acceptance Criteria**
- Calibration and effect metrics report expected trends (for example, worse calibration at low density or high eta, weight intervals widening with both aleatoric and epistemic signals).
- Stability metrics highlight stronger degradation under epistemic shifts (`holes`, `shift`, high dimensions) than under pure aleatoric changes (`hetero.scale`, low `n_cal`).
- Selective utility and failure detection show DE or density gating outperforming eta and random baselines across tasks and ablations.
- Thresholded reliability curves remain near target for well-specified regimes and surface deviations under stressed settings.
- Baseline and ablation tables provide clear evidence of calibration and utility trade-offs for the paper.

**Risks & Mitigations**
- Runtime and scale: monitor grid expansion (especially the complementary config); prune combinations or use `--n-jobs` judiciously.
- Storage: ensure `dump_rules` and `dump_alternatives` outputs are required before long sweeps; archive or clean intermediate artifacts after aggregation.
- Visualisation drift: update `make_figures.py` if new metrics require additional plots; track figure regeneration alongside CSV timestamps.

**Next Steps (Short Order)**
1. Launch the core grid and verify derived CSV freshness; regenerate figures.
2. Review whether to deprecate or reclassify legacy rule reliability CSVs; update `RESULTS_GUIDE.md` accordingly.
3. Produce initial analysis notebooks summarising core metrics across uncertainty factors.
4. Execute complementary grid focusing on ablations and baselines; compare against core outcomes.
5. Draft paper-ready summaries for each key question, citing specific CSVs, figures, and config sweeps.
