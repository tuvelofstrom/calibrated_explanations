# Calibrated Explanations (CE) — Uncertainty Evaluation Plan (v2)

**Purpose & Scope**
- Evaluate CE explanations under uncertainty: rule-level calibration, stability, faithfulness, and decision utility.
- Focus: binary classification and thresholded regression (main); continuous regression supports uncertainty separation and sanity checks.
- Controlled factors: aleatoric (σ(x) for regression, η(x) for classification) and epistemic (kNN density, support holes, covariate shift δ).

**Key Questions**
- Rule calibration: Are per-rule confidences calibrated across aleatoric and epistemic regimes? (classification `rule_predict`, thresholded `p_rule_t`)
- Robustness: How do rule stability and faithfulness degrade with epistemic sparsity vs pure aleatoric noise?
- Actionability: Does epistemic gating (DE/density) improve selective explanation utility and predict rule failures?
- Thresholded robustness: Are exceedance probabilities calibrated across t at instance and rule levels?

**Finished Steps (Summary)**
- Grid + runner + artifacts: runs persisted per spec, with resume and parallel options (grid_runner.py:1).
- Prediction metrics: ECE/Brier (classification); coverage/width and conditional coverage (regression) (runner.py:204, runner.py:136).
- Uncertainty signals: DE (nonconformity/disagreement/density), inverse-density, and σ/η proxies logged (runner.py:191, runner.py:214; aggregate coverage by σ/density: aggregate.py:83, aggregate.py:800).
- Selective utility: Risk/utility vs coverage using DE/density/η/random (aggregate.py:880, aggregate.py:1000, aggregate.py:1076).
- Thresholded regression: instance-level exceedance reliability across t (aggregate.py:1115).
- CE factual/alternative dump: per-instance rule records with weight intervals and rule-level predictions (hooks.py:54, hooks.py:165, hooks.py:237). Config toggle exists (`logging.dump_rules`, `logging.dump_alternatives`).

**Gaps Addressed In This v2 Plan**
- Remove mis-specified rule-level calibration vs y_true using `rule_predict`.
- Replace with effect-centric evaluation aligned with CE semantics (weights and intervals).
- Add DE→rule failure detection (AUROC/AP) and failure calibration.
- Analyze weight-interval width vs aleatoric η (not only density) and joint η×density.
- Expand stability beyond exact top-1 match to Jaccard@k of rule sets.
- Add sensitivity slopes vs shift δ and hole size s for key metrics.
- Add rule-level exceedance calibration across thresholds (thresholded regression).
- Minimal baselines/ablations path aligned with M3 without external deps.

**Experimental Design**
- Tasks: classification, thresholded regression; regression for σ/epistemic separation.
- Models: DT(max_depth=6), RF(n_estimators=100–200, max_depth=None).
- Calibration: Venn–Abers (classification); split/cross/jackknife+ CP (regression); exceedance mapping (interval_map now, CPS optional later).
- Data: 2D heteroscedastic with configurable holes and shift; 10D variant for scalability.
- Sweeps: seeds (≥3), n_cal ∈ {50,100,200}, holes ∈ {none, small, large}, shift δ ∈ {0.0, 0.2}, t-values auto or list.

**Metrics & Definitions**
- Effect-centric correctness (CE factual rules)
  - Effect interval coverage: fraction of rules where a fresh counterfactual estimate Δp_cf (within the antecedent region) lies inside [w_low, w_high]; stratify by density/η and |w|.
  - Effect magnitude calibration: bins of |w| vs observed mean |Δp_cf|; report monotonicity and slope.
  - Effect sign faithfulness: P(sign(Δp_cf) = sign(w)) vs density/η; rank correlation Spearman(|w|, |Δp_cf|).
- Thresholded (rule-level)
  - Model-consistency exceedance: compare `p_rule_t` to fresh empirical exceedance under the rule region; report ECE vs t (model-consistent, not label-supervised).
- Rule failure prediction
  - Define failure at point-level for top‑1 rule: empirical rule precision < τ (e.g., τ=0.7). Report AUROC/AP for DE, density, η, and random.
- Weight uncertainty
  - Mean `w_width` by density tertiles and by η tertiles; optional 3×3 density×η.
- Stability
  - Exact top‑1 match rate and Jaccard@k (k=3) across seeds; stratify by density and η tertiles.
- Faithfulness (sanity)
  - Rule Causal Sensitivity@k: mean |Δp| from in‑region and out‑of‑region constrained perturbations; stratify by density/η.
- Prediction level (sanity)
  - ECE/Brier (classification), coverage/width and conditional coverage by σ/density and σ×density (regression).
- Selective utility
  - Utility–coverage curves and AUC for gating by DE/density/η/random (classification and thresholded).
- Sensitivity indices
  - Slopes/deltas vs δ and hole size s for Rule‑ECE, stability, and utility AUC.

**Implementation Plan (Concrete Additions)**
- Aggregation
  - Deprecate previous rule-level reliability vs y_true; stop emitting `rule_reliability_by_bin.csv` and `rule_ece.csv` for classification.
  - Add effect-centric CSVs: `effect_interval_coverage.csv`, `effect_magnitude_calibration.csv`, `effect_sign_consistency.csv`, `effect_rank_correlation.csv`.
  - For thresholded, add `thresh_reg_rule_consistency.csv` (model-consistent ECE vs t) using fresh sampling inside rule regions.
  - Compute DE→rule failure AUROC/AP (`rule_failure_auroc.csv`) using top‑1 rule per point.
  - Extend weight uncertainty to η (`ce_weight_uncertainty_by_eta.csv`) and joint density×η (optional).
  - Add stability Jaccard@k across seeds (`rule_stability_jaccard.csv`).
  - Add δ/s slopes tables for selected metrics (`uncertainty_sensitivity_slopes.csv`).
- Runner/config
  - Ensure `logging.dump_alternatives: true` where alternative analysis is desired (exp_full.yaml:logging).
  - Keep existing toggles for `dump_rules`, `save_metrics`, `save_predictions`.
- Viz
  - Reliability plots (rule-level), stability bars (exact/Jaccard), weight-uncertainty vs density/η, selective utility curves, exceedance rule reliability vs t.
- Baselines/ablations (M3-aligned, minimal)
  - CE ablations: CE‑uncalibrated, CE‑no‑DE, DE components (density‑only, disagreement‑only), random.
  - Minimal baselines: shallow surrogate tree and stump-on-proba (sklearn‑only) producing `baseline_rules.jsonl` aligned to CE schema.

**Acceptance Criteria**
- Rule-level: Reliable measurement of Rule‑ECE for factual and alternative explanations; observed degradation in low support and high η consistent with expectations.
- Failure detection: DE AUROC/AP > η/random on rule‑failure detection in most settings.
- Stability: Exact and Jaccard@k show stronger epistemic sensitivity than aleatoric.
- Thresholded: Low and flat rule‑exceedance ECE across t where mapping is sound; deviations identified otherwise.
- Utility: DE/density gating yield higher utility AUC than η/random on average.

**Risks & Mitigations**
- Runtime/scale: Use paper‑track matrix; parallelize aggregation heavy steps; resume runs.
- Baseline availability: Stick to sklearn‑only surrogates; defer external methods to appendix.
- Faithfulness variability: Keep k and sample_points bounded; report CI; treat as a sanity complement, not a headline.

**Next Steps (Short Order)**
1) Implement rule‑level reliability (classification/thresholded) in aggregator; add plots.
2) Implement DE→rule failure AUROC/AP and tables; add plots.
3) Add w_width vs η and Jaccard@k stability; add δ/s slopes.
4) Enable `dump_alternatives` in configs where needed and include alternative analyses.
5) Optional: wire minimal baselines/ablations for head‑to‑head tables.
