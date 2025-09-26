# Calibrated Explanations (CE) Evaluation — CE‑First Action Plan and ADRs

## Purpose and Scope
- Primary goal: Evaluate calibrated_explanations (CE) itself — its explanation reliability, stability, faithfulness, and actionable value — not just calibration methods.
- Scope (paper focus): Binary classification and thresholded regression. Regression (continuous) remains as appendix/supporting.
- Synthetic controls: Explicit aleatoric profiles (σ(x), η(x)) and epistemic factors (holes/shift/density) to stress CE under known conditions.

## CE Under Uncertainty — Primary Questions
- Reliability across regimes: Are CE’s rule confidences calibrated within aleatoric (σ/η) and epistemic (density/shift) bins?
- Robustness to shifts: How do CE’s rule precision, stability, and faithfulness change as we increase σ or reduce density/support (holes/shift)?
- Actionability under uncertainty: Does CE’s DE‑gated policy maintain decision utility when epistemic rises and signal where to collect data?

## Uncertainty Axes and Sensitivity Indices
- Axes: σ(x) tertiles (or η(x) levels); density tertiles (kNN vs X_cal), hole size s∈[0,1], shift δ∈[0,0.5].
- Indices (per explainer):
  - Δ‑Rule‑ECE (high vs low density) and (high vs low σ/η)
  - Slope‑Rule‑ECE vs δ and s
  - Stability‑Slope (per density) and σ‑Elasticity (percent drop per σ increase)
  - Faithfulness‑Slope (per axis)
  - DE‑AUROC per bin and its degradation under δ and s

## Primary Outcomes (CE)
- Rule Reliability (calibrated explanations):
  - Rule Reliability Diagrams: expected vs empirical precision for rules; Rule‑ECE, Rule‑Brier, Rule‑NLL.
  - Conditional rule coverage/precision across uncertainty bins (aleatoric σ/η tertiles vs epistemic density tertiles), pre/post calibration.
- Actionable Utility (decision/value):
  - Selective Explanation Utility Curves: utility vs coverage when showing rules below a DE threshold (epistemic abstention) vs aleatoric‑first and random.
  - Rule Failure Prediction: AUROC/AP for DE ranking of rule failures (precision<τ); monotone calibration of DE→failure probability.
- Rule Stability and Faithfulness:
  - Stability (Jaccard over seeds) vs uncertainty bins; hypothesis: stability degrades with epistemic, less with pure aleatoric.
  - Faithfulness (Rule Causal Sensitivity@k): Δ in calibrated probability when antecedent features are counterfactually perturbed inside the rule region.
- Thresholded Regression (exceedance):
  - Rule‑exceedance reliability across thresholds t; ECE vs t for P(Y>t | rule antecedent).

## Secondary Outcomes (pipeline sanity)
- Prediction‑level ECE/Brier/NLL (classification), coverage/width and conditional coverage (regression/CPS), exceedance reliability across t.

## Baselines and CE Ablations
- Baselines: Anchors, RuleFit, SkopeRules, shallow surrogate trees, Bayesian Rule Lists (if feasible).
- CE Ablations: CE‑uncalibrated (no VA/CP), CE‑no‑DE (always show rules), CE‑random tiebreaks, CE‑density‑only, CE‑disagreement‑only.

## Experimental Design (paper track)
- Tasks: classification, thresholded regression.
- Models: DT (max_depth=6), RF (n_estimators=200, max_depth=None).
- Calibration: VA for classification; Jackknife+ CPS for thresholded regression (split/cross in appendix).
- N_cal: {100, 500}. Seeds: 10.
- Data: 2D heteroscedastic with support holes and small shift; 10D variant in appendix for scalability check.
- Shift sweeps: δ∈{0, 0.1, 0.2, 0.3, 0.5}; hole size sweeps s∈{0.0, 0.2, 0.5, 0.8, 1.0} for sensitivity curves.
- Repository track: full grid remains available; plots/tables emphasize CE metrics above.

## Metrics (definitions)
- Rule‑ECE/Brier/NLL: across bins of CE’s rule precision estimate (using calibrated probabilities under rule antecedent).
- Precision@Coverage: precision within top‑k% rule‑applicable points; stratify by DE bins and pre/post calibration.
- Selective Utility AUC: area under utility‑coverage curve for DE‑gated explanations vs alternatives (aleatoric‑first, random).
- Stability (Jaccard): overlap of rule sets across seeds; plot vs density/σ bins.
- Faithfulness Δ: average absolute change in calibrated probability after perturbing rule antecedent features (counterfactual knockoffs) while staying within rule region.
- Failure Detection: AUROC/AP where positives are rule instances with precision<τ; DE used as score.
- Exceedance Reliability: ECE across thresholds t for P(Y>t | rule antecedent).

## Figures and Tables (CE‑centric)
- Fig A (star): Rule Reliability Diagrams (CE vs Anchors vs RuleFit) with Rule‑ECE bars.
- Fig B: Selective explanation utility curves (DE‑gated vs aleatoric vs random).
- Fig C: Heatmaps of rule stability and precision across aleatoric (σ/η) vs epistemic (density) bins; margins show sensitivity slopes.
- Fig D: Thresholded regression rule‑exceedance reliability across t.
- Table A: Head‑to‑head (CE vs baselines): Rule‑ECE, Precision@Coverage, Stability, Faithfulness, DE AUROC for failure.
- Appendix: classic calibration/coverage plots and extended grid comparisons.

## Integration Plan (repo modules to extend)
 - rules/: expose rule_confidence(x, r) for factual explanations and batch APIs to dump per‑instance rule sets with calibrated confidence. Alternative explanations are logged as non‑factual and excluded from rule reliability by default.
- difficulty/: add predict_rule_failure(prob_threshold=τ) producing per‑point failure risk for selective gating.
- metrics/: implement rule_ece, precision_at_coverage, stability_jaccard, faithfulness_delta, selective_utility_auc, failure_auroc/aps.
- viz/: add plot_rule_reliability, plot_selective_explanation_utility, heatmap_rule_stability_vs_uncertainty, and thresholded rule reliability.
- runner/aggregate: add CE‑specific endpoints, baselines/ablations switches, and derived CSVs with full context (incl. run_id, explainer, ablation).
- grid: add axes for explainer baseline and CE ablation; provide a trimmed “paper spec” and a full “repo spec”.

## Updated ADRs (CE‑first)
- ADR‑CE01 — Evaluation Focus: CE is the primary object of evaluation.
  - Decision: Elevate explanation‑level outcomes (Rule‑ECE, selective utility, stability, faithfulness) to primary metrics; demote prediction‑level calibration to secondary.
  - Consequences: Figures and claims center on CE’s value rather than generic calibration.

- ADR‑CE02 — Explanation‑Level Calibration
  - Decision: Define per‑rule calibrated precision from VA/CP under rule antecedent and evaluate with Rule Reliability Diagrams and Rule‑ECE.
  - Consequences: Makes “calibrated explanations” falsifiable and measurable.

- ADR‑CE03 — Baselines and CE Ablations
  - Decision: Include Anchors, RuleFit, SkopeRules, surrogate trees (and BRL if feasible); add CE‑uncalibrated/DE ablations.
  - Consequences: Attributes wins/losses to CE rather than underlying calibration/model choices.

- ADR‑CE04 — DE as Predictive Gate
  - Decision: Evaluate DE by its ability to predict rule failure (precision<τ) and to improve utility via abstention; report AUROC/AP and utility AUC.
  - Consequences: DE becomes actionable, not just correlational.

- ADR‑CE05 — Paper‑Track Matrix Narrowing
  - Decision: Limit paper experiments to two tasks, two models, two N_cal, and Jackknife+ CPS for thresholded regression; keep full grid in repo appendix.
  - Consequences: Clear, focused story; reproducible and efficient runs.

- ADR‑CE06 — Faithfulness as Sanity Check
  - Decision: Include Rule Causal Sensitivity@k via perturbations constrained to rule region.
  - Consequences: Guards against spurious rules; complements stability and calibration.

- ADR‑CE07 — Provenance and Traceability
  - Decision: All derived CSVs include run_id, explainer type, ablation, and full context; figures link to these IDs.
  - Consequences: Seamless drill‑down from paper figures to artifacts.

## Milestones
- M1 (done): Core pipeline; VA; split/cross/jackknife+; grid + aggregation + figures.
- M2: Implement CE‑specific metrics (Rule‑ECE, stability, faithfulness, selective utility) and viz; add run context columns; add uncertainty‑sensitivity indices and sweep support.
- M3: Add baselines (Anchors, RuleFit, SkopeRules, surrogate) and CE ablations; integrate into grid + aggregation.
- M4: CPS for thresholded regression (exceedance) + rule‑exceedance reliability.
- M5: Paper‑track spec + CI smoke; finalize figures/tables; write results section.

## Hypotheses (acceptance criteria)
- H1 (Explanation calibration): CE achieves ≥25% lower Rule‑ECE than baselines across seeds/settings; maintains advantage under shift.
- H2 (DE gating): DE AUROC≥0.75 for rule failure and ≥10% higher utility at 70% coverage vs aleatoric/random gating.
- H3 (Epistemic sensitivity): Rule stability/precision decline with epistemic sparsity (density↓) more than with pure aleatoric increases.
- H4 (Thresholded): CE keeps rule‑exceedance ECE within ±2% across t and outperforms naive mappings.

## Risks & Mitigations
- Baseline availability/perf: If BRL unavailable, document and proceed with others; cap training time and depth.
- Runtime: Use paper‑track spec (small matrix), parallel n_jobs, resume; profile bottlenecks.
- Metric leakage: Keep train/cal/test disjoint; rule faithfulness perturbs within rule constraints only.

## Immediate Next Steps
1) Implement rule‑level metrics + viz; add derived CSVs and figure generation.
2) Wire explainer baselines + CE ablations into grid and aggregator.
3) Integrate CPS P(Y>t) and rule‑exceedance reliability; update thresholded figures.
4) Add shift/holes sweeps and uncertainty‑sensitivity indices to aggregation and viz.
