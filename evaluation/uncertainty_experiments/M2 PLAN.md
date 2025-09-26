# M2 — CE-First Metrics (Local, Per-Instance), Uncertainty Sensitivity, and Visuals

M2 is a specification of one part of the UNCERTAINTY_EVAL_ACTION_PLAN. If anything is unclear in this sub-plan, refer to the main plan for guidance!

Status: Completed

Goal
- Deliver calibrated-explanations (CE)–first metrics and figures that respect CE’s local, per-rule nature: weight-centric reliability (conditioning base calibration on rule strengths), per-instance explanation stability, faithfulness, and selective utility, with explicit sensitivity to aleatoric (σ/η) and epistemic (density/shift/holes) factors. Produce self-describing derived CSVs and reproducible plots.

Scope
- Paper-track tasks: classification, thresholded regression; DT(depth=6), RF(200).
- Add uncertainty sensitivity: per-bin (σ/η tertiles, density tertiles) and sweeps (shift δ, hole size s).

Step-by-step Plan (revised for local CE)

1) Per-Rule Extraction (local, factual explanations only)
- Purpose: Extract, per instance x, the local CE rule(s) with their per-rule effect (weight) and uncertainty interval, along with the instance’s base calibrated prediction. No global applicability scanning is needed.
- Clarification: In CE, “applicability” is inherent — factual explanations are, by definition, the rules that apply to x; alternative explanations are logged separately and excluded from reliability metrics.
- Implementation:
  - rules/hooks.py: for each x, extract: primary rule antecedent (factual) and additional local rules (if returned), weight w with [w_low, w_high], base calibrated probability p_pred, y_true/y_pred, and uncertainty covariates (DE, inv_density, η/σ). Canonicalize antecedent into a stable string and `rule_id = hash(canonical)`.
  - runner.py: when `save_predictions` is true and `dump_rules` is enabled, write `rules.jsonl` (one record per explained (x, rule) pair): `{run_id, point_id, explanation_type='factual', rule_rank, rule_id, antecedent_str, w, w_low, w_high, w_width, p_pred, y_true, y_pred, de, inv_density, eta}`.
  - Optional: alternatives with `explanation_type='alternative'` (excluded from reliability).
- Success: rules.jsonl created for smoke/exp_full; rule_ids deterministic across seeds when antecedents match.

2) CE Reliability by Rule Strength (weight-conditioned base calibration)
- Purpose: Assess whether base calibrated probabilities are equally reliable across instances with weak vs strong explanations.
- Implementation:
  - aggregate.py: read rules.jsonl; per instance, select top‑1 rule by |w|; stratify instances by |w| tertiles; compute base reliability per p_pred bin (10 bins) and ECE per tertile. Outputs: `ce_reliability_by_weight.csv`, `ce_ece_by_weight.csv`.
  - viz: `ce_reliability_by_weight.png` (three curves vs 45°), `ce_ece_by_weight.png` (bars with CI).
- Success: Curves/metrics render; differences (if any) are clear.

3) Selective Explanation Utility (DE-Gated)
- Purpose: Quantify decision value of CE under uncertainty.
- Implementation:
  - aggregate.py: compute utility–coverage curves by sorting by DE (epistemic), by aleatoric proxy (σ/η), and random; utility U=1(correct&shown), 0(abstain), −c(incorrect&shown); config default c=0.5.
  - Outputs: `selective_utility_curves.csv`, `selective_utility_auc.csv` (include bins and sweep tags).
  - viz: `plot_selective_explanation_utility` overlays curves and prints AUC table.
- Success: DE-gated AUC ≥ others in majority of settings; plots render.

4) Per-Instance Explanation Stability (across seeds)
- Purpose: Evaluate consistency of local explanations for the same x across different seeds.
- Implementation:
  - Using canonical rule IDs (step 1), for a set of runs with identical configs except `seed`, gather the primary factual rule per `point_id` and compute stability:
    - Exact-match rate: fraction of points where the canonical antecedent matches across seeds.
    - Jaccard over predicate sets for partial similarity (optional).
  - Stratify per uncertainty bins using point-level σ/η and density.
  - Outputs: `rule_stability.csv` with `{run_group_id, explainer, ablation, bin_type, bin_id, exact_match_rate, jaccard_mean, ci95_*}`.
  - viz: stability vs σ/η × density heatmap and marginal plots.
- Success: Stability decreases in low-density bins; metrics sensible across seeds.

5) Faithfulness (Rule Causal Sensitivity@k)
- Purpose: Sanity-check that antecedent features control calibrated probability.
- Implementation:
  - For each applicable (x, r), sample k perturbations constrained to rule antecedent (e.g., jitter within bin or small ±ε that stays in rule region); recompute calibrated probability.
  - Metric: `faithfulness_delta = mean(|Δp_calibrated|)` over samples; stratify by bins.
  - Outputs: `rule_faithfulness.csv` with per-run/bins mean and CI.
  - Config: `faithfulness.k`, `faithfulness.eps`, `faithfulness.sample_points` to subsample for runtime.
- Success: Non-zero deltas; sensible ordering across bins; runtime managed via subsampling (≤5 min per run for this phase).

6) Thresholded Reliability (model-level)
- Purpose: Reliability of exceedance P(Y>t) (model-level), as context. CE per-rule plots remain weight-centric.
- Implementation:
  - aggregate: `thresh_reg_reliability.csv` (ECE vs t with context columns).
  - viz: `thresh_reg_ece_by_t.png`.
- Success: ECE vs t plots render.

7) Implement parallelism in the aggregate file. 
- Purpose: Speed up the aggregation of results. 
- Implementation: 
- Implementation Plan:
  - Refactor `aggregate.py` to support parallel processing of input files and/or per-run aggregations.
  - Use Python's `concurrent.futures` (e.g., `ProcessPoolExecutor`) or `joblib.Parallel` to parallelize expensive aggregation steps, such as reading and processing large `rules.jsonl` or per-run CSVs.
  - Add a command-line argument (e.g., `--n-jobs`) to control the number of parallel workers, defaulting to the value in config (`n_jobs`).
  - Ensure that parallelism is applied at a level that avoids race conditions (e.g., per-run or per-file, not per-row).
  - Collect and merge results from all workers before writing final output CSVs.
  - Validate that outputs are identical to serial execution (except for runtime).
  - Document parallelism usage in the README and help string.
  - Success: Aggregation runtime is reduced proportionally to available cores; outputs are correct and reproducible.

8) Uncertainty Sensitivity Indices
- Purpose: Summarize CE robustness to aleatoric and epistemic changes.
- Implementation:
  - aggregate: compute indices per explainer and context:
    - Δ‑Rule‑ECE (high–low) for density and σ/η
    - Slope‑Rule‑ECE vs shift δ and holes size s
    - Stability‑Slope vs density; σ‑Elasticity (% drop per σ unit)
    - Faithfulness‑Slope vs density and σ/η
    - DE‑AUROC/AP per bin and degradation vs δ,s
  - Outputs: `uncertainty_sensitivity_summary.csv` with all indices.
- Success: Summary CSV populated; indices show expected trends (density impacts more than σ for stability, etc.).

9) Provenance and Context Columns
- Purpose: Ensure all outputs are self-describing and traceable.
- Implementation:
  - All new CSVs include: `run_id, experiment, task, seed, dims, n_train, n_cal, n_test, shift, holes_label, model, model_params, calib_*, knn_k, de_weights, explainer, ce_ablation, [bin/sweep tags]`.
  - Add a small validator (`scripts/validate_derived.py`) to check required columns and non-empty values.
- Success: Validator passes for smoke and exp_full derived outputs.

10) Visualization Additions
- Purpose: Paper-ready plots for CE-first narrative with uncertainty emphasis.
- Implementation:
  - viz/make_figures.py: add new plotters (rule reliability, selective utility, stability heatmap, faithfulness vs bins, thresholded rule reliability) and call them from main.
- Success: Figures render from derived CSVs without manual edits; README updated accordingly.

11) Performance & Runtime Controls
- Purpose: Keep M2 runnable within reasonable time.
- Implementation:
  - Subsample points for faithfulness (configurable count per run); cap k; parallelize across runs (`--n-jobs`).
  - Provide a paper‑track spec with trimmed matrix for daily iteration.
- Success: M2 end-to-end on paper-spec finishes within a few hours on a 4–8 core machine; smoke test <10 minutes.

Deliverables
- New artifacts: `rules.jsonl` (per-rule weight-centric extraction).
- Derived CSVs: `ce_reliability_by_weight.csv`, `ce_ece_by_weight.csv`, `ce_weight_uncertainty_by_density.csv`, `ce_rule_direction_consistency.csv`, `selective_utility_curves.csv`, `selective_utility_auc.csv`, `rule_stability.csv`, `rule_faithfulness.csv`, `thresh_reg_reliability.csv`, `uncertainty_sensitivity_summary.csv`.
- New figures: CE reliability by rule strength, CE weight uncertainty vs density, CE rule direction consistency, selective utility curves and AUC distributions, stability/faithfulness, thresholded reliability.
- Updated README: usage and interpretation for the CE-first outputs.

Acceptance Criteria (M2)
- CE reliability by rule strength and related diagrams produced; sensitivity indices computed.
- DE-gated utility curves show advantage over baselines (aleatoric/random) in majority of tested contexts.
- Stability and faithfulness figures show plausible degradation with epistemic (density) and bounded effects with aleatoric (σ/η).
- All outputs include full context and pass the validator.
