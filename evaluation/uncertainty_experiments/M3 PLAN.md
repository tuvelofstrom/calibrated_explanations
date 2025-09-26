# M3 — Baselines, Ablations, and Head‑to‑Head CE Evaluation (Paper Track)

Status: Planned (revised: no plugin dependencies)

Goal
- Extend the CE‑first evaluation by adding implementable, dependency‑free baselines and CE ablations; produce head‑to‑head comparisons on CE‑centric metrics (reliability conditioned on rule strength, stability, faithfulness, and selective utility), with uncertainty sensitivity summaries.

Scope
- Tasks: binary classification (primary), thresholded regression (context track).
- Models: DT(max_depth=6), RF(200). Same synthetic uncertainty controls (σ/η, density, holes, shift).
- Baselines (no plugins, no extra packages):
  - Global surrogate tree (shallow sklearn DecisionTree) trained on model predictions; per‑instance “path rule” extraction.
  - Stump‑on‑proba baseline (1‑level tree per model on calibration predictions) for a minimal rule baseline.
  - CE‑uncalibrated (no VA/CP) as a strong ablation baseline; CE‑no‑DE (always show) and gating ablations (density‑only, disagreement‑only), CE‑random.
  - Note: external methods (Anchors, RuleFit, SkopeRules, BRL) are excluded due to plugin/dep constraints; add as future work.

Deliverables
- Baseline artifacts: `baseline_rules.jsonl` per run/baseline (per‑instance rule records aligned to CE schema where possible).
- Derived CSVs (baseline‑tagged): reliability_by_weight, ece_by_weight, rule_direction_consistency, rule_stability, rule_faithfulness (in/out), selective_utility_curves/auc, uncertainty_sensitivity_summary. Weight uncertainty plots only where intervals are defined.
- Figures: head‑to‑head bars/lines for CE vs baselines and CE ablations.
- Config updates: enable baselines/ablations via `exp_full.yaml`.
- README: usage for baselines/ablations; validator updated.

Step-by-step Plan

1) Baseline Implementations (No Plugins, Pure sklearn)
- Purpose: Provide dependency‑free baselines with rule extraction usable in the CE pipeline.
- Implementation:
  - Create `evaluation/uncertainty_experiments/baselines/`:
    - `surrogate_tree.py`: trains a shallow DecisionTreeClassifier/Regressor on model predictions (`y_hat_proba` for classification; `y_hat` for regression). For each test instance, extract the decision path as a conjunction of predicates (e.g., `f1 <= t1 and f3 > t3 ...`). Define:
      - `rule_id = md5(canonical_antecedent)`, `antecedent_str` as conjunction.
      - Weight proxy `w`: leaf probability (classification) minus parent probability along the path (last split), signed to match predicted class; or leaf−global difference if parent unavailable.
      - Interval proxy: Wilson interval for leaf probability using calibration counts at that leaf; if counts < 5, set width to NaN.
    - `stump_on_proba.py`: trains a depth‑1 tree on calibration predictions; rule per instance is the stump predicate covering the instance. Same weight/interval proxies as surrogate.
  - Common interface (module‑level functions):
    - `fit_baseline(model, X_train, y_train, **kwargs)` → returns fitted baseline object with predict/explain methods.
    - `explain_rules(baseline, X_cal, y_cal, X_test, y_test)` → yields per‑instance records with fields aligned to CE rules (see below).
  - Record schema (aligned to CE): `{run_id, baseline, point_id, rule_rank, rule_id, antecedent_str, w?, w_low?, w_high?, w_width?, p_pred_base, y_true, y_pred_base, inv_density, eta}`.
  - For thresholded regression (context), skip baselines unless trivial to add; focus on classification for paper track.

2) Baseline Rule Extraction and Persistence
- Purpose: Generate per-instance baseline rules in the same runs and persist to `baseline_rules.jsonl` per baseline.
- Implementation:
  - Extend `runner.py` to accept `baselines: [surrogate_tree, stump_on_proba]` and `logging.dump_baselines: true`.
  - For each baseline in the list, run its adapter on the test set, and write `baseline_rules.jsonl` aligned to fields used in aggregation:
    - Required: `run_id, baseline, point_id, rule_rank, rule_id, antecedent_str, y_true, y_pred_base, p_pred_base`.
    - Recommended: `w, w_low, w_high, w_width` or `conf`, and covariates `inv_density, eta`, etc.
  - For thresholds/regression, limit to context track (optional).

3) CE Ablations (No Plugins)
- Purpose: Isolate the effect of calibration and gating on CE metrics.
- Implementation:
  - Configuration flags in `exp_full.yaml` under `ablations:` (e.g., `ce_uncalibrated`, `ce_no_de`, `gate_density_only`, `gate_disagreement_only`, `random`).
  - In `grid_runner.py` expansion, create separate runs for each ablation profile.
  - In `runner.py`, propagate flags to explainer (e.g., skip VA/CP for uncalibrated), and to selective utility computation.

4) Aggregation (Baselines + Ablations)
- Purpose: Compute CE-first metrics for baselines and ablations using the same machinery.
- Implementation:
  - Add readers for `baseline_rules.jsonl` (per baseline) mirroring CE’s rules processing:
    - reliability_by_weight (or by confidence tertiles when `w` absent), ece_by_weight, weight_uncertainty_by_density (if intervals available), rule_direction_consistency, rule_stability across seeds, faithfulness (in/out jitter if baseline supplies a usable rule region and model).
  - Tag outputs with `baseline` (string) and `ablation` (string) columns.
  - Reuse sensitivity summary and include per-baseline/per-ablation indices.
  - Parallelize per-run baseline aggregations via joblib (same as faithfulness) when heavy.

5) Visualization (Head‑to‑Head)
- Purpose: Paper-ready comparisons between CE and baselines.
- Implementation:
  - Reliability: bar charts of ECE by weight tertile (CE vs baselines) with CI.
  - Weight uncertainty vs density: grouped bars if baselines provide intervals; otherwise omit.
  - Direction consistency: grouped bars (CE vs baselines) by density and η tertiles.
  - Stability: bars (exact-match rate) by density/η tertiles (CE vs baselines).
  - Faithfulness: grouped bars (mean |Δp| in/out) by density/η; CE vs baselines where defined.
  - Selective utility: AUC distribution bars (means ± CI) comparing CE gating vs baseline alternatives.
  - Thresholded regression (context): ECE vs t lines per model (CE + selected baselines if applicable).

6) Config & Runtime Controls
- Purpose: Control breadth of M3 runs.
- Implementation:
  - Extend `exp_full.yaml` with `baselines:` list and `ablations:` list.
  - Add `n_jobs` propagation to aggregator for faster processing.
  - Provide a “paper-track” reduced matrix (few seeds, limited baselines) and a “full” matrix for appendix.

7) Validator & README
- Purpose: Keep outputs consistent and self-describing.
- Implementation:
  - Update validator to include baseline-specific CSVs and required columns.
  - Extend README with installation notes for optional baseline packages and examples for enabling baselines/ablations.

Acceptance Criteria (M3)
- Baselines run successfully (no extra deps) and produce `baseline_rules.jsonl`.
- Head-to-head figures show CE advantages on at least two of: reliability by weight, stability, faithfulness, utility AUC.
- Sensitivity indices populated per baseline/ablation and consistent with expectations (e.g., CE-uncalibrated worse on reliability).
- Aggregation and plotting complete with parallelism; validator passes.

Notes and References
- Use code patterns from README and notebooks (demo_under_the_hood, demo_binary_classification, demo_regression, demo_probabilistic_regression) for accessing CE objects, predictions, and per‑rule fields.
- Surrogate trees and stumps rely only on sklearn and data present in runs; no plugin system or third‑party explainers are used.
