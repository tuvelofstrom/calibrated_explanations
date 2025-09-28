# Uncertainty Experiments - Setup Guide

This guide explains how to configure and launch the calibrated uncertainty sandbox in `evaluation/uncertainty_experiments`. Each parameter block lists what/how/why, the uncertainty facet it influences (Aleatoric = A, Epistemic = E, Other = O), suggested sweeps (with complementary knobs), and how to interpret downstream impact. Use it alongside `RESULTS_GUIDE.md` to connect setup choices to analysis outputs.

## Launch Paths

### `python evaluation/uncertainty_experiments/runner.py --config <yaml>`
- **What**: Single-run entry point that consumes one YAML/JSON config (e.g., `configs/smoke.yaml`).
- **How**: Duplicate a config, edit values, then call the script from repo root; artifacts land in `output_dir/run_*`.
- **Why**: Fast for smoke tests, debugging new parameters, or running a single ablation without expanding a grid.
- **Uncertainty**: O — execution harness only; it leaves aleatoric and epistemic structure untouched.
- **Interpretation**: Each invocation writes a timestamped folder; keep track of which manual edits produced which artifacts because there is no automatic indexing.

### `python -m evaluation.uncertainty_experiments.grid_runner --config <yaml> [--n-jobs k] [--resume]`
- **What**: Grid executor that expands list-valued fields (tasks, seeds, data sizes, models, calibration methods, ablations) into a Cartesian set of runs.
- **How**: Edit `configs/exp_full.yaml` (or another spec), then invoke via module path; `--n-jobs` overrides the spec default, and `--resume` skips runs with existing `metrics.json`.
- **Why**: Reproducible batch sweeps with automatic run indexing (`index_<experiment>.json`) and slugged `run_<task>_...` directories.
- **Uncertainty**: O — orchestrates batches but does not alter the stochastic profile of individual runs.
- **Interpretation**: Runtime grows with the product of list lengths; prune dimensions up front to keep compute manageable and to avoid overwriting partial grids when resume is off.

## Top-Level Execution Controls

### `experiment`, `output_dir`, `seeds`, `n_jobs`, `resume`
- **What**: Metadata and scheduling flags at the top of the spec (`configs/exp_full.yaml`).
- **How**: Set `experiment` (str) for bookkeeping, choose `output_dir`, provide an explicit `seeds` list, adjust `n_jobs` (used by grid runner unless CLI overrides), and toggle `resume`.
- **Why**: Namespacing keeps artifacts organized; more seeds improve stability estimates; `n_jobs>1` parallelizes runs; resume prevents rerunning completed slugs after an interruption.
- **Uncertainty**: O with an E-side effect — larger `seeds` pools average over training randomness and shrink epistemic variance estimates.
- **Suggested sweeps**: Start with `seeds: [0,1,2]` and expand to `[0,1,2,3,4]` when assessing stability; pair those sweeps with `data.n_cal` or `uncertainty.de_weights` changes to see whether calibration or difficulty metrics remain robust as randomness is reduced.
- **Interpretation**: Larger seed sets multiply runtime linearly; ensure disk space in `output_dir` before launching wide grids, and remember `resume: false` reruns will overwrite `status.json` while leaving previous metrics in place.

### `tasks` vs `task`
- **What**: `tasks` (list) drives grid expansion across `regression`, `classification`, and `thresh_reg`; standalone configs use `task` (scalar).
- **How**: Edit the YAML to include only the task families you need; grid runner expands per task, while the single runner reads `task`.
- **Why**: Limits computation to the modalities relevant to the study or ablation.
- **Uncertainty**: O — chooses which uncertainty story you probe (classification emphasises A-calibration, regression blends A and E coverage, thresholded investigates selective risk).
- **Suggested sweeps**: Combine `tasks: [classification, regression]` with `data.hetero.scale` and `data.holes` sweeps to contrast aleatoric vs epistemic behaviours; add `ablations` such as `ce_no_de` to see how modality interacts with gating.
- **Interpretation**: Each selected task writes distinct metrics and artifacts; keep them separate when aggregating because regression and classification metrics are not comparable.

## Data Generator Controls (`config[data]`)

### `dims`
- **What**: Feature dimensionality (default `[2]`, typical alternatives `[10]`, `[2, 10]`).
- **How**: Provide either a single int (runner) or a list (grid). Higher values expand the grid.
- **Why**: `2` enables direct visualization and easier manual inspection; `10` stresses models and density estimators.
- **Uncertainty**: E — higher dimensionality dilutes neighbourhood density estimates and inflates epistemic difficulty.
- **Suggested sweeps**: Compare `[2]` vs `[10]` (optionally `[2,10]`); increase `uncertainty.knn_k` and try both `dt` and `rf` models in parallel to maintain meaningful density and ensemble disagreement signals.
- **Interpretation**: Density-based uncertainty degrades in higher dimensions; use low dims when comparing against figures and high dims when stress-testing scalability.

### `n_train`, `n_cal`, `n_test`
- **What**: Dataset sizes for train, calibration, and test splits.
- **How**: Supply integers or lists; e.g., `n_cal: [50, 100, 200]` explores calibration-data scarcity.
- **Why**: Varying these sizes highlights sensitivity of calibration and density estimates to sample count.
- **Uncertainty**: Mixed — smaller `n_train` raises E-risk by under-covering feature space, smaller `n_cal` weakens A-calibration, while `n_test` controls evaluation noise (O).
- **Suggested sweeps**: Sweep `n_cal` over `[50, 100, 400]` while pairing with `calibration.regression` methods to observe coverage resilience; increase `n_train` to `[1000, 5000]` alongside `holes` presets to see how additional support mitigates epistemic gaps.
- **Interpretation**: Expect wider confidence intervals and noisier calibration metrics when `n_cal` is small; ensure `n_test` stays large enough (>=500) for reliable aggregate measurements.

### `hetero`
- **What**: Parameters of the heteroscedastic noise surface `sigma(x) = base + scale * sigmoid(a * x1 + b * x2)`.
- **How**: Set `base`, `scale`, `a`, and `b`; omit the block to fall back to homoscedastic noise.
- **Why**: Controls aleatoric uncertainty strength and orientation across the feature space.
- **Uncertainty**: A — larger `scale` and extreme slopes increase label noise, widening inherent uncertainty.
- **Suggested sweeps**: Try `scale` in `[0.0, 0.4, 0.8, 1.2]` with `base` in `[0.05, 0.1, 0.2]`; pair with `calibration.alpha` and `thresholded.t_values` to confirm coverage targets hold as noise intensifies.
- **Interpretation**: Larger `scale` or extreme `a,b` increases label noise gradients; interpret regression coverage figures relative to these settings.

### `holes`
- **What**: Train-support hole presets controlling epistemic gaps (`none`, `small`, `large`) or custom boxes.
- **How**: Use preset strings (grid runner maps them to coordinates) or provide explicit bounding boxes (`[[x1_min, x2_min, x1_max, x2_max], ...]`).
- **Why**: Simulates regions without training data to probe epistemic uncertainty.
- **Uncertainty**: E — removing support increases inverse-density signals and rule instability.
- **Suggested sweeps**: Run `holes: [none, small, large]` and, for targeted studies, add custom rectangles; sweep jointly with `shift` and `uncertainty.knn_k` so density-aware components can adapt to the missing regions.
- **Interpretation**: Larger or additional holes should increase inverse-density signals and degrade rule stability; align analysis bins with the `holes_label` stored in artifacts.

### `shift`
- **What**: Covariate shift magnitude applied to `x2` at test time (e.g., `[0.0, 0.2]`).
- **How**: Provide floats; grid runner iterates across the list.
- **Why**: Tests robustness of calibration and explanations under distribution shift.
- **Uncertainty**: E — shift induces out-of-support evaluation samples, stressing extrapolation.
- **Suggested sweeps**: Explore `[0.0, 0.1, 0.2, 0.3]` while pairing with `holes` sweeps and `calibration.regression` choices to see which combinations recover coverage under shift.
- **Interpretation**: Higher shift values are expected to increase epistemic errors; compare derived metrics across `shift` columns in the CSVs.

## Model Family (`config.models` or `config.model`)

### `type`, `params`
- **What**: Learner choices (`dt`, `rf`) and their hyperparameters.
- **How**: Supply a list of dictionaries for grid runs; single-run configs use `model` with the same schema.
- **Why**: Decision trees expose interpretable structures; random forests stabilize predictions and provide ensemble disagreement signals.
- **Uncertainty**: O — these knobs change inductive bias; they indirectly modulate both A and E behaviour via model fit.
- **Suggested sweeps**: Compare `dt` with `max_depth` in `[3, 6, null]` and `rf` with `n_estimators` `[50, 200]`; pair with `calibration` and `uncertainty.de_weights` sweeps to see how model expressivity interacts with calibration and difficulty gating.
- **Interpretation**: Deeper trees (`max_depth: null`) reduce calibration but add rule variety; more estimators in the forest increase runtime but smooth predictive distributions.

## Calibration Options (`config.calibration`)

### Classification (`venn_abers`, `none`)
- **What**: Methods for calibrating class probabilities.
- **How**: Set `classification: [venn_abers]` (default) or include `none` for ablations; grid runner expands each choice.
- **Why**: Venn-Abers produces label-conditional probability intervals; disabling calibration isolates raw model behaviour.
- **Uncertainty**: A — directly regulates probability calibration quality.
- **Suggested sweeps**: Alternate `['venn_abers']` vs `['venn_abers','none']` while sweeping `data.n_cal` and `seeds` to measure how calibration data volume and randomness affect ECE.
- **Interpretation**: Expect higher ECE in derived metrics when `none` is selected; Venn-Abers is slower but crucial for the calibrated-explainer baseline.

### Regression (`split_cp`, `cross_cp`, `jackknife_plus`)
- **What**: Conformal interval constructors.
- **How**: Provide a list under `regression`; each entry spawns its own run with shared `alpha` and `n_folds`.
- **Why**: `split_cp` is fastest but variance-prone; `cross_cp` and `jackknife_plus` trade compute for tighter, more stable intervals.
- **Uncertainty**: Mixed — controls A coverage guarantees while cross-fitting mitigates E shift sensitivity.
- **Suggested sweeps**: Evaluate `['split_cp', 'cross_cp', 'jackknife_plus']` alongside `data.holes` and `shift` sweeps to see which method holds coverage under epistemic stress.
- **Interpretation**: Compare coverage vs width across methods; tighter intervals at the same coverage indicate better efficiency.

### `alpha`, `n_folds`
- **What**: `alpha` is the miscoverage rate (e.g., `0.1` => 90% nominal coverage); `n_folds` controls the cross-conformal and jackknife folds.
- **How**: Set scalars in the calibration block; `n_folds` is ignored for pure split conformal.
- **Why**: Adjust tolerance for misses and the strength of the cross-fitting procedure.
- **Uncertainty**: A — lower `alpha` tightens coverage; higher `n_folds` reduces variance in residual estimates (with mild E benefits).
- **Suggested sweeps**: Try `alpha` in `[0.05, 0.1, 0.2]` and `n_folds` in `[3, 5, 10]`, paired with `hetero.scale` sweeps to confirm that tighter targets still hold in noisy regions.
- **Interpretation**: Lower `alpha` widens intervals; increasing `n_folds` improves stability but increases runtime roughly linearly.

### Thresholded Regression (`thresholded.method`, `thresholded.t_values`)
- **What**: Configuration for exceedance probabilities `P(Y > t)`.
- **How**: Choose `method: interval_map` (current default) or plan for `cps` when wired; set `t_values` to `auto` (quantiles) or an explicit list (`[-0.5, 0.0, 0.5]`).
- **Why**: Different downstream consumers require either automatic thresholds or fixed decision levels.
- **Uncertainty**: A — leverages calibrated intervals to estimate threshold exceedance; sensitivity grows with aleatoric noise.
- **Suggested sweeps**: Keep `method: interval_map` and compare `t_values: auto` vs explicit grids such as `[-1.0, -0.5, 0.0, 0.5]`; sweep with `calibration.alpha` to study selective-risk curves under different coverage targets.
- **Interpretation**: More thresholds increase per-run metrics; ensure `t_values` cover the region of interest to avoid empty bins in the selective-utility outputs.

## Uncertainty & Difficulty Estimation (`config.uncertainty`)

### `knn_k`
- **What**: Number of neighbours for inverse-density estimation on calibration/test sets.
- **How**: Set an integer (typical 20-100); applies to both calibration and transfer calls in `runner.py`.
- **Why**: Balances smoothness vs locality of the density proxy; smaller `k` captures fine structure but is noisy.
- **Uncertainty**: E — higher `k` smooths density, lower `k` emphasizes local epistemic spikes.
- **Suggested sweeps**: Explore `[15, 30, 60, 120]` while adjusting `data.dims` and `data.holes`; increase `k` when dims rise or hole size grows to avoid degenerate density estimates.
- **Interpretation**: Monitor sensitivity plots (e.g., weight uncertainty by density) when changing `k`; too-small values may yield unstable tertile assignments.

### `de_weights`
- **What**: Weights for the composite Difficulty Estimator over `nonconformity`, `disagreement`, and `inv_density` components.
- **How**: Provide a dict summing to roughly 1.0; zero out components to remove their influence without editing code.
- **Why**: Controls how strongly each uncertainty signal contributes to the difficulty score used in rules and aggregations.
- **Uncertainty**: Mixed — `nonconformity` tracks A residuals, `disagreement` and `inv_density` capture E structure.
- **Suggested sweeps**: Compare `{0.5, 0.3, 0.2}` (default) with `{0.7, 0.2, 0.1}` and `{0.3, 0.3, 0.4}`; pair with `ablations` such as `gate_density_only` or `ce_no_de` to isolate which component drives selective performance.
- **Interpretation**: Reweighting changes `de_score` scales; inspect derived CSV columns (e.g., `de_weights` metadata) to understand how selective-utility curves shift.

### `ablations`
- **What**: Profiles for dropping or isolating difficulty components (`none`, `no_disagreement`, `no_density`, `nonconformity_only`).
- **How**: Add entries to the list; grid runner expands combinations per task/model.
- **Why**: Systematically isolate which component drives performance or stability degradations.
- **Uncertainty**: Mixed — each option removes an A (nonconformity) or E (density/disagreement) contributor.
- **Suggested sweeps**: Run `[none, no_density, nonconformity_only]` alongside `de_weights` adjustments and `holes`/`shift` sweeps to see how each component responds to epistemic gaps.
- **Interpretation**: Compare runs tagged with the ablation name in aggregated CSVs; expect directional shifts (e.g., removing density weakens epistemic separation).

## Ablations and Baselines (`config.ablations`, `config.baselines`)

### CE Ablations (`none`, `ce_uncalibrated`, `ce_no_de`, `gate_density_only`, `gate_disagreement_only`, `random`)
- **What**: Switches that modify calibrated explainer behaviour.
- **How**: Populate the `ablations` list; grid runner injects `cfg["ablation"]` per run, consumed inside `runner.py`.
- **Why**: Enables the planned study of calibration, difficulty weighting, and gating strategies.
- **Uncertainty**: Mixed — `ce_uncalibrated` removes A calibration, `ce_no_de` drops E-aware gating, `gate_density_only`/`gate_disagreement_only` isolate specific E signals, `random` serves as an O control.
- **Suggested sweeps**: Combine `[none, ce_uncalibrated, ce_no_de]` with `data.holes` and `uncertainty.de_weights` sweeps to characterise how calibration and gating respond to epistemic stress; include `gate_density_only` when testing density sensitivity at higher `knn_k`.
- **Interpretation**: Derived CSVs include an `ablation` column; use it to facet figures and ensure baselines are compared against the appropriate control.

### Baselines (`surrogate_tree`, `stump_on_proba`)
- **What**: Auxiliary explanation methods persisted alongside CE outputs.
- **How**: Add baseline names to the `baselines` list and enable `logging.dump_baselines`.
- **Why**: Provides rule-based comparators without external dependencies.
- **Uncertainty**: O — baselines do not change uncertainty estimates but offer reference explanations.
- **Suggested sweeps**: Run both baselines with the same `seeds` and `holes` sweeps used for CE so you can attribute differences to the explainer rather than the data.
- **Interpretation**: Baseline rule JSONL files live alongside the main run artifacts; aggregate scripts can ingest them when baseline columns are present.

## Faithfulness Sampling (`config.faithfulness`)

### `k`, `sample_points`
- **What**: Controls for the number of jitter samples per point and the cap on points evaluated for faithfulness metrics.
- **How**: Set integers (defaults `k: 8`, `sample_points: 200`).
- **Why**: Higher `k` reduces Monte Carlo noise; `sample_points` bounds runtime when test sets are large.
- **Uncertainty**: O — affects evaluation variance rather than underlying uncertainty.
- **Suggested sweeps**: Increase `k` to `[8, 16, 32]` when studying subtle faithfulness shifts, pairing with `holes` or `shift` sweeps to ensure epistemic gaps are well-resolved.
- **Interpretation**: Faithfulness CSVs and figures will inherit sampling variance; increase `k` before drawing conclusions about subtle differences.

## Logging & Artifact Toggles (`config.logging`)

### `save_predictions`, `save_metrics`, `dump_rules`, `dump_alternatives`, `dump_baselines`, `mlflow`
- **What**: Flags controlling which artifacts are persisted per run.
- **How**: Set booleans; defaults keep predictions, metrics, rules, and baselines, while MLflow stays off.
- **Why**: Disable heavy dumps when storage is tight or when you only need summary metrics.
- **Uncertainty**: O — logging choice does not change experiment dynamics, but it does gate which uncertainty diagnostics you can compute later.
- **Suggested sweeps**: Keep defaults for diagnostic runs; disable `dump_rules` or `dump_baselines` only when sweeping large `seeds`×`holes` grids and disk is constrained, and pair that decision with verifying `aggregate` inputs so required files remain available.
- **Interpretation**: Turning off dumps saves disk but removes inputs for aggregation; ensure the aggregator requirements align with your logging choices before launching a sweep.

## Post-Run Utilities

### `python -m evaluation.uncertainty_experiments.aggregate --root <artifacts> --out <derived> [--n-jobs k] [--jaccard-k m] [--rule-failure-tau t]`
- **What**: Aggregates per-run JSON artifacts into analysis-ready CSVs.
- **How**: Point `--root` at the run directory (defaults to `evaluation/uncertainty_experiments/artifacts`) and choose an output directory; optional flags control parallelism, rule-overlap depth, and the threshold for labeling rule failures.
- **Why**: Required precursor to figure generation and any table-based analysis.
- **Uncertainty**: O — post-processing only, though `--jaccard-k` tunes the granularity of E-based stability metrics.
- **Suggested sweeps**: Evaluate `--jaccard-k` values in `{3, 5, 10}` together with `uncertainty.de_weights` sweeps to see how stability summaries respond; adjust `--rule-failure-tau` when experimenting with `ce_uncalibrated` so AUROC/AUPRC remain informative.
- **Interpretation**: `--jaccard-k` tunes explanation stability metrics; adjust `--rule-failure-tau` if AUROC/AUPRC outputs are degenerate (check logs for warnings).

### `python -m evaluation.uncertainty_experiments.viz.make_figures --derived <dir> --figdir <dir>`
- **What**: Converts derived CSVs into the standard figure set under `evaluation/uncertainty_experiments/figures`.
- **How**: Provide the derived CSV folder and a target figure directory; rerun after changing derived data.
- **Why**: Maintains a consistent visualization pipeline tied to the derived schema.
- **Uncertainty**: O — figures reflect whatever A/E effects are encoded in the derived data.
- **Suggested sweeps**: Regenerate figures after any `aggregate` sweep involving new `holes`, `shift`, or `de_weights` settings to keep visuals aligned with the updated uncertainty landscape.
- **Interpretation**: If figures fail to update, verify that the derived directory contains fresh CSVs for each metric family impacted by your setup change.

## Tips for Custom Configurations

- **Clone and tweak**: Copy `configs/exp_full.yaml` to a new file when experimenting; commit the spec to track provenance.
- **Prune dimensions early**: Comment out unused list entries before launching the grid to avoid combinatorial blow-up.
- **Check slugs**: `grid_runner` encodes key config fields in the run slug; use it to spot misconfigured sweeps without opening `config.json`.
- **Validate small first**: Run a reduced config (one seed, one task, small data) to ensure new parameters wire through end-to-end before scaling up.
