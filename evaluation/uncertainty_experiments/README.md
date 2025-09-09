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
  - Regression: Split Conformal Intervals (90%/80% default); CPS placeholder wiring for exceedance.
- Uncertainty: Ensemble disagreement, kNN density, nonconformity residuals.
- Difficulty Estimator: Composite DE combining normalized components (weights in config) with ablation switches.
- Metrics: ECE/Brier (classification), coverage/width (regression), reliability utility functions.
- Runner: YAML-driven, seeds, artifacts saved under `evaluation/uncertainty_experiments/artifacts/<run_id>/`.

Planned extensions (hooks exist)
- Cross-/jackknife+ conformal; CPS-based exceedance `P(Y>t)` sweeps; ablation matrix and MLflow logging; rule stability and conditional CE analyses across uncertainty bins.

Quickstart
1) Requirements: repo `requirements.txt` already includes numpy, scikit-learn, venn-abers, crepes. If optional libs are missing, the runner degrades gracefully.
2) Smoke run (tiny, fast):
   - `python evaluation/uncertainty_experiments/runner.py --config evaluation/uncertainty_experiments/configs/smoke.yaml`
   - Outputs artifacts (JSON metrics, small CSVs) under `artifacts/<timestamp>.*`.

Repo Integration
- Uses `calibrated_explanations` internal Venn–Abers and `CalibratedExplainer` hooks for rules/conditional CE (see `rules/hooks.py`).
- No tests added; designed not to affect library packaging. Everything lives under `evaluation/`.

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

Notes
- This is a research sandbox. Keep it light to avoid repo bloat.
- Prefer adding new experiments/configs here rather than modifying core library code.

