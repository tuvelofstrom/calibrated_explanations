from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure relative imports work when invoked directly
import sys
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_float_safe(v: Any, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _agg_mean_ci(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if not (v is None or isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (np.nan, np.nan)
    m = float(np.mean(vals))
    if len(vals) <= 1:
        return (m, np.nan)
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return (m, 1.96 * se)


def fig_ece_by_ncal(derived: Path, figdir: Path):  # legacy (appendix)
    rows = _read_csv(derived / "ece_by_ncal.csv")
    if not rows:
        return
    # Group by model
    per_model: Dict[str, Dict[float, Tuple[float, float]]] = defaultdict(dict)
    for r in rows:
        model = r.get("model", "model")
        n_cal = _to_float_safe(r.get("n_cal", np.nan))
        mean_ece = _to_float_safe(r.get("mean_ece", np.nan))
        ci = _to_float_safe(r.get("ci95", np.nan))
        per_model[model][n_cal] = (mean_ece, ci)

    plt.figure(figsize=(6, 4))
    for i, (model, series) in enumerate(sorted(per_model.items())):
        xs = sorted(series.keys())
        ys = [series[x][0] for x in xs]
        cis = [series[x][1] for x in xs]
        plt.errorbar(xs, ys, yerr=cis, marker="o", capsize=3, label=model)
    plt.xlabel("N_cal")
    plt.ylabel("ECE (mean ± 95% CI)")
    plt.title("Classification: ECE vs calibration size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "classification_ece_by_ncal.png", dpi=150)
    plt.close()


def fig_ece_by_density(derived: Path, figdir: Path):  # legacy (appendix)
    rows = _read_csv(derived / "ece_by_density.csv")
    if not rows:
        return
    # Aggregate ECE by density_bin
    bins = [0, 1, 2]
    vals: Dict[int, List[float]] = {b: [] for b in bins}
    for r in rows:
        b = int(_to_float_safe(r.get("density_bin", np.nan)))
        e = _to_float_safe(r.get("ece", np.nan))
        if b in vals:
            vals[b].append(e)
    means = [np.mean(vals[b]) if vals[b] else np.nan for b in bins]
    plt.figure(figsize=(5, 4))
    plt.bar(["low", "mid", "high"], means)
    plt.ylabel("ECE")
    plt.title("Classification: ECE by density tertiles")
    plt.tight_layout()
    plt.savefig(figdir / "classification_ece_by_density.png", dpi=150)
    plt.close()


def fig_risk_coverage(derived: Path, figdir: Path):  # legacy (appendix)
    rows = _read_csv(derived / "risk_coverage_classification.csv")
    if not rows:
        return
    # Aggregate mean risk per coverage per signal
    grid: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        signal = r.get("signal", "signal")
        c = _to_float_safe(r.get("coverage", np.nan))
        risk = _to_float_safe(r.get("risk", np.nan))
        grid[signal][c].append(risk)

    plt.figure(figsize=(6, 4))
    for signal, series in sorted(grid.items()):
        xs = sorted(series.keys())
        ys = [np.mean(series[x]) for x in xs]
        plt.plot(xs, ys, marker="o", label=signal)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (error rate)")
    plt.title("Classification: Risk–coverage curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "classification_risk_coverage.png", dpi=150)
    plt.close()


def fig_regression_coverage_by_sigma(derived: Path, figdir: Path):
    rows = _read_csv(derived / "coverage_by_sigma.csv")
    if not rows:
        return
    bins = [0, 1, 2]
    vals: Dict[int, List[float]] = {b: [] for b in bins}
    widths: Dict[int, List[float]] = {b: [] for b in bins}
    for r in rows:
        b = int(_to_float_safe(r.get("sigma_bin", np.nan)))
        cov = _to_float_safe(r.get("coverage", np.nan))
        w = _to_float_safe(r.get("avg_width", np.nan))
        if b in vals:
            vals[b].append(cov)
            widths[b].append(w)
    cov_means = [np.mean(vals[b]) if vals[b] else np.nan for b in bins]
    width_means = [np.mean(widths[b]) if widths[b] else np.nan for b in bins]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(["low", "mid", "high"], cov_means, color="#4C78A8")
    ax1.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    ax1.set_ylabel("Coverage")
    ax2 = ax1.twinx()
    ax2.plot([0, 1, 2], width_means, color="#F58518", marker="o")
    ax2.set_ylabel("Avg width")
    plt.title("Regression: Coverage and width by σ(x) tertiles")
    fig.tight_layout()
    fig.savefig(figdir / "regression_coverage_sigma.png", dpi=150)
    plt.close(fig)


def fig_regression_coverage_by_density(derived: Path, figdir: Path):
    rows = _read_csv(derived / "coverage_by_density.csv")
    if not rows:
        return
    bins = [0, 1, 2]
    vals: Dict[int, List[float]] = {b: [] for b in bins}
    widths: Dict[int, List[float]] = {b: [] for b in bins}
    for r in rows:
        b = int(_to_float_safe(r.get("density_bin", np.nan)))
        cov = _to_float_safe(r.get("coverage", np.nan))
        w = _to_float_safe(r.get("avg_width", np.nan))
        if b in vals:
            vals[b].append(cov)
            widths[b].append(w)
    cov_means = [np.mean(vals[b]) if vals[b] else np.nan for b in bins]
    width_means = [np.mean(widths[b]) if widths[b] else np.nan for b in bins]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(["low", "mid", "high"], cov_means, color="#4C78A8")
    ax1.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    ax1.set_ylabel("Coverage")
    ax2 = ax1.twinx()
    ax2.plot([0, 1, 2], width_means, color="#F58518", marker="o")
    ax2.set_ylabel("Avg width")
    plt.title("Regression: Coverage and width by density tertiles")
    fig.tight_layout()
    fig.savefig(figdir / "regression_coverage_density.png", dpi=150)
    plt.close(fig)


def fig_regression_heatmap_sigma_density(derived: Path, figdir: Path):
    rows = _read_csv(derived / "coverage_by_sigma_density.csv")
    if not rows:
        return
    grid = np.full((3, 3), np.nan)
    counts = np.zeros((3, 3))
    sums = np.zeros((3, 3))
    for r in rows:
        si = int(_to_float_safe(r.get("sigma_bin", np.nan)))
        di = int(_to_float_safe(r.get("density_bin", np.nan)))
        cov = _to_float_safe(r.get("coverage", np.nan))
        if 0 <= si <= 2 and 0 <= di <= 2:
            sums[si, di] += cov
            counts[si, di] += 1
    with np.errstate(invalid="ignore"):
        grid = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Coverage")
    plt.xticks([0, 1, 2], ["low", "mid", "high"])  # density
    plt.yticks([0, 1, 2], ["low", "mid", "high"])  # sigma
    plt.xlabel("Density tertiles")
    plt.ylabel("σ(x) tertiles")
    plt.title("Regression: Coverage heatmap by σ × density")
    plt.tight_layout()
    plt.savefig(figdir / "regression_coverage_sigma_density_heatmap.png", dpi=150)
    plt.close()


def fig_thresh_reg_reliability(derived: Path, figdir: Path):
    rows = _read_csv(derived / "thresh_reg_reliability.csv")
    if not rows:
        return
    # Parse threshold key 't=...'
    from collections import defaultdict
    per_model: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not isinstance(r, dict):
            continue
        model = r.get("model", "model")
        tk = r.get("threshold", "t=0")
        try:
            tval = float(str(tk).split("=")[-1])
        except Exception:
            continue
        e = _to_float_safe(r.get("ece", np.nan))
        if not np.isnan(e):
            per_model[model][tval].append(e)

    plt.figure(figsize=(6, 4))
    for model, series in sorted(per_model.items()):
        xs = sorted(series.keys())
        # align across runs per t
        means = []
        cis = []
        for t in xs:
            vals = series[t]
            if not vals:
                means.append(np.nan)
                cis.append(np.nan)
            else:
                m = float(np.mean(vals))
                if len(vals) <= 1:
                    ci = 0.0
                else:
                    ci = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
                means.append(m)
                cis.append(ci)
        plt.plot(xs, means, marker="o", label=model)
        # CI ribbon
        lo = [m - c for m, c in zip(means, cis)]
        hi = [m + c for m, c in zip(means, cis)]
        plt.fill_between(xs, lo, hi, alpha=0.15)
    plt.xlabel("Threshold t")
    plt.ylabel("ECE (mean ± 95% CI)")
    plt.title("Thresholded regression: ECE across t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "thresh_reg_ece_by_t.png", dpi=150)
    plt.close()


def fig_correlations(derived: Path, figdir: Path):  # optional (appendix)
    rows = _read_csv(derived / "correlations_de_sigma.csv")
    if not rows:
        return
    # mean correlation per model
    per_model: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        model = r.get("model", "model")
        c = _to_float_safe(r.get("spearman_de_sigma", np.nan))
        per_model[model].append(c)

    labels = sorted(per_model.keys())
    means = [np.mean(per_model[k]) if per_model[k] else np.nan for k in labels]
    cis = []
    for k in labels:
        m, ci = _agg_mean_ci(per_model[k])
        cis.append(ci)

    plt.figure(figsize=(6, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, means, yerr=cis, capsize=3)
    plt.xticks(xs, labels)
    plt.ylabel("Spearman(DE, σ(x))")
    plt.title("Alignment: DE vs σ(x)")
    plt.tight_layout()
    plt.savefig(figdir / "alignment_de_sigma.png", dpi=150)
    plt.close()


def fig_ce_reliability_by_weight(derived: Path, figdir: Path):
    rows = _read_csv(derived / "ce_reliability_by_weight.csv")
    if not rows:
        return
    # Aggregate across runs: mean acc vs p per bin per weight tertile
    from collections import defaultdict
    series = defaultdict(lambda: defaultdict(list))  # wbin -> p_bin_center -> list of (p_mean, acc_mean, acc_ci)
    for r in rows:
        wbin = int(_to_float_safe(r.get("weight_bin", np.nan)))
        lo = _to_float_safe(r.get("bin_edge_lo", np.nan))
        hi = _to_float_safe(r.get("bin_edge_hi", np.nan))
        ctr = 0.5 * (lo + hi)
        p_m = _to_float_safe(r.get("p_mean", np.nan))
        a_m = _to_float_safe(r.get("acc_mean", np.nan))
        a_ci = _to_float_safe(r.get("acc_ci95", np.nan))
        series[wbin][ctr].append((p_m, a_m, a_ci))

    plt.figure(figsize=(6, 6))
    xs_ref = np.linspace(0, 1, 50)
    plt.plot(xs_ref, xs_ref, color="gray", linestyle="--", linewidth=1, label="Ideal")
    labels = {0: "low |w|", 1: "mid |w|", 2: "high |w|"}
    for wbin in [0, 1, 2]:
        if wbin not in series:
            continue
        xs = sorted(series[wbin].keys())
        p_means = [np.mean([v[0] for v in series[wbin][x]]) for x in xs]
        acc_means = [np.mean([v[1] for v in series[wbin][x]]) for x in xs]
        acc_cis = [np.mean([v[2] for v in series[wbin][x]]) for x in xs]
        plt.errorbar(p_means, acc_means, yerr=acc_cis, marker="o", capsize=3, label=labels.get(wbin, str(wbin)))
    plt.xlabel("Base calibrated probability")
    plt.ylabel("Empirical accuracy")
    plt.title("CE reliability by top-1 |w| tertiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "ce_reliability_by_weight.png", dpi=150)
    plt.close()


def fig_ce_ece_by_weight(derived: Path, figdir: Path):
    rows = _read_csv(derived / "ce_ece_by_weight.csv")
    if not rows:
        return
    from collections import defaultdict
    vals = defaultdict(list)
    for r in rows:
        wbin = int(_to_float_safe(r.get("weight_bin", np.nan)))
        ece = _to_float_safe(r.get("ece", np.nan))
        vals[wbin].append(ece)
    labels = [0, 1, 2]
    means = [np.mean(vals[b]) if vals[b] else np.nan for b in labels]
    cis = [_agg_mean_ci(vals[b])[1] if vals[b] else np.nan for b in labels]
    plt.figure(figsize=(5, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, means, yerr=cis, capsize=3)
    plt.xticks(xs, ["low", "mid", "high"])
    plt.ylabel("ECE")
    plt.title("CE ECE by top-1 |w| tertiles")
    plt.tight_layout()
    plt.savefig(figdir / "ce_ece_by_weight.png", dpi=150)
    plt.close()


def fig_ce_weight_uncertainty_by_density(derived: Path, figdir: Path):
    rows = _read_csv(derived / "ce_weight_uncertainty_by_density.csv")
    if not rows:
        return
    from collections import defaultdict
    vals = defaultdict(list)
    ci = defaultdict(list)
    for r in rows:
        b = int(_to_float_safe(r.get("density_bin", np.nan)))
        vals[b].append(_to_float_safe(r.get("mean_w_width", np.nan)))
        ci[b].append(_to_float_safe(r.get("ci95", np.nan)))
    labels = [0, 1, 2]
    means = [np.mean(vals[b]) if vals[b] else np.nan for b in labels]
    cis = [np.mean(ci[b]) if ci[b] else np.nan for b in labels]
    plt.figure(figsize=(5, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, means, yerr=cis, capsize=3)
    plt.xticks(xs, ["low", "mid", "high"])
    plt.ylabel("Mean weight interval width")
    plt.title("CE weight uncertainty by density tertiles")
    plt.tight_layout()
    plt.savefig(figdir / "ce_weight_uncertainty_by_density.png", dpi=150)
    plt.close()


def fig_ce_weight_uncertainty_by_eta(derived: Path, figdir: Path):
    rows = _read_csv(derived / "ce_weight_uncertainty_by_eta.csv")
    if not rows:
        return
    from collections import defaultdict
    vals = defaultdict(list)
    for r in rows:
        eb = r.get("eta_bin")
        try:
            b = int(float(eb))
        except Exception:
            continue
        w = _to_float_safe(r.get("mean_w_width", np.nan))
        if not math.isnan(w):
            vals[b].append(w)
    labels = [0, 1, 2]
    means = [np.mean(vals[b]) if vals[b] else np.nan for b in labels]
    cis = [_agg_mean_ci(vals[b])[1] if vals[b] else np.nan for b in labels]
    plt.figure(figsize=(5, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, means, yerr=cis, capsize=3)
    plt.xticks(xs, ["low", "mid", "high"])
    plt.ylabel("Mean weight interval width")
    plt.title("CE weight uncertainty by η tertiles")
    plt.tight_layout()
    plt.savefig(figdir / "ce_weight_uncertainty_eta.png", dpi=150)
    plt.close()


# Removed legacy rule-level reliability plots (classification)
# The CE story focuses on prediction-level calibration conditioned by |w|
# and baseline proxies live under baseline_* CSVs.


def fig_th_rule_ece_by_t(derived: Path, figdir: Path):
    rows = _read_csv(derived / "thresh_reg_rule_reliability_by_bin.csv")
    if not rows:
        return
    # Use the compact rows where bin==0 and 'rule_ece' present
    from collections import defaultdict
    per_t: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        b = _to_float_safe(r.get("bin", np.nan))
        if int(b) != 0:
            continue
        tkey = r.get("threshold", "t=0")
        e = _to_float_safe(r.get("rule_ece", np.nan))
        if not math.isnan(e):
            per_t[tkey].append(e)
    if not per_t:
        return
    # Convert thresholds to numeric for ordering
    items = []
    for k, vs in per_t.items():
        try:
            t = float(str(k).split("=")[-1])
        except Exception:
            continue
        items.append((t, k, vs))
    items.sort(key=lambda x: x[0])
    labels = [k for _, k, _ in items]
    means = [np.mean(vs) if vs else np.nan for _, _, vs in items]
    cis = [_agg_mean_ci(vs)[1] for _, _, vs in items]
    plt.figure(figsize=(6, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, means, yerr=cis, capsize=3)
    plt.xticks(xs, labels, rotation=0)
    plt.ylabel("Rule ECE")
    plt.title("Thresholded regression: Rule-level ECE by t")
    plt.tight_layout()
    plt.savefig(figdir / "thresh_reg_rule_ece_by_t.png", dpi=150)
    plt.close()


def fig_rule_stability_jaccard(derived: Path, figdir: Path):
    rows = _read_csv(derived / "rule_stability_jaccard.csv")
    if not rows:
        return
    from collections import defaultdict
    dens = defaultdict(list)
    eta = defaultdict(list)
    k_vals = []
    for r in rows:
        bt = r.get("bin_type")
        b = int(_to_float_safe(r.get("bin_id", np.nan)))
        v = _to_float_safe(r.get("avg_jaccard", np.nan))
        kv = _to_float_safe(r.get("jaccard_k", np.nan))
        if not math.isnan(kv):
            k_vals.append(int(kv))
        if bt == "density":
            dens[b].append(v)
        elif bt == "eta":
            eta[b].append(v)
    k_disp = None
    if k_vals:
        uniq = sorted(set(k_vals))
        k_disp = str(uniq[0]) if len(uniq) == 1 else "varies"
    if dens:
        labels = [0, 1, 2]
        means = [np.mean(dens[b]) if dens[b] else np.nan for b in labels]
        cis = [_agg_mean_ci(dens[b])[1] if dens[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Avg pairwise Jaccard@3")
        title = "Rule stability (Jaccard@k) by density tertiles"
        if k_disp:
            title = f"Rule stability (Jaccard@{k_disp}) by density tertiles"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figdir / "rule_stability_jaccard_density.png", dpi=150)
        plt.close()
    if eta:
        labels = [0, 1, 2]
        means = [np.mean(eta[b]) if eta[b] else np.nan for b in labels]
        cis = [_agg_mean_ci(eta[b])[1] if eta[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Avg pairwise Jaccard@3")
        title = "Rule stability (Jaccard@k) by η tertiles"
        if k_disp:
            title = f"Rule stability (Jaccard@{k_disp}) by η tertiles"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figdir / "rule_stability_jaccard_eta.png", dpi=150)
        plt.close()


def fig_rule_failure_metrics(derived: Path, figdir: Path):
    auc = _read_csv(derived / "rule_failure_auroc.csv")
    ap = _read_csv(derived / "rule_failure_ap.csv")
    if not auc and not ap:
        return
    from collections import defaultdict
    auc_vals = defaultdict(list)
    ap_vals = defaultdict(list)
    tau_vals = []
    for r in auc:
        s = r.get("signal", "signal")
        auc_vals[s].append(_to_float_safe(r.get("auroc", np.nan)))
        tv = _to_float_safe(r.get("tau", np.nan))
        if not math.isnan(tv):
            tau_vals.append(tv)
    for r in ap:
        s = r.get("signal", "signal")
        ap_vals[s].append(_to_float_safe(r.get("ap", np.nan)))
        tv = _to_float_safe(r.get("tau", np.nan))
        if not math.isnan(tv):
            tau_vals.append(tv)
    tau_disp = None
    if tau_vals:
        uniq = sorted(set([round(t, 6) for t in tau_vals]))
        tau_disp = str(uniq[0]) if len(uniq) == 1 else "varies"
    if auc_vals:
        labels_raw = sorted(auc_vals.keys())
        labels, means, cis = [], [], []
        for k in labels_raw:
            m, ci = _agg_mean_ci(auc_vals[k])
            if not math.isnan(m):
                labels.append(k)
                means.append(m)
                cis.append(ci)
        if labels:
            plt.figure(figsize=(6, 4))
            xs = np.arange(len(labels))
            plt.bar(xs, means, yerr=cis, capsize=3)
            plt.xticks(xs, labels)
            plt.ylabel("AUROC")
            title = "Rule failure detection: AUROC by signal"
            if tau_disp:
                title = f"Rule failure detection (τ={tau_disp}): AUROC by signal"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(figdir / "rule_failure_auroc.png", dpi=150)
            plt.close()
    if ap_vals:
        labels_raw = sorted(ap_vals.keys())
        labels, means, cis = [], [], []
        for k in labels_raw:
            m, ci = _agg_mean_ci(ap_vals[k])
            if not math.isnan(m):
                labels.append(k)
                means.append(m)
                cis.append(ci)
        if labels:
            plt.figure(figsize=(6, 4))
            xs = np.arange(len(labels))
            plt.bar(xs, means, yerr=cis, capsize=3)
            plt.xticks(xs, labels)
            plt.ylabel("Average Precision")
            title = "Rule failure detection: AP by signal"
            if tau_disp:
                title = f"Rule failure detection (τ={tau_disp}): AP by signal"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(figdir / "rule_failure_ap.png", dpi=150)
            plt.close()


def fig_sensitivity_slopes(derived: Path, figdir: Path):
    rows = _read_csv(derived / "uncertainty_sensitivity_slopes.csv")
    if not rows:
        return
    from collections import defaultdict
    # Aggregate by metric -> list of slopes
    by_metric_shift = defaultdict(list)
    by_metric_holes = defaultdict(list)
    for r in rows:
        metric = r.get("metric", "metric")
        sh = _to_float_safe(r.get("slope_shift", np.nan))
        hl = _to_float_safe(r.get("slope_holes", np.nan))
        if not math.isnan(sh):
            by_metric_shift[metric].append(sh)
        if not math.isnan(hl):
            by_metric_holes[metric].append(hl)
    # Plot shift slopes
    if by_metric_shift:
        labels = sorted(by_metric_shift.keys())
        means = [np.mean(by_metric_shift[k]) if by_metric_shift[k] else np.nan for k in labels]
        cis = [_agg_mean_ci(by_metric_shift[k])[1] for k in labels]
        plt.figure(figsize=(6, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, labels, rotation=15)
        plt.ylabel("Slope vs shift δ")
        plt.title("Uncertainty sensitivity (shift)")
        plt.tight_layout()
        plt.savefig(figdir / "uncertainty_sensitivity_slopes_shift.png", dpi=150)
        plt.close()
    if by_metric_holes:
        labels = sorted(by_metric_holes.keys())
        means = [np.mean(by_metric_holes[k]) if by_metric_holes[k] else np.nan for k in labels]
        cis = [_agg_mean_ci(by_metric_holes[k])[1] for k in labels]
        plt.figure(figsize=(6, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, labels, rotation=15)
        plt.ylabel("Slope vs hole size s")
        plt.title("Uncertainty sensitivity (holes)")
        plt.tight_layout()
        plt.savefig(figdir / "uncertainty_sensitivity_slopes_holes.png", dpi=150)
        plt.close()


def fig_ce_rule_direction_consistency(derived: Path, figdir: Path):
    rows = _read_csv(derived / "ce_rule_direction_consistency.csv")
    if not rows:
        return
    from collections import defaultdict
    dens = defaultdict(list)
    eta = defaultdict(list)
    for r in rows:
        bt = r.get("bin_type")
        b = int(_to_float_safe(r.get("bin_id", np.nan)))
        rate = _to_float_safe(r.get("support_rate", np.nan))
        if bt == "density":
            dens[b].append(rate)
        elif bt == "eta":
            eta[b].append(rate)
    # Density
    if dens:
        labels = [0, 1, 2]
        means = [np.mean(dens[b]) if dens[b] else np.nan for b in labels]
        cis = [_agg_mean_ci(dens[b])[1] if dens[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Support rate (sign matches prediction)")
        plt.title("CE rule direction consistency by density tertiles")
        plt.tight_layout()
        plt.savefig(figdir / "ce_rule_direction_consistency_density.png", dpi=150)
        plt.close()
    # Eta
    if eta:
        labels = [0, 1, 2]
        means = [np.mean(eta[b]) if eta[b] else np.nan for b in labels]
        cis = [_agg_mean_ci(eta[b])[1] if eta[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Support rate (sign matches prediction)")
        plt.title("CE rule direction consistency by η tertiles")
        plt.tight_layout()
        plt.savefig(figdir / "ce_rule_direction_consistency_eta.png", dpi=150)
        plt.close()


def fig_rule_stability(derived: Path, figdir: Path):
    rows = _read_csv(derived / "rule_stability.csv")
    if not rows:
        return
    from collections import defaultdict
    dens = defaultdict(list)
    eta = defaultdict(list)
    for r in rows:
        bt = r.get("bin_type")
        b = _to_float_safe(r.get("bin_id", np.nan))
        rate = _to_float_safe(r.get("exact_match_rate", np.nan))
        ci = _to_float_safe(r.get("ci95", np.nan))
        if bt == "density":
            dens[int(b)].append((rate, ci))
        elif bt == "eta":
            eta[int(b)].append((rate, ci))
    # Plot bars with CI for density and eta
    if dens:
        labels = [0, 1, 2]
        means = [np.mean([v[0] for v in dens[b]]) if dens[b] else np.nan for b in labels]
        cis = [np.mean([v[1] for v in dens[b]]) if dens[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Exact match rate")
        plt.title("Rule stability (exact match) by density tertiles")
        plt.tight_layout()
        plt.savefig(figdir / "rule_stability_density.png", dpi=150)
    plt.close()
    # Eta tertiles
    if eta:
        labels = [0, 1, 2]
        means = [np.mean([v[0] for v in eta[b]]) if eta[b] else np.nan for b in labels]
        cis = [np.mean([v[1] for v in eta[b]]) if eta[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Exact match rate")
        plt.title("Rule stability (exact match) by η tertiles")
        plt.tight_layout()
        plt.savefig(figdir / "rule_stability_eta.png", dpi=150)
        plt.close()


def fig_effect_interval_coverage(derived: Path, figdir: Path):
    rows = _read_csv(derived / "effect_interval_coverage.csv")
    if not rows:
        return
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        bt = r.get("bin_type")
        bi = int(_to_float_safe(r.get("bin_id", float("nan"))))
        cov = _to_float_safe(r.get("coverage", float("nan")))
        if bt in ("density", "eta", "absw"):
            acc[(bt, bi)].append(cov)
    for bt in ("density", "eta", "absw"):
        vals = [acc.get((bt, b), []) for b in [0, 1, 2]]
        if not any(vals):
            continue
        labels = [0, 1, 2]
        means = [float(sum(v) / len(v)) if v else float("nan") for v in vals]
        cis = [_agg_mean_ci(v)[1] if v else float("nan") for v in vals]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylim(0, 1)
        plt.ylabel("Coverage inside [w_low, w_high]")
        title = {
            "density": "Effect interval coverage by density",
            "eta": "Effect interval coverage by η",
            "absw": "Effect interval coverage by |w|",
        }[bt]
        plt.title(title)
        plt.tight_layout()
        outname = {
            "density": "effect_interval_coverage_density.png",
            "eta": "effect_interval_coverage_eta.png",
            "absw": "effect_interval_coverage_absw.png",
        }[bt]
        plt.savefig(figdir / outname, dpi=150)
        plt.close()


def fig_effect_magnitude_calibration(derived: Path, figdir: Path):
    rows = _read_csv(derived / "effect_magnitude_calibration.csv")
    if not rows:
        return
    from collections import defaultdict
    bins = defaultdict(list)
    for r in rows:
        bi = int(_to_float_safe(r.get("absw_bin", float("nan"))))
        m = _to_float_safe(r.get("mean_abs_delta", float("nan")))
        if m == m:
            bins[bi].append(m)
    labels = [0, 1, 2]
    means = [np.mean(bins[b]) if bins[b] else np.nan for b in labels]
    cis = [_agg_mean_ci(bins[b])[1] if bins[b] else np.nan for b in labels]
    plt.figure(figsize=(5, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, means, yerr=cis, capsize=3)
    plt.xticks(xs, ["low |w|", "mid |w|", "high |w|"])
    plt.ylabel("Mean |Δp_cf|")
    plt.title("Effect magnitude vs |w| tertiles")
    plt.tight_layout()
    plt.savefig(figdir / "effect_magnitude_vs_absw.png", dpi=150)
    plt.close()


def fig_effect_sign_consistency(derived: Path, figdir: Path):
    rows = _read_csv(derived / "effect_sign_consistency.csv")
    if not rows:
        return
    from collections import defaultdict
    dens = defaultdict(list)
    eta = defaultdict(list)
    for r in rows:
        bt = r.get("bin_type")
        bi = int(_to_float_safe(r.get("bin_id", float("nan"))))
        v = _to_float_safe(r.get("rate", float("nan")))
        if bt == "density":
            dens[bi].append(v)
        elif bt == "eta":
            eta[bi].append(v)
    for name, series in (("density", dens), ("eta", eta)):
        labels = [0, 1, 2]
        means = [np.mean(series[b]) if series[b] else np.nan for b in labels]
        cis = [_agg_mean_ci(series[b])[1] if series[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylim(0, 1)
        plt.ylabel("P(sign(Δp_cf)=sign(w))")
        plt.title(f"Effect sign consistency by {name}")
        plt.tight_layout()
        plt.savefig(figdir / f"effect_sign_consistency_{name}.png", dpi=150)
        plt.close()


def fig_effect_rank_correlation(derived: Path, figdir: Path):
    rows = _read_csv(derived / "effect_rank_correlation.csv")
    if not rows:
        return
    vals = [ _to_float_safe(r.get("spearman_absw_absdelta", float("nan"))) for r in rows ]
    vals = [v for v in vals if v == v]
    if not vals:
        return
    mean, ci = _agg_mean_ci(vals)
    plt.figure(figsize=(5, 4))
    plt.bar([0], [mean], yerr=[ci], capsize=3)
    plt.xticks([0], ["Spearman(|w|,|Δp_cf|)"])
    plt.title("Effect magnitude rank correlation")
    plt.tight_layout()
    plt.savefig(figdir / "effect_rank_correlation.png", dpi=150)
    plt.close()


def fig_rule_faithfulness(derived: Path, figdir: Path):
    rows = _read_csv(derived / "rule_faithfulness.csv")
    if not rows:
        return
    from collections import defaultdict
    dens_in = defaultdict(list)
    dens_out = defaultdict(list)
    eta_in = defaultdict(list)
    eta_out = defaultdict(list)
    for r in rows:
        bt = r.get("bin_type")
        b = int(_to_float_safe(r.get("bin_id", np.nan)))
        m = _to_float_safe(r.get("mean_delta", np.nan))
        ci = _to_float_safe(r.get("ci95", np.nan))
        region = r.get("region", "in")
        if bt == "density":
            (dens_in if region == "in" else dens_out)[b].append((m, ci))
        elif bt == "eta":
            (eta_in if region == "in" else eta_out)[b].append((m, ci))
    if dens_in or dens_out:
        labels = [0, 1, 2]
        means_in = [np.mean([v[0] for v in dens_in[b]]) if dens_in[b] else np.nan for b in labels]
        cis_in = [np.mean([v[1] for v in dens_in[b]]) if dens_in[b] else np.nan for b in labels]
        means_out = [np.mean([v[0] for v in dens_out[b]]) if dens_out[b] else np.nan for b in labels]
        cis_out = [np.mean([v[1] for v in dens_out[b]]) if dens_out[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        width = 0.35
        plt.bar(xs - width/2, means_in, yerr=cis_in, capsize=3, width=width, label="in")
        plt.bar(xs + width/2, means_out, yerr=cis_out, capsize=3, width=width, label="out")
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Mean |Δp| (in-region jitter)")
        plt.title("Faithfulness Δ vs density tertiles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "rule_faithfulness_density.png", dpi=150)
        plt.close()
    if eta_in or eta_out:
        labels = [0, 1, 2]
        means_in = [np.mean([v[0] for v in eta_in[b]]) if eta_in[b] else np.nan for b in labels]
        cis_in = [np.mean([v[1] for v in eta_in[b]]) if eta_in[b] else np.nan for b in labels]
        means_out = [np.mean([v[0] for v in eta_out[b]]) if eta_out[b] else np.nan for b in labels]
        cis_out = [np.mean([v[1] for v in eta_out[b]]) if eta_out[b] else np.nan for b in labels]
        plt.figure(figsize=(5, 4))
        xs = np.arange(len(labels))
        width = 0.35
        plt.bar(xs - width/2, means_in, yerr=cis_in, capsize=3, width=width, label="in")
        plt.bar(xs + width/2, means_out, yerr=cis_out, capsize=3, width=width, label="out")
        plt.xticks(xs, ["low", "mid", "high"])
        plt.ylabel("Mean |Δp| (in-region jitter)")
        plt.title("Faithfulness Δ vs η tertiles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "rule_faithfulness_eta.png", dpi=150)
        plt.close()


def fig_selective_utility(derived: Path, figdir: Path):
    curves = _read_csv(derived / "selective_utility_curves.csv")
    aucs = _read_csv(derived / "selective_utility_auc.csv")
    if not curves or not aucs:
        return
    from collections import defaultdict
    # Classification curves aggregated across runs: mean±CI per coverage per signal
    class_grid = defaultdict(lambda: defaultdict(list))  # signal -> coverage -> [U]
    for r in curves:
        if not isinstance(r, dict) or r.get("task") != "classification":
            continue
        sig = r.get("signal", "signal")
        cov = _to_float_safe(r.get("coverage", np.nan))
        U = _to_float_safe(r.get("utility", np.nan))
        class_grid[sig][cov].append(U)
    if class_grid:
        plt.figure(figsize=(6, 4))
        for sig, series in sorted(class_grid.items()):
            xs = sorted(series.keys())
            means = [np.mean(series[x]) for x in xs]
            ci95 = [_agg_mean_ci(series[x])[1] for x in xs]
            plt.plot(xs, means, marker="o", label=sig)
            # Uncertainty overlay as ribbon
            lo = [m - c if not math.isnan(c) else m for m, c in zip(means, ci95)]
            hi = [m + c if not math.isnan(c) else m for m, c in zip(means, ci95)]
            plt.fill_between(xs, lo, hi, alpha=0.15)
        plt.xlabel("Coverage")
        plt.ylabel("Utility")
        plt.title("Selective explanation utility (classification)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "selective_utility_classification.png", dpi=150)
        plt.close()

    # AUC comparison (classification)
    class_auc = defaultdict(list)
    for r in aucs:
        if not isinstance(r, dict) or r.get("task") != "classification":
            continue
        sig = r.get("signal", "signal")
        a = _to_float_safe(r.get("auc", np.nan))
        class_auc[sig].append(a)
    if class_auc:
        labels = sorted(class_auc.keys())
        means = [np.mean(class_auc[k]) if class_auc[k] else np.nan for k in labels]
        cis = [_agg_mean_ci(class_auc[k])[1] for k in labels]
        plt.figure(figsize=(6, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, labels)
        plt.ylabel("AUC of utility–coverage")
        plt.title("Selective utility: AUC (classification)")
        plt.tight_layout()
        plt.savefig(figdir / "selective_utility_auc_classification.png", dpi=150)
        plt.close()

    # Thresholded regression (aggregate across t by averaging per coverage)
    th_grid = defaultdict(lambda: defaultdict(list))
    for r in curves:
        if not isinstance(r, dict) or r.get("task") != "thresh_reg":
            continue
        sig = r.get("signal", "signal")
        cov = _to_float_safe(r.get("coverage", np.nan))
        U = _to_float_safe(r.get("utility", np.nan))
        th_grid[sig][cov].append(U)
    if th_grid:
        plt.figure(figsize=(6, 4))
        for sig, series in sorted(th_grid.items()):
            xs = sorted(series.keys())
            means = [np.mean(series[x]) for x in xs]
            ci95 = [_agg_mean_ci(series[x])[1] for x in xs]
            plt.plot(xs, means, marker="o", label=sig)
            lo = [m - c if not math.isnan(c) else m for m, c in zip(means, ci95)]
            hi = [m + c if not math.isnan(c) else m for m, c in zip(means, ci95)]
            plt.fill_between(xs, lo, hi, alpha=0.15)
        plt.xlabel("Coverage")
        plt.ylabel("Utility")
        plt.title("Selective explanation utility (thresholded regression)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "selective_utility_thresh_reg.png", dpi=150)
        plt.close()

    # Removed cumulative AUC-vs-coverage curves (confusing); keep AUC distribution + utility–coverage.

    # Distributions of AUC (actual AUC plots) with uncertainty overlays
    # Classification: violin/box style AUC per signal
    class_auc = defaultdict(list)
    for r in aucs:
        if not isinstance(r, dict) or r.get("task") != "classification":
            continue
        sig = r.get("signal", "signal")
        a = _to_float_safe(r.get("auc", np.nan))
        class_auc[sig].append(a)
    if class_auc:
        labels = sorted(class_auc.keys())
        data = [class_auc[k] for k in labels]
        plt.figure(figsize=(7, 4))
        parts = plt.violinplot(data, showmeans=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.3)
        means = [np.mean(d) if d else np.nan for d in data]
        cis = [_agg_mean_ci(d)[1] for d in data]
        xs = np.arange(1, len(labels) + 1)
        plt.errorbar(xs, means, yerr=cis, fmt='o', capsize=3, color='k')
        plt.xticks(xs, labels)
        plt.ylabel("AUC of utility–coverage")
        plt.title("Selective utility AUC distribution (classification)")
        plt.tight_layout()
        plt.savefig(figdir / "selective_utility_auc_distribution_classification.png", dpi=150)
        plt.close()

    # Thresholded regression AUC distribution
    th_auc = defaultdict(list)
    for r in aucs:
        if not isinstance(r, dict) or r.get("task") != "thresh_reg":
            continue
        sig = r.get("signal", "signal")
        a = _to_float_safe(r.get("auc", np.nan))
        th_auc[sig].append(a)
    if th_auc:
        labels = sorted(th_auc.keys())
        data = [th_auc[k] for k in labels]
        plt.figure(figsize=(7, 4))
        parts = plt.violinplot(data, showmeans=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.3)
        means = [np.mean(d) if d else np.nan for d in data]
        cis = [_agg_mean_ci(d)[1] for d in data]
        xs = np.arange(1, len(labels) + 1)
        plt.errorbar(xs, means, yerr=cis, fmt='o', capsize=3, color='k')
        plt.xticks(xs, labels)
        plt.ylabel("AUC of utility–coverage")
        plt.title("Selective utility AUC distribution (thresholded regression)")
        plt.tight_layout()
        plt.savefig(figdir / "selective_utility_auc_distribution_thresh_reg.png", dpi=150)
        plt.close()

    th_auc = defaultdict(list)
    for r in aucs:
        if not isinstance(r, dict) or r.get("task") != "thresh_reg":
            continue
        sig = r.get("signal", "signal")
        a = _to_float_safe(r.get("auc", np.nan))
        th_auc[sig].append(a)
    if th_auc:
        labels = sorted(th_auc.keys())
        means = [np.mean(th_auc[k]) if th_auc[k] else np.nan for k in labels]
        cis = [_agg_mean_ci(th_auc[k])[1] for k in labels]
        plt.figure(figsize=(6, 4))
        xs = np.arange(len(labels))
        plt.bar(xs, means, yerr=cis, capsize=3)
        plt.xticks(xs, labels)
        plt.ylabel("AUC of utility–coverage")
        plt.title("Selective utility: AUC (thresholded regression)")
        plt.tight_layout()
        plt.savefig(figdir / "selective_utility_auc_thresh_reg.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Make overview figures from derived CSVs")
    ap.add_argument("--derived", required=True, help="Path to derived CSV directory")
    ap.add_argument("--figdir", required=True, help="Where to save figures")
    args = ap.parse_args()

    derived = Path(args.derived)
    figdir = Path(args.figdir)
    _ensure_dir(figdir)

    # Generate figures (core story only)
    fig_regression_coverage_by_sigma(derived, figdir)
    fig_regression_coverage_by_density(derived, figdir)
    fig_regression_heatmap_sigma_density(derived, figdir)
    fig_thresh_reg_reliability(derived, figdir)
    # Prediction-level sanity check (classification)
    fig_ece_by_density(derived, figdir)
    # Thresholded regression: rule-level exceedance reliability across t
    fig_th_rule_ece_by_t(derived, figdir)
    # CE-first: rule reliability and ECE
    fig_ce_reliability_by_weight(derived, figdir)
    fig_ce_ece_by_weight(derived, figdir)
    fig_ce_weight_uncertainty_by_density(derived, figdir)
    fig_ce_weight_uncertainty_by_eta(derived, figdir)
    fig_ce_rule_direction_consistency(derived, figdir)
    fig_rule_stability(derived, figdir)
    fig_rule_stability_jaccard(derived, figdir)
    fig_rule_faithfulness(derived, figdir)
    fig_selective_utility(derived, figdir)
    fig_rule_failure_metrics(derived, figdir)
    fig_sensitivity_slopes(derived, figdir)
    # Effect-centric figures
    fig_effect_interval_coverage(derived, figdir)
    fig_effect_magnitude_calibration(derived, figdir)
    fig_effect_sign_consistency(derived, figdir)
    fig_effect_rank_correlation(derived, figdir)
    # Appendix (optional): uncomment to render
    # fig_ece_by_ncal(derived, figdir)
    # fig_risk_coverage(derived, figdir)
    # fig_risk_coverage(derived, figdir)
    # fig_correlations(derived, figdir)

    print(f"Figures written to {figdir}")


if __name__ == "__main__":
    main()
