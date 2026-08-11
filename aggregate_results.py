"""
aggregate_results.py — Aggregate per-seed runs into the tables and figures the
README reports.

Reads every `evaluation_report.json` under `results_main/seed*/` (the main
learned-vs-baseline comparison) and `results_ablation_<mode>/seed*/` (the
selection-mechanism ablation arms), and writes:

    results_aggregate/summary.json          every reported number, machine-readable
    results_aggregate/tables.md             the markdown tables pasted into README.md
    results_aggregate/learning_curves.png   success rate vs episodes, mean +/- 95% CI
    results_aggregate/ablation_comparison.png

Reporting follows Agarwal et al. (2021), *Deep Reinforcement Learning at the
Edge of the Statistical Precipice*: interval estimates from a stratified
bootstrap rather than bare means, and the interquartile mean alongside the mean
so that one lucky seed cannot carry an aggregate.

Deltas between arms are differences of success rates and are therefore reported
in **percentage points (pp)**, never with a `%` sign.

    python aggregate_results.py
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["morris_water_maze", "visual_foraging", "dynamic_obstacles", "visual_search"]
TASK_LABELS = {
    "morris_water_maze": "Morris Water Maze",
    "visual_foraging": "Visual Foraging",
    "dynamic_obstacles": "Dynamic Obstacles",
    "visual_search": "Visual Search",
}
ABLATION_MODES = ["random", "fixed-explore", "fixed-approach", "fixed-exploit", "uniform-sum"]
OUT_DIR = "results_aggregate"
N_BOOTSTRAP = 10000
RNG = np.random.RandomState(0)


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(per_seed_values, n_boot=N_BOOTSTRAP, alpha=0.05):
    """
    Percentile bootstrap CI for the mean over seeds.

    `per_seed_values` may be 1-D (one number per seed) or 2-D (seeds x tasks),
    in which case the resampling is stratified by task: the same resampled seed
    set is applied to every task, which is what keeps the interval honest when
    tasks are correlated within a seed.
    """
    values = np.asarray(per_seed_values, dtype=float)
    n_seeds = values.shape[0]
    if n_seeds == 0:
        return float("nan"), float("nan"), float("nan")
    if n_seeds == 1:
        m = float(values.mean())
        return m, m, m

    idx = RNG.randint(0, n_seeds, size=(n_boot, n_seeds))
    means = values[idx].reshape(n_boot, -1).mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def interquartile_mean(values):
    """Mean of the middle 50% of the runs — robust to a single outlying seed."""
    v = np.sort(np.asarray(values, dtype=float).ravel())
    if v.size == 0:
        return float("nan")
    lo, hi = int(np.floor(v.size * 0.25)), int(np.ceil(v.size * 0.75))
    core = v[lo:hi]
    return float(core.mean()) if core.size else float(v.mean())


def fmt(mean, lo, hi, scale=1.0, decimals=3):
    """`mean [lo, hi]` with a fixed number of decimals."""
    if np.isnan(mean):
        return "--"
    return (f"{mean * scale:.{decimals}f} "
            f"[{lo * scale:.{decimals}f}, {hi * scale:.{decimals}f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_runs(pattern):
    """Load every completed run matching a `.../seed<N>/` glob, keyed by seed."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(pattern, "seed*", "evaluation_report.json"))):
        seed = int(os.path.basename(os.path.dirname(path)).replace("seed", ""))
        with open(path) as f:
            runs[seed] = json.load(f)
    return dict(sorted(runs.items()))


def arm_success_matrix(runs, arm="augmented"):
    """(seeds x tasks) matrix of final multi-task success rates."""
    seeds = sorted(runs)
    if not seeds:
        return np.zeros((0, len(TASKS))), []
    matrix = np.array([
        [runs[s]["multitask"][arm][t]["success_rate"] for t in TASKS]
        for s in seeds
    ], dtype=float)
    return matrix, seeds


def arm_transfer_matrix(runs, arm="augmented"):
    seeds = sorted(runs)
    if not seeds:
        return np.zeros((0, len(TASKS))), []
    matrix = np.array([
        [runs[s]["zero_shot_transfer"][arm][t]["mean_success"] for t in TASKS]
        for s in seeds
    ], dtype=float)
    return matrix, seeds


# ═══════════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def per_task_table(aug, base, seeds):
    """Per-task success with CIs, plus the paired per-seed delta in pp."""
    lines = [
        f"| Task | Augmented (learned) | Baseline | Delta (pp) |",
        "|---|---|---|---|",
    ]
    rows = {}
    for j, task in enumerate(TASKS):
        a_m, a_lo, a_hi = bootstrap_ci(aug[:, j])
        b_m, b_lo, b_hi = bootstrap_ci(base[:, j])
        # Paired: the delta is computed within a seed, then bootstrapped.
        d_m, d_lo, d_hi = bootstrap_ci(aug[:, j] - base[:, j])
        lines.append(
            f"| {TASK_LABELS[task]} | {fmt(a_m, a_lo, a_hi)} | {fmt(b_m, b_lo, b_hi)} | "
            f"{fmt(d_m, d_lo, d_hi, scale=100, decimals=1)} |"
        )
        rows[task] = {
            "augmented": {"mean": a_m, "ci95": [a_lo, a_hi]},
            "baseline": {"mean": b_m, "ci95": [b_lo, b_hi]},
            "delta_pp": {"mean": 100 * d_m, "ci95": [100 * d_lo, 100 * d_hi]},
        }

    a_m, a_lo, a_hi = bootstrap_ci(aug)
    b_m, b_lo, b_hi = bootstrap_ci(base)
    d_m, d_lo, d_hi = bootstrap_ci(aug.mean(axis=1) - base.mean(axis=1))
    lines.append(
        f"| **Overall** | **{fmt(a_m, a_lo, a_hi)}** | **{fmt(b_m, b_lo, b_hi)}** | "
        f"**{fmt(d_m, d_lo, d_hi, scale=100, decimals=1)}** |"
    )
    lines.append(
        f"| **Overall (IQM)** | **{interquartile_mean(aug):.3f}** | "
        f"**{interquartile_mean(base):.3f}** | "
        f"**{100 * (interquartile_mean(aug) - interquartile_mean(base)):+.1f}** |"
    )

    wins = int((aug > base).sum())
    cells = aug.size
    rows["overall"] = {
        "augmented": {"mean": a_m, "ci95": [a_lo, a_hi], "iqm": interquartile_mean(aug)},
        "baseline": {"mean": b_m, "ci95": [b_lo, b_hi], "iqm": interquartile_mean(base)},
        "delta_pp": {"mean": 100 * d_m, "ci95": [100 * d_lo, 100 * d_hi]},
        "wins": wins,
        "cells": cells,
        "seeds": seeds,
    }
    return lines, rows


def ablation_table(arms):
    """
    One row per selection mechanism.  `arms` maps mode -> (matrix, seeds).
    Deltas are against the `learned` arm, paired on the seeds the two share.
    """
    learned_matrix, learned_seeds = arms["learned"]
    lines = [
        "| Selection mechanism | Seeds | Multi-task success | vs. learned (pp) | Trained params |",
        "|---|---|---|---|---|",
    ]
    rows = {}
    for mode, (matrix, seeds) in arms.items():
        if matrix.shape[0] == 0:
            continue
        m, lo, hi = bootstrap_ci(matrix)
        params = 231050 if mode == "learned" else 152518

        if mode == "learned":
            delta_str = "--"
            delta = None
        else:
            shared = [s for s in seeds if s in learned_seeds]
            if shared:
                a = matrix[[seeds.index(s) for s in shared]].mean(axis=1)
                b = learned_matrix[[learned_seeds.index(s) for s in shared]].mean(axis=1)
                d_m, d_lo, d_hi = bootstrap_ci(a - b)
                delta_str = fmt(d_m, d_lo, d_hi, scale=100, decimals=1)
                delta = {"mean_pp": 100 * d_m, "ci95_pp": [100 * d_lo, 100 * d_hi],
                         "paired_seeds": shared}
            else:
                delta_str, delta = "--", None

        lines.append(
            f"| `{mode}` | {len(seeds)} | {fmt(m, lo, hi)} | {delta_str} | {params:,} |"
        )
        rows[mode] = {
            "seeds": seeds,
            "multitask_success": {"mean": m, "ci95": [lo, hi], "iqm": interquartile_mean(matrix)},
            "delta_vs_learned": delta,
            "trained_parameters": params,
        }
    return lines, rows


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def _curve_band(histories, grid):
    """Interpolate each seed's eval history onto a shared grid; return mean and CI."""
    curves = []
    for hist in histories:
        if len(hist) < 2:
            continue
        x = np.array([h["episode"] for h in hist], dtype=float)
        y = np.array([h["success_rate"] for h in hist], dtype=float)
        order = np.argsort(x)
        curves.append(np.interp(grid, x[order], y[order]))
    if not curves:
        return None, None, None
    curves = np.array(curves)
    mean = curves.mean(axis=0)
    if curves.shape[0] < 2:
        return mean, mean, mean
    idx = RNG.randint(0, curves.shape[0], size=(2000, curves.shape[0]))
    boots = curves[idx].mean(axis=1)
    return mean, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def plot_learning_curves(main_runs, out_path):
    """Per-task success rate over training, mean over seeds with a 95% band."""
    seeds = sorted(main_runs)
    if not seeds:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, task in zip(axes.ravel(), TASKS):
        aug_hist = [main_runs[s]["_aug_eval_history"][task] for s in seeds
                    if task in main_runs[s].get("_aug_eval_history", {})]
        base_hist = [main_runs[s]["_base_eval_history"][task] for s in seeds
                     if task in main_runs[s].get("_base_eval_history", {})]
        all_x = [h["episode"] for hs in (aug_hist + base_hist) for h in hs]
        if not all_x:
            continue
        grid = np.linspace(min(all_x), max(all_x), 60)

        for hist, colour, label in ((aug_hist, "#4C72B0", "Augmented (learned)"),
                                    (base_hist, "#DD8452", "Baseline")):
            mean, lo, hi = _curve_band(hist, grid)
            if mean is None:
                continue
            ax.plot(grid, mean, color=colour, label=label, lw=2)
            ax.fill_between(grid, lo, hi, color=colour, alpha=0.2, lw=0)

        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Training episodes (this task)")
        ax.set_ylabel("Success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Learning curves, mean over {len(seeds)} seeds with 95% bootstrap CI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_ablation(arms, out_path):
    """Multi-task success per selection mechanism, with CI whiskers and seed dots."""
    modes = [m for m in arms if arms[m][0].shape[0] > 0]
    if not modes:
        return
    means, los, his = [], [], []
    for m in modes:
        mean, lo, hi = bootstrap_ci(arms[m][0])
        means.append(mean); los.append(mean - lo); his.append(hi - mean)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(modes))
    colours = ["#4C72B0" if m == "learned" else "#937860" if m == "random" else "#B0B0B0"
               for m in modes]
    ax.bar(x, means, yerr=[los, his], capsize=5, color=colours)
    for i, m in enumerate(modes):
        per_seed = arms[m][0].mean(axis=1)
        ax.scatter(np.full(per_seed.shape, x[i]) + RNG.uniform(-0.12, 0.12, per_seed.shape),
                   per_seed, s=14, color="black", alpha=0.55, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Multi-task success rate")
    ax.set_title("Selection mechanism ablation\n(intrinsic rewards held fixed; "
                 "bars are mean with 95% bootstrap CI, dots are individual seeds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    main_runs = load_runs("results_main")
    if not main_runs:
        raise SystemExit(
            "No completed runs in results_main/seed*/. Run ./scripts/run_all.sh first."
        )

    # The main report stores per-task eval history under the per-task results,
    # not at the top level; lift it so the curve plotter can find it.
    for seed, report in main_runs.items():
        report["_aug_eval_history"] = {}
        report["_base_eval_history"] = {}
        for task in TASKS:
            per_task = report.get("_per_task", {})
            if task in per_task:
                report["_aug_eval_history"][task] = per_task[task]["augmented"]["eval_history"]
                report["_base_eval_history"][task] = per_task[task]["baseline"]["eval_history"]

    aug, seeds = arm_success_matrix(main_runs, "augmented")
    base, _ = arm_success_matrix(main_runs, "baseline")
    aug_t, _ = arm_transfer_matrix(main_runs, "augmented")
    base_t, _ = arm_transfer_matrix(main_runs, "baseline")

    print(f"Main comparison: {len(seeds)} seeds {seeds}")

    task_lines, task_rows = per_task_table(aug, base, seeds)

    # Ablation arms.  `learned` comes from the main runs — it is the same
    # training path, so re-running it as a separate arm would only burn compute.
    arms = {"learned": (aug, seeds)}
    for mode in ABLATION_MODES:
        runs = load_runs(f"results_ablation_{mode}")
        if not runs:
            print(f"  (no runs yet for ablation arm '{mode}')")
            continue
        arm_seeds = sorted(runs)
        matrix = np.array([
            [runs[s]["multitask"]["augmented"][t]["success_rate"] for t in TASKS]
            for s in arm_seeds
        ], dtype=float)
        arms[mode] = (matrix, arm_seeds)
        print(f"Ablation '{mode}': {len(arm_seeds)} seeds {arm_seeds}")

    abl_lines, abl_rows = ablation_table(arms)

    zs_a = bootstrap_ci(aug_t)
    zs_b = bootstrap_ci(base_t)
    zs_d = bootstrap_ci(aug_t.mean(axis=1) - base_t.mean(axis=1))

    tables = []
    tables.append(f"### Multi-task success rate ({len(seeds)} seeds, mean [95% CI])\n")
    tables += task_lines
    tables.append("")
    tables.append("### Selection-mechanism ablation\n")
    tables += abl_lines
    tables.append("")
    tables.append("### Zero-shot transfer to unseen task variants\n")
    tables.append("| Arm | Mean success [95% CI] |")
    tables.append("|---|---|")
    tables.append(f"| Augmented (learned) | {fmt(*zs_a)} |")
    tables.append(f"| Baseline | {fmt(*zs_b)} |")
    tables.append(f"| Delta (pp) | {fmt(*zs_d, scale=100, decimals=1)} |")

    tables_path = os.path.join(OUT_DIR, "tables.md")
    with open(tables_path, "w") as f:
        f.write("\n".join(tables) + "\n")
    print(f"  Saved {tables_path}")

    summary = {
        "seeds": seeds,
        "episodes_per_task": 5000,
        "multitask": task_rows,
        "zero_shot_transfer": {
            "augmented": {"mean": zs_a[0], "ci95": list(zs_a[1:])},
            "baseline": {"mean": zs_b[0], "ci95": list(zs_b[1:])},
            "delta_pp": {"mean": 100 * zs_d[0], "ci95": [100 * zs_d[1], 100 * zs_d[2]]},
        },
        "ablation": abl_rows,
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")

    plot_learning_curves(main_runs, os.path.join(OUT_DIR, "learning_curves.png"))
    plot_ablation(arms, os.path.join(OUT_DIR, "ablation_comparison.png"))

    print("\n" + "\n".join(tables))


if __name__ == "__main__":
    main()
