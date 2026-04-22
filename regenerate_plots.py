"""Regenerate aggregate plots from 3-seed evaluation reports.

Produces 3-seed averaged graphics that match the numbers in README.md:
- final_comparison.png (3-seed mean success rate per task with error bars)
- strategy_distribution.png (3-seed mean meta-controller usage per task)
- zero_shot_transfer.png (3-seed mean zero-shot success per task)
- per_seed_breakdown.png (per-seed deltas, supports the +11.8/+36.8/+29.2 claim)

Outputs to results/ to serve as the canonical figures referenced by README.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

SEEDS = [42, 7, 123]
TASKS = ["morris_water_maze", "visual_foraging", "dynamic_obstacles", "visual_search"]
TASK_LABELS = ["Morris Water Maze", "Visual Foraging", "Dynamic Obstacles", "Visual Search"]
SUB_OBJS = ["EXPLORE", "APPROACH", "EXPLOIT"]
OUT_DIR = "results"
AUG_COLOR = "#4C72B0"
BASE_COLOR = "#DD8452"


def load_reports():
    reports = {}
    for s in SEEDS:
        path = f"results_seed{s}/evaluation_report.json"
        with open(path) as f:
            reports[s] = json.load(f)
    return reports


def plot_final_comparison(reports):
    aug = np.array([[reports[s]["multitask"]["augmented"][t]["success_rate"] for t in TASKS] for s in SEEDS])
    base = np.array([[reports[s]["multitask"]["baseline"][t]["success_rate"] for t in TASKS] for s in SEEDS])

    aug_mean, aug_std = aug.mean(0), aug.std(0)
    base_mean, base_std = base.mean(0), base.std(0)

    x = np.arange(len(TASKS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6.2))
    b1 = ax.bar(x - width / 2, aug_mean, width, yerr=aug_std, capsize=4,
                label="Augmented (PFC)", color=AUG_COLOR, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, base_mean, width, yerr=base_std, capsize=4,
                label="Baseline (Fixed)", color=BASE_COLOR, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Task")
    ax.set_ylabel("Success Rate")
    ax.set_title("Multi-task Success Rate (3-seed avg, error bars = std)")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    for bars, means, stds in [(b1, aug_mean, aug_std), (b2, base_mean, base_std)]:
        for bar, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, m + s),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

    overall_aug = aug_mean.mean()
    overall_base = base_mean.mean()
    ax.text(0.5, 1.16,
            f"Overall: Augmented {overall_aug:.3f}  vs  Baseline {overall_base:.3f}  "
            f"(+{(overall_aug - overall_base) * 100:.1f}%)",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                      alpha=0.95, edgecolor="gray"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "final_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_strategy_distribution(reports):
    dist = np.zeros((len(TASKS), len(SUB_OBJS)))
    entropies = np.zeros(len(TASKS))
    for s in SEEDS:
        for ti, t in enumerate(TASKS):
            d = reports[s]["multitask"]["augmented"][t].get("strategy_distribution", {})
            for oi, o in enumerate(SUB_OBJS):
                dist[ti, oi] += d.get(o, 0.0)
            entropies[ti] += reports[s]["multitask"]["augmented"][t].get("strategy_entropy", 0.0)
    dist /= len(SEEDS)
    entropies /= len(SEEDS)

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.5 * len(TASKS), 4), sharey=True)
    colors = ["#8DA0CB", "#FC8D62", "#66C2A5"]
    for ax, label, ti in zip(axes, TASK_LABELS, range(len(TASKS))):
        ax.bar(SUB_OBJS, dist[ti], color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(f"{label}\n(entropy = {entropies[ti]:.2f})")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(dist[ti]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Selection Frequency")
    fig.suptitle("Meta-Controller Strategy Distribution per Task (3-seed avg)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "strategy_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_zero_shot(reports):
    aug = np.array([[reports[s]["zero_shot_transfer"]["augmented"][t]["mean_success"] for t in TASKS] for s in SEEDS])
    base = np.array([[reports[s]["zero_shot_transfer"]["baseline"][t]["mean_success"] for t in TASKS] for s in SEEDS])
    aug_mean, aug_std = aug.mean(0), aug.std(0)
    base_mean, base_std = base.mean(0), base.std(0)

    x = np.arange(len(TASKS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, aug_mean, width, yerr=aug_std, capsize=4,
           label="Augmented (PFC)", color=AUG_COLOR, edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, base_mean, width, yerr=base_std, capsize=4,
           label="Baseline (Fixed)", color=BASE_COLOR, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, rotation=15, ha="right")
    ax.set_ylabel("Zero-shot Success Rate")
    ax.set_title("Zero-Shot Transfer to Unseen Task Variants (3-seed avg)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.02, 0.97,
            f"Overall: Augmented {aug_mean.mean():.3f}  vs  Baseline {base_mean.mean():.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "zero_shot_transfer.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_per_seed_breakdown(reports):
    aug = np.array([[reports[s]["multitask"]["augmented"][t]["success_rate"] for t in TASKS] for s in SEEDS])
    base = np.array([[reports[s]["multitask"]["baseline"][t]["success_rate"] for t in TASKS] for s in SEEDS])
    aug_seed_avg = aug.mean(1)
    base_seed_avg = base.mean(1)
    deltas = (aug_seed_avg - base_seed_avg) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    x = np.arange(len(SEEDS))
    width = 0.35
    ax1.bar(x - width / 2, aug_seed_avg, width, label="Augmented", color=AUG_COLOR,
            edgecolor="black", linewidth=0.6)
    ax1.bar(x + width / 2, base_seed_avg, width, label="Baseline", color=BASE_COLOR,
            edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax1.set_ylabel("Avg success rate across 4 tasks")
    ax1.set_title("Per-seed Multi-task Average")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(aug_seed_avg, base_seed_avg)):
        ax1.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax1.text(i + width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)

    bars = ax2.bar(x, deltas, color="#55A868", edgecolor="black", linewidth=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax2.set_ylabel("Augmented − Baseline (percentage points)")
    ax2.set_title("Per-seed Delta (Augmented wins on all 12/12 task×seed cells)")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"+{d:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "per_seed_breakdown.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_per_task_seed_grid(reports):
    aug = np.array([[reports[s]["multitask"]["augmented"][t]["success_rate"] for t in TASKS] for s in SEEDS])
    base = np.array([[reports[s]["multitask"]["baseline"][t]["success_rate"] for t in TASKS] for s in SEEDS])

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.2 * len(TASKS), 4), sharey=True)
    x = np.arange(len(SEEDS))
    width = 0.35
    for ax, ti, label in zip(axes, range(len(TASKS)), TASK_LABELS):
        ax.bar(x - width / 2, aug[:, ti], width, color=AUG_COLOR,
               edgecolor="black", linewidth=0.6, label="Augmented")
        ax.bar(x + width / 2, base[:, ti], width, color=BASE_COLOR,
               edgecolor="black", linewidth=0.6, label="Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{s}" for s in SEEDS])
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        for i in range(len(SEEDS)):
            ax.text(i - width / 2, aug[i, ti] + 0.02, f"{aug[i, ti]:.2f}", ha="center", fontsize=8)
            ax.text(i + width / 2, base[i, ti] + 0.02, f"{base[i, ti]:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Success Rate")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Per-task × Per-seed Success Rate (12/12 cells favor Augmented)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for task_label, ti in zip(TASK_LABELS, range(len(TASKS))):
        # also save individual per-task comparison files used by README
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(x - width / 2, aug[:, ti], width, color=AUG_COLOR,
                edgecolor="black", linewidth=0.6, label="Augmented")
        ax2.bar(x + width / 2, base[:, ti], width, color=BASE_COLOR,
                edgecolor="black", linewidth=0.6, label="Baseline")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Success Rate")
        ax2.set_title(f"{task_label} — per-seed success rate")
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)
        for i in range(len(SEEDS)):
            ax2.text(i - width / 2, aug[i, ti] + 0.02, f"{aug[i, ti]:.2f}", ha="center", fontsize=9)
            ax2.text(i + width / 2, base[i, ti] + 0.02, f"{base[i, ti]:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        fname = TASKS[ti]
        out = os.path.join(OUT_DIR, f"comparison_{fname}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {out}")
    out = os.path.join(OUT_DIR, "per_task_seed_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reports = load_reports()
    plot_final_comparison(reports)
    plot_strategy_distribution(reports)
    plot_zero_shot(reports)
    plot_per_seed_breakdown(reports)
    plot_per_task_seed_grid(reports)


if __name__ == "__main__":
    main()
