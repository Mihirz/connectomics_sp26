"""
run_experiment.py — Main Entry Point for the Self-Optimizing Training Experiment

This script orchestrates the full experimental pipeline:

1. SETUP     — Initialize configs, models, environments
2. TRAIN     — Train both augmented and baseline models on the task suite
3. EVALUATE  — Run the full generalizability evaluation
4. REPORT    — Generate comparison plots and statistics

Modes:
  smoke_test  — Quick sanity check (1 task, few episodes, CPU)
  single_task — Train and compare on a single task
  full        — Complete experiment across all tasks with full evaluation
"""

import argparse
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from collections import defaultdict

from config import ExperimentConfig
from models import AugmentedModel, BaselineModel, print_model_comparison
from training import AugmentedTrainer, BaselineTrainer
from evaluate import (
    evaluate_model, run_full_evaluation, eval_strategy_diversity
)
from sub_objectives import SUB_OBJ_NAMES


def set_seed(seed: int):
    """Reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_on_task(
    trainer,
    cfg: ExperimentConfig,
    task_name: str,
    num_episodes: int,
    model_name: str = "model",
    log_fn=None,
) -> dict:
    """
    Train a model on a single task, collecting metrics along the way.

    This is the inner training loop.  It alternates between:
    1. Collecting a rollout (trajectory of transitions)
    2. Running a PPO update
    3. Periodically evaluating performance

    Args:
        trainer: AugmentedTrainer or BaselineTrainer
        cfg: experiment config
        task_name: which task to train on
        num_episodes: total training episodes (approximated via steps)
        model_name: for logging ("augmented" or "baseline")
        log_fn: optional callback for external logging

    Returns:
        Dict with training curves and final performance.
    """
    is_augmented = isinstance(trainer, AugmentedTrainer)

    history = defaultdict(list)
    eval_history = []
    total_episodes_approx = 0
    rollout_count = 0

    # Approximate episodes per rollout
    eps_per_rollout = cfg.train.num_parallel_envs * cfg.train.rollout_steps / cfg.env.max_steps_per_episode

    print(f"\n{'─' * 50}")
    print(f"Training {model_name} on {task_name}")
    print(f"Target: ~{num_episodes} episodes ({num_episodes / eps_per_rollout:.0f} rollouts)")
    print(f"{'─' * 50}")

    start_time = time.time()

    # Early stopping state: track peak performance and stop if it degrades
    best_eval_success = 0.0
    evals_since_best = 0
    best_model_state = None
    early_stopped = False

    while total_episodes_approx < num_episodes:
        # ── Step 1: Collect rollout ──
        episode_stats = trainer.collect_rollout()
        rollout_count += 1
        total_episodes_approx += eps_per_rollout

        # ── Step 2: PPO update ──
        update_metrics = trainer.update()

        # Track metrics
        for k, v in update_metrics.items():
            history[k].append(v)

        if episode_stats.get("success"):
            recent_success_rate = np.mean(episode_stats["success"][-20:])
            history["success_rate"].append(recent_success_rate)
        else:
            history["success_rate"].append(0.0)

        # ── Step 3: Periodic evaluation ──
        if rollout_count % max(1, int(cfg.train.eval_interval / eps_per_rollout)) == 0:
            eval_result = evaluate_model(
                trainer.model, cfg, task_name,
                num_episodes=min(20, cfg.train.num_eval_episodes),
                is_augmented=is_augmented,
            )
            eval_history.append({
                "episode": total_episodes_approx,
                **eval_result,
            })

            elapsed = time.time() - start_time
            eps_per_sec = total_episodes_approx / elapsed if elapsed > 0 else 0

            status = (
                f"  [{model_name}] ep≈{total_episodes_approx:.0f} | "
                f"eval_success={eval_result['success_rate']:.3f} | "
                f"avg_len={eval_result['avg_episode_length']:.1f} | "
                f"policy_loss={update_metrics.get('policy_loss', 0):.4f}"
            )
            if is_augmented:
                status += f" | obj_ent={update_metrics.get('obj_entropy', 0):.3f}"
            status += f" | {eps_per_sec:.1f} ep/s"

            print(status)

            # ── Early stopping: save best model, stop if performance degrades ──
            current_success = eval_result['success_rate']
            if current_success > best_eval_success:
                best_eval_success = current_success
                evals_since_best = 0
                # Save best model weights
                import copy
                best_model_state = copy.deepcopy(trainer.model.state_dict())
            else:
                evals_since_best += 1

            # Stop if we had good performance (>0.5) and it's been declining for 4 evals
            if best_eval_success >= 0.5 and evals_since_best >= 4:
                print(f"  [early stopping: best={best_eval_success:.3f}, "
                      f"current={current_success:.3f}, restoring best weights]")
                trainer.model.load_state_dict(best_model_state)
                early_stopped = True
                break

    # Final evaluation
    final_eval = evaluate_model(
        trainer.model, cfg, task_name,
        num_episodes=cfg.train.num_eval_episodes,
        is_augmented=is_augmented,
    )

    elapsed = time.time() - start_time
    print(f"\n  ✓ {model_name} on {task_name}: "
          f"success_rate={final_eval['success_rate']:.3f} "
          f"({elapsed:.1f}s)")

    return {
        "history": dict(history),
        "eval_history": eval_history,
        "final": final_eval,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_comparison(
    aug_results: dict,
    base_results: dict,
    task_name: str,
    save_dir: str,
):
    """Generate comparison plots for a single task."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Training Comparison: {task_name}", fontsize=14, fontweight="bold")

    # ── Plot 1: Success rate over training ──
    ax = axes[0]
    if aug_results["eval_history"]:
        eps = [e["episode"] for e in aug_results["eval_history"]]
        success = [e["success_rate"] for e in aug_results["eval_history"]]
        ax.plot(eps, success, "b-o", label="Augmented (PFC)", markersize=3)
    if base_results["eval_history"]:
        eps = [e["episode"] for e in base_results["eval_history"]]
        success = [e["success_rate"] for e in base_results["eval_history"]]
        ax.plot(eps, success, "r-s", label="Baseline (Fixed)", markersize=3)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate During Training")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Policy loss ──
    ax = axes[1]
    if "policy_loss" in aug_results["history"]:
        ax.plot(aug_results["history"]["policy_loss"], "b-", alpha=0.7, label="Augmented")
    if "policy_loss" in base_results["history"]:
        ax.plot(base_results["history"]["policy_loss"], "r-", alpha=0.7, label="Baseline")
    ax.set_xlabel("PPO Updates")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Episode length ──
    ax = axes[2]
    if aug_results["eval_history"]:
        eps = [e["episode"] for e in aug_results["eval_history"]]
        lengths = [e["avg_episode_length"] for e in aug_results["eval_history"]]
        ax.plot(eps, lengths, "b-o", label="Augmented", markersize=3)
    if base_results["eval_history"]:
        eps = [e["episode"] for e in base_results["eval_history"]]
        lengths = [e["avg_episode_length"] for e in base_results["eval_history"]]
        ax.plot(eps, lengths, "r-s", label="Baseline", markersize=3)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Episode Length")
    ax.set_title("Episode Length (lower = more efficient)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"comparison_{task_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {path}")


def plot_strategy_distribution(
    diversity_results: dict,
    save_dir: str,
):
    """Plot sub-objective usage across tasks for the augmented model."""
    os.makedirs(save_dir, exist_ok=True)

    tasks = [t for t in diversity_results.keys() if isinstance(diversity_results[t], dict)]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        dist = diversity_results[task].get("strategy_distribution", {})
        if dist:
            names = list(dist.keys())
            values = list(dist.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            ax.bar(names, values, color=colors)
            ax.set_title(f"{task}\n(entropy={diversity_results[task].get('strategy_entropy', 0):.2f})")
            ax.set_ylabel("Selection Frequency")
            ax.set_ylim(0, 1)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("Meta-Controller Strategy Distribution per Task", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "strategy_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {path}")


def plot_final_summary(
    all_results: dict,
    save_dir: str,
):
    """Bar chart comparing final success rates across all tasks."""
    os.makedirs(save_dir, exist_ok=True)

    tasks = list(all_results.keys())
    aug_scores = [all_results[t]["augmented"]["final"]["success_rate"] for t in tasks]
    base_scores = [all_results[t]["baseline"]["final"]["success_rate"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, aug_scores, width, label="Augmented (PFC)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, base_scores, width, label="Baseline (Fixed)", color="#DD8452")

    ax.set_xlabel("Task")
    ax.set_ylabel("Success Rate")
    ax.set_title("Final Success Rate Comparison Across Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "final_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT MODES
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test(cfg: ExperimentConfig):
    """Quick sanity check that everything runs."""
    print("\n🔬 SMOKE TEST — Verifying experiment pipeline\n")
    cfg.train.total_episodes = 100
    cfg.train.eval_interval = 50
    cfg.train.num_eval_episodes = 10
    cfg.train.num_parallel_envs = 4
    cfg.tasks = ["morris_water_maze"]

    print_model_comparison(cfg.model, cfg.env)

    task = cfg.tasks[0]
    aug_trainer = AugmentedTrainer(cfg, task)
    base_trainer = BaselineTrainer(cfg, task)

    aug_results = train_on_task(aug_trainer, cfg, task, cfg.train.total_episodes, "augmented")
    base_results = train_on_task(base_trainer, cfg, task, cfg.train.total_episodes, "baseline")

    print("\n✓ Smoke test passed! Both models train without errors.")
    print(f"  Augmented final success: {aug_results['final']['success_rate']:.3f}")
    print(f"  Baseline final success:  {base_results['final']['success_rate']:.3f}")


def run_single_task(cfg: ExperimentConfig, task_name: str):
    """Train and compare on a single task."""
    print(f"\n🔬 SINGLE TASK EXPERIMENT — {task_name}\n")
    print_model_comparison(cfg.model, cfg.env)

    aug_trainer = AugmentedTrainer(cfg, task_name)
    base_trainer = BaselineTrainer(cfg, task_name)

    aug_results = train_on_task(aug_trainer, cfg, task_name, cfg.train.total_episodes, "augmented")
    base_results = train_on_task(base_trainer, cfg, task_name, cfg.train.total_episodes, "baseline")

    plot_training_comparison(aug_results, base_results, task_name, cfg.results_dir)

    # Strategy analysis for augmented model
    diversity = eval_strategy_diversity(aug_trainer.model, cfg)
    plot_strategy_distribution(diversity, cfg.results_dir)

    return {task_name: {"augmented": aug_results, "baseline": base_results}}


def run_full_experiment(cfg: ExperimentConfig):
    """
    Complete experiment: train both models on all tasks, then evaluate.

    Training strategy:
    - Augmented model: Trained SEQUENTIALLY on all tasks (testing if the
      strategy library persists across tasks).
    - Baseline models: One model per task, each trained independently
      (this is the best-case scenario for the baseline).
    """
    print("\n🔬 FULL EXPERIMENT — All tasks, both models\n")
    print_model_comparison(cfg.model, cfg.env)

    # ── Train augmented model on all tasks sequentially ──
    augmented_model = AugmentedModel(cfg.model, cfg.env).to(cfg.device)
    performance_history = {}
    all_results = {}

    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)

    # Interleaved training from scratch with shared optimizer.
    # No MWM pre-training — all tasks get equal footing so the encoder
    # develops balanced features for all tasks, not MWM-biased features.
    print("\n  Interleaved training from scratch (shared optimizer)...")
    import torch.optim as optim
    shared_optimizer = optim.Adam([
        {"params": augmented_model.encoder.parameters(), "lr": cfg.train.lr_policy},
        {"params": augmented_model.policy_head.parameters(), "lr": cfg.train.lr_policy},
        {"params": augmented_model.value_head.parameters(), "lr": cfg.train.lr_policy},
        {"params": augmented_model.obj_embeddings.parameters(), "lr": cfg.train.lr_policy},
        {"params": augmented_model.meta_controller.parameters(), "lr": cfg.train.lr_meta},
        {"params": augmented_model.meta_value_head.parameters(), "lr": cfg.train.lr_meta},
    ])
    aug_trainers = {}
    for task in cfg.tasks:
        trainer = AugmentedTrainer(cfg, task, existing_model=augmented_model)
        trainer.optimizer = shared_optimizer  # Replace per-trainer optimizer
        aug_trainers[task] = trainer

    eps_per_rollout = cfg.train.num_parallel_envs * cfg.train.rollout_steps / cfg.env.max_steps_per_episode
    total_interleaved_eps = int(cfg.train.total_episodes * len(cfg.tasks) * 2.0)  # 2x budget for harder two-level optimization
    task_episode_counts = {t: 0 for t in cfg.tasks}
    task_eval_history = {t: [] for t in cfg.tasks}
    task_history = {t: defaultdict(list) for t in cfg.tasks}
    rollout_idx = 0

    # Multi-task early stopping: save best model across all tasks combined
    import copy
    best_multitask_score = 0.0
    best_model_state = copy.deepcopy(augmented_model.state_dict())
    evals_since_best = 0

    start_time = time.time()
    while sum(task_episode_counts.values()) < total_interleaved_eps:
        # Round-robin through tasks
        task = cfg.tasks[rollout_idx % len(cfg.tasks)]
        trainer = aug_trainers[task]

        episode_stats = trainer.collect_rollout()
        update_metrics = trainer.update()
        task_episode_counts[task] += eps_per_rollout

        for k, v in update_metrics.items():
            task_history[task][k].append(v)

        # Periodic evaluation (every ~1000 episodes total)
        if rollout_idx % (len(cfg.tasks) * max(1, int(250 / eps_per_rollout))) == 0 and rollout_idx > 0:
            for eval_task in cfg.tasks:
                eval_result = evaluate_model(
                    augmented_model, cfg, eval_task,
                    num_episodes=30,
                    is_augmented=True,
                )
                task_eval_history[eval_task].append({
                    "episode": task_episode_counts[eval_task],
                    **eval_result,
                })
            elapsed = time.time() - start_time
            total_eps = sum(task_episode_counts.values())
            successes = {t: task_eval_history[t][-1]["success_rate"] if task_eval_history[t] else 0
                        for t in cfg.tasks}
            # Breadth-weighted score: count tasks above 25%, plus uncapped sum
            tasks_above = sum(1 for s in successes.values() if s >= 0.25)
            raw_sum = sum(successes.values())  # No cap — keep improving
            multitask_score = tasks_above * 2.0 + raw_sum  # Prioritize breadth, then depth
            obj_ent = update_metrics.get('obj_entropy', 0)
            print(f"  [interleaved] total_eps≈{total_eps:.0f} | "
                  + " | ".join(f"{t[:4]}={successes[t]:.2f}" for t in cfg.tasks)
                  + f" | above25={tasks_above} | score={multitask_score:.2f}"
                  + f" | obj_ent={obj_ent:.3f} | {total_eps/elapsed:.1f} ep/s")

            # Multi-task early stopping (breadth-weighted)
            if multitask_score > best_multitask_score:
                best_multitask_score = multitask_score
                best_model_state = copy.deepcopy(augmented_model.state_dict())
                evals_since_best = 0
            else:
                evals_since_best += 1

            # Stop if peaked and declining for 10 evals, AND we've trained enough
            if best_multitask_score >= 10.0 and evals_since_best >= 10 and total_eps > total_interleaved_eps * 0.7:
                print(f"  [multi-task early stop: best_score={best_multitask_score:.2f}, "
                      f"current={multitask_score:.2f}, restoring best]")
                augmented_model.load_state_dict(best_model_state)
                break

        rollout_idx += 1

    # Restore best model if we didn't early stop
    if evals_since_best > 0:
        print(f"  [restoring best model: sum={best_multitask_score:.2f}]")
        augmented_model.load_state_dict(best_model_state)

    # Final evaluation per task
    for task in cfg.tasks:
        final_eval = evaluate_model(
            augmented_model, cfg, task,
            num_episodes=cfg.train.num_eval_episodes,
            is_augmented=True,
        )
        print(f"  ✓ augmented on {task}: success_rate={final_eval['success_rate']:.3f}")
        all_results.setdefault(task, {})["augmented"] = {
            "history": dict(task_history[task]),
            "eval_history": task_eval_history[task],
            "final": final_eval,
        }
        performance_history[task] = final_eval["success_rate"]

    # ── Train baseline models (one per task) ──
    baseline_models = {}
    for task in cfg.tasks:
        base_trainer = BaselineTrainer(cfg, task)
        base_results = train_on_task(
            base_trainer, cfg, task, cfg.train.total_episodes, f"baseline"
        )
        baseline_models[task] = base_trainer.model
        all_results[task]["baseline"] = base_results

    # ── Generate training comparison plots ──
    print("\n" + "=" * 70)
    print("PHASE 2: GENERATING PLOTS")
    print("=" * 70)

    for task in cfg.tasks:
        plot_training_comparison(
            all_results[task]["augmented"],
            all_results[task]["baseline"],
            task,
            cfg.results_dir,
        )

    plot_final_summary(all_results, cfg.results_dir)

    # ── Run full evaluation suite ──
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION")
    print("=" * 70)

    report = run_full_evaluation(
        augmented_model, baseline_models, cfg, performance_history
    )

    # Strategy distribution plot
    plot_strategy_distribution(report["strategy_diversity"], cfg.results_dir)

    # Save report
    report_path = os.path.join(cfg.results_dir, "evaluation_report.json")
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(report_path, "w") as f:
        json.dump(convert_for_json(report), f, indent=2, default=str)
    print(f"\n  Saved evaluation report: {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Self-Optimizing Training Paradigm Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --mode smoke_test
  python run_experiment.py --mode single_task --task morris_water_maze --episodes 2000
  python run_experiment.py --mode full --episodes 5000 --device cuda
        """,
    )
    parser.add_argument("--mode", choices=["smoke_test", "single_task", "full"],
                        default="smoke_test", help="Experiment mode")
    parser.add_argument("--task", type=str, default="morris_water_maze",
                        choices=["morris_water_maze", "visual_foraging",
                                 "dynamic_obstacles", "visual_search"],
                        help="Task for single_task mode")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Training episodes per task")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    # ── Configure ──
    cfg = ExperimentConfig()
    cfg.seed = args.seed
    cfg.train.total_episodes = args.episodes
    cfg.results_dir = args.results_dir

    if args.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = args.device

    print(f"Device: {cfg.device}")
    if cfg.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    set_seed(cfg.seed)

    # ── Run ──
    if args.mode == "smoke_test":
        run_smoke_test(cfg)
    elif args.mode == "single_task":
        run_single_task(cfg, args.task)
    elif args.mode == "full":
        run_full_experiment(cfg)

    print("\n✓ Experiment complete. Results saved to:", cfg.results_dir)


if __name__ == "__main__":
    main()
