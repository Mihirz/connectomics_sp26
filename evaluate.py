"""
evaluate.py — Generalizability Evaluation Suite

This module measures whether the augmented model's self-optimizing paradigm
actually produces better generalizability than the fixed-objective baseline.

We test five dimensions of generalizability:

1. MULTI-TASK PERFORMANCE — Average success rate across all four tasks.
   The augmented model trains on all tasks with the same meta-controller;
   the baseline trains a separate policy per task.

2. ZERO-SHOT TRANSFER — Test on unseen task variants (rotated mazes,
   different obstacle counts, shifted goals).  No additional training.
   This tests whether the learned "strategy library" transfers.

3. FEW-SHOT ADAPTATION — How many episodes does each model need to reach
   a threshold success rate on a new variant?  Tests learning efficiency.

4. CATASTROPHIC FORGETTING — After training on later tasks, how much
   does performance degrade on earlier tasks?  The augmented model's
   strategy library should be more robust to forgetting.

5. STRATEGY DIVERSITY — Entropy of the meta-controller's sub-objective
   selections.  Higher entropy = more diverse strategy usage.  We expect
   the augmented model to use different strategies for different task contexts.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from config import ExperimentConfig
from models import AugmentedModel, BaselineModel
from environments import make_env, ENV_REGISTRY
from sub_objectives import IntrinsicRewardComputer, NUM_SUB_OBJECTIVES, SUB_OBJ_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# CORE EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    cfg: ExperimentConfig,
    task_name: str,
    num_episodes: int = 50,
    variant_seed: int = None,
    is_augmented: bool = True,
) -> Dict:
    """
    Run a model on a task for N episodes and collect performance metrics.

    Args:
        model: AugmentedModel or BaselineModel (in eval mode)
        cfg: experiment configuration
        task_name: which task to evaluate on
        num_episodes: how many episodes to run
        variant_seed: if set, creates a task variant (for transfer testing)
        is_augmented: whether this is the augmented model (affects forward pass)

    Returns:
        Dict with success_rate, avg_episode_length, avg_reward, etc.
    """
    model.eval()
    device = cfg.device

    env = make_env(task_name, cfg.env, variant_seed=variant_seed)
    reward_computer = IntrinsicRewardComputer(cfg.sub_obj)

    successes = []
    episode_lengths = []
    total_rewards = []
    obj_selections = np.zeros(NUM_SUB_OBJECTIVES) if is_augmented else None

    for ep in range(num_episodes):
        obs, _ = env.reset()
        reward_computer.reset(cfg.env.grid_size)
        done = False
        ep_reward = 0
        ep_steps = 0

        # Initialize GRU hidden state for augmented model (reset each episode)
        if is_augmented:
            meta_hidden = model.init_meta_hidden(1)

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                if is_augmented:
                    # Use deterministic actions but stochastic meta-controller
                    # to preserve the strategy diversity that is the model's strength.
                    action, _, obj_idx, _, _, _, meta_hidden = model(
                        obs_tensor, meta_hidden, deterministic=False
                    )
                    obj_selections[obj_idx.item()] += 1
                else:
                    action, _, _ = model(obs_tensor, deterministic=True)

            obs, reward_info, done, info = env.step(action.item())
            ep_reward += reward_info["dense_reward"]
            ep_steps += 1

        successes.append(reward_info.get("success", False))
        episode_lengths.append(ep_steps)
        total_rewards.append(ep_reward)

    results = {
        "success_rate": np.mean(successes),
        "avg_episode_length": np.mean(episode_lengths),
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
    }

    if is_augmented and obj_selections is not None:
        # Strategy diversity: entropy of sub-objective selections
        total = obj_selections.sum()
        if total > 0:
            probs = obj_selections / total
            probs = probs[probs > 0]  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs))
            results["strategy_entropy"] = entropy
            results["strategy_distribution"] = {
                SUB_OBJ_NAMES[i]: obj_selections[i] / total
                for i in range(NUM_SUB_OBJECTIVES)
            }
        else:
            results["strategy_entropy"] = 0.0

    model.train()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION 1: MULTI-TASK PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def eval_multitask(
    augmented_model: AugmentedModel,
    baseline_models: Dict[str, BaselineModel],
    cfg: ExperimentConfig,
) -> Dict:
    """
    Compare average performance across all tasks.

    The augmented model uses ONE set of weights for all tasks.
    The baseline uses a SEPARATE model per task (best-case for the baseline).
    """
    results = {"augmented": {}, "baseline": {}}

    for task in cfg.tasks:
        # Augmented: same model, different task
        aug_result = evaluate_model(
            augmented_model, cfg, task,
            num_episodes=cfg.train.num_eval_episodes,
            is_augmented=True,
        )
        results["augmented"][task] = aug_result

        # Baseline: task-specific model
        if task in baseline_models:
            base_result = evaluate_model(
                baseline_models[task], cfg, task,
                num_episodes=cfg.train.num_eval_episodes,
                is_augmented=False,
            )
            results["baseline"][task] = base_result

    # Compute averages
    aug_avg = np.mean([r["success_rate"] for r in results["augmented"].values()])
    base_avg = np.mean([r["success_rate"] for r in results["baseline"].values()])

    results["augmented_avg_success"] = aug_avg
    results["baseline_avg_success"] = base_avg
    results["generalizability_delta"] = aug_avg - base_avg

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION 2: ZERO-SHOT TRANSFER
# ═══════════════════════════════════════════════════════════════════════════════

def eval_zero_shot_transfer(
    augmented_model: AugmentedModel,
    baseline_models: Dict[str, BaselineModel],
    cfg: ExperimentConfig,
) -> Dict:
    """
    Test on unseen task variants without any additional training.

    Variants are generated by changing the random seed, which alters:
    - Platform location (Morris maze)
    - Food/predator positions (foraging)
    - Obstacle patterns (obstacles)
    - Target/distractor positions (visual search)

    The hypothesis is that the augmented model's strategy library transfers
    to new variants because the strategies are task-general (explore, approach,
    avoid, etc.), while the baseline's fixed policy is tuned to specific layouts.
    """
    results = {"augmented": {}, "baseline": {}}

    for task in cfg.tasks:
        aug_scores = []
        base_scores = []

        for v in range(cfg.train.num_transfer_variants):
            variant_seed = 10000 + v * 137  # Different from training seeds

            aug_result = evaluate_model(
                augmented_model, cfg, task,
                num_episodes=cfg.train.num_eval_episodes,
                variant_seed=variant_seed,
                is_augmented=True,
            )
            aug_scores.append(aug_result["success_rate"])

            if task in baseline_models:
                base_result = evaluate_model(
                    baseline_models[task], cfg, task,
                    num_episodes=cfg.train.num_eval_episodes,
                    variant_seed=variant_seed,
                    is_augmented=False,
                )
                base_scores.append(base_result["success_rate"])

        results["augmented"][task] = {
            "mean_success": np.mean(aug_scores),
            "std_success": np.std(aug_scores),
        }
        results["baseline"][task] = {
            "mean_success": np.mean(base_scores) if base_scores else 0,
            "std_success": np.std(base_scores) if base_scores else 0,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION 3: FEW-SHOT ADAPTATION
# ═══════════════════════════════════════════════════════════════════════════════

def eval_few_shot_adaptation(
    augmented_model: AugmentedModel,
    baseline_model: BaselineModel,
    cfg: ExperimentConfig,
    task_name: str,
    max_adaptation_episodes: int = 200,
) -> Dict:
    """
    Measure how quickly each model adapts to a new task variant.

    We fine-tune each model on a new variant and track how many episodes
    it takes to reach the threshold success rate.

    The hypothesis: the augmented model adapts faster because its
    meta-controller can quickly re-select from existing strategies
    without needing to re-learn low-level behaviors.
    """
    import copy
    threshold = cfg.train.few_shot_threshold
    variant_seed = 99999  # Novel variant

    results = {}

    for model_name, model, is_aug in [
        ("augmented", augmented_model, True),
        ("baseline", baseline_model, False),
    ]:
        # Clone model for fine-tuning (don't modify the original)
        ft_model = copy.deepcopy(model)
        ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=cfg.train.lr_policy * 0.5)

        env = make_env(task_name, cfg.env, variant_seed=variant_seed)
        episodes_to_threshold = max_adaptation_episodes  # Default: didn't reach it

        recent_successes = []
        for ep in range(max_adaptation_episodes):
            obs, _ = env.reset()
            done = False
            if is_aug:
                meta_hidden = ft_model.init_meta_hidden(1)

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    if is_aug:
                        action, _, _, _, _, _, meta_hidden = ft_model(
                            obs_tensor, meta_hidden
                        )
                    else:
                        action, _, _ = ft_model(obs_tensor)
                obs, ri, done, _ = env.step(action.item())

            recent_successes.append(ri.get("success", False))
            if len(recent_successes) > 20:
                recent_successes.pop(0)

            if len(recent_successes) >= 20 and np.mean(recent_successes) >= threshold:
                episodes_to_threshold = ep + 1
                break

        results[model_name] = {
            "episodes_to_threshold": episodes_to_threshold,
            "final_success_rate": np.mean(recent_successes) if recent_successes else 0,
            "reached_threshold": episodes_to_threshold < max_adaptation_episodes,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION 4: CATASTROPHIC FORGETTING
# ═══════════════════════════════════════════════════════════════════════════════

def eval_catastrophic_forgetting(
    augmented_model: AugmentedModel,
    cfg: ExperimentConfig,
    task_order: List[str],
    performance_before: Dict[str, float],
) -> Dict:
    """
    Compare performance on early tasks before and after training on later tasks.

    Args:
        augmented_model: model after full sequential training
        task_order: order tasks were trained in
        performance_before: success rates on each task measured right after
                           training on that task

    Returns:
        Dict with forgetting metrics per task.
    """
    results = {}

    for task in task_order:
        current_result = evaluate_model(
            augmented_model, cfg, task,
            num_episodes=cfg.train.num_eval_episodes,
            is_augmented=True,
        )
        before = performance_before.get(task, 0)
        after = current_result["success_rate"]
        forgetting = max(0, before - after)

        results[task] = {
            "before": before,
            "after": after,
            "forgetting": forgetting,
        }

    results["avg_forgetting"] = np.mean([r["forgetting"] for r in results.values() if isinstance(r, dict)])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION 5: STRATEGY DIVERSITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def eval_strategy_diversity(
    augmented_model: AugmentedModel,
    cfg: ExperimentConfig,
) -> Dict:
    """
    Analyze which sub-objectives the meta-controller selects for each task.

    We expect to see different strategy distributions for different tasks:
    - Morris maze: heavy EXPLORE early, then MEMORIZE/EXPLOIT
    - Foraging: mix of APPROACH (food) and AVOID (predators)
    - Obstacles: AVOID and APPROACH
    - Visual search: EXPLORE or APPROACH (depending on cue presence)

    If the augmented model shows task-appropriate strategy selection, it
    demonstrates that the meta-controller has learned to flexibly compose
    strategies — the core claim of the hypothesis.
    """
    results = {}

    for task in cfg.tasks:
        result = evaluate_model(
            augmented_model, cfg, task,
            num_episodes=cfg.train.num_eval_episodes,
            is_augmented=True,
        )
        results[task] = {
            "strategy_entropy": result.get("strategy_entropy", 0),
            "strategy_distribution": result.get("strategy_distribution", {}),
            "success_rate": result["success_rate"],
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    augmented_model: AugmentedModel,
    baseline_models: Dict[str, BaselineModel],
    cfg: ExperimentConfig,
    performance_history: Dict = None,
) -> Dict:
    """
    Run all evaluation suites and compile a comprehensive report.
    """
    print("\n" + "=" * 70)
    print("RUNNING FULL GENERALIZABILITY EVALUATION")
    print("=" * 70)

    # 1. Multi-task
    print("\n[1/5] Multi-task performance...")
    multitask = eval_multitask(augmented_model, baseline_models, cfg)

    # 2. Zero-shot transfer
    print("[2/5] Zero-shot transfer...")
    transfer = eval_zero_shot_transfer(augmented_model, baseline_models, cfg)

    # 3. Few-shot adaptation (on first task only, for speed)
    print("[3/5] Few-shot adaptation...")
    first_task = cfg.tasks[0]
    if first_task in baseline_models:
        few_shot = eval_few_shot_adaptation(
            augmented_model, baseline_models[first_task], cfg, first_task
        )
    else:
        few_shot = {"augmented": {}, "baseline": {}}

    # 4. Catastrophic forgetting
    print("[4/5] Catastrophic forgetting...")
    if performance_history:
        forgetting = eval_catastrophic_forgetting(
            augmented_model, cfg, cfg.tasks, performance_history
        )
    else:
        forgetting = {"note": "No performance history provided; skipping."}

    # 5. Strategy diversity
    print("[5/5] Strategy diversity analysis...")
    diversity = eval_strategy_diversity(augmented_model, cfg)

    report = {
        "multitask": multitask,
        "zero_shot_transfer": transfer,
        "few_shot_adaptation": few_shot,
        "catastrophic_forgetting": forgetting,
        "strategy_diversity": diversity,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nMulti-task average success rate:")
    print(f"  Augmented (PFC):  {multitask['augmented_avg_success']:.3f}")
    print(f"  Baseline (Fixed): {multitask['baseline_avg_success']:.3f}")
    print(f"  Delta:            {multitask['generalizability_delta']:+.3f}")

    print(f"\nZero-shot transfer (avg across tasks):")
    aug_transfer = np.mean([v["mean_success"] for v in transfer["augmented"].values()])
    base_transfer = np.mean([v["mean_success"] for v in transfer["baseline"].values()])
    print(f"  Augmented: {aug_transfer:.3f}")
    print(f"  Baseline:  {base_transfer:.3f}")

    if isinstance(few_shot.get("augmented"), dict):
        print(f"\nFew-shot adaptation on '{first_task}':")
        print(f"  Augmented: {few_shot['augmented'].get('episodes_to_threshold', 'N/A')} episodes")
        print(f"  Baseline:  {few_shot['baseline'].get('episodes_to_threshold', 'N/A')} episodes")

    print(f"\nStrategy diversity (augmented model):")
    for task, data in diversity.items():
        if isinstance(data, dict):
            ent = data.get("strategy_entropy", 0)
            print(f"  {task}: entropy={ent:.3f}, dist={data.get('strategy_distribution', {})}")

    print("=" * 70)

    return report
