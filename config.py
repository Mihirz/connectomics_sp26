"""
config.py — Central configuration for the Self-Optimizing Training Paradigm Experiment.

All hyperparameters live here so that the experiment is easy to tune and reproduce.
We keep everything small enough to run on a single GPU with ≤ 8 GB VRAM.
"""

from dataclasses import dataclass, field
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# We use small 2D grid-worlds rendered as 3-channel images.  32×32 is large
# enough for interesting navigation while keeping memory tiny.

@dataclass
class EnvConfig:
    grid_size: int = 20            # NxN grid world (20 keeps exploration tractable)
    obs_channels: int = 3          # RGB observation
    max_steps_per_episode: int = 300  # Episode timeout (negative signal trigger)
    num_actions: int = 5           # up, down, left, right, stay

    # ── Morris Water Maze ──
    # The agent is placed at a random edge of a circular "pool" and must find
    # a hidden platform.  Landmark cues on the pool walls provide spatial
    # reference (matching the real experiment protocol).
    mwm_pool_radius: float = 0.45  # Fraction of grid_size
    mwm_platform_radius: float = 0.15  # Larger platform for learnable signal

    # ── Visual Foraging ──
    # Collect food items while avoiding moving predator zones.
    forage_num_food: int = 8
    forage_num_predators: int = 3
    forage_predator_speed: float = 0.5   # Cells per step (fractional = stochastic)
    forage_food_collect_target: int = 6  # Must collect this many to "succeed"

    # ── Dynamic Obstacle Course ──
    # Navigate from start to goal through moving obstacles.
    obstacle_num_obstacles: int = 10
    obstacle_speed: float = 0.7

    # ── Visual Search with Cues ──
    # Find a hidden target; partial cues (arrows/gradients) may be present.
    search_num_distractors: int = 12
    search_cue_probability: float = 0.5  # Probability that a helpful cue appears


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Both models share the same CNN encoder so that the *only* architectural
# difference is the meta-controller (present in the augmented model, absent
# in the baseline).

@dataclass
class ModelConfig:
    # ── Shared CNN encoder ──
    # 4 conv layers: 3→16→32→64→64, each with stride 2 and ReLU.
    # Input: (3, 32, 32) → Output: (64, 2, 2) = 256-dim flattened.
    encoder_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 64])
    latent_dim: int = 256          # Flattened encoder output
    hidden_dim: int = 128          # Hidden layer width for policy / value heads

    # ── Meta-controller (PFC analogue) — augmented model only ──
    # Outputs a categorical distribution over K sub-objectives.
    num_sub_objectives: int = 5    # EXPLORE, APPROACH, AVOID, EXPLOIT, MEMORIZE
    meta_hidden_dim: int = 64      # Smaller network; this is a "selector", not a policy
    objective_embed_dim: int = 16  # Embedding size per sub-objective

    # ── Memory module (augmented model) ──
    # A small spatial memory buffer the model can write to and read from.
    # This supports the MEMORIZE sub-objective.
    memory_size: int = 64          # Number of memory slots


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # ── PPO hyperparameters ──
    # We use Proximal Policy Optimization for both models because it is stable,
    # sample-efficient, and well-understood.
    lr_policy: float = 3e-4             # Learning rate for action policy
    lr_meta: float = 3e-4              # Learning rate for meta-controller
    gamma: float = 0.99                 # Discount factor
    gae_lambda: float = 0.95           # GAE parameter
    clip_epsilon: float = 0.2          # PPO clipping
    entropy_coef: float = 0.01         # Entropy bonus for action policy
    meta_entropy_coef: float = 0.05    # Higher entropy for meta-controller (encourage exploration of strategies)
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Meta-controller temporal commitment ──
    # The meta-controller selects a sub-objective every N steps, not every step.
    # Between selections, the same sub-objective is held constant.  This gives
    # the action policy time to execute a coherent strategy and reduces the
    # meta-controller's credit assignment from ~300 decisions to ~20 per episode.
    # Biologically: PFC executive control operates on seconds, not milliseconds.
    meta_decision_interval: int = 15   # Steps between meta-controller decisions

    # ── Rollout settings ──
    # These are deliberately small to fit in ≤ 8 GB VRAM.
    num_parallel_envs: int = 16        # Vectorized environments
    rollout_steps: int = 128           # Steps per rollout before PPO update
    ppo_epochs: int = 4                # Passes over each rollout
    mini_batch_size: int = 64          # Mini-batch size within PPO epoch

    # ── Experiment schedule ──
    total_episodes: int = 5000         # Per task (can override via CLI)
    eval_interval: int = 250           # Evaluate every N episodes
    num_eval_episodes: int = 50        # Episodes per evaluation

    # ── Reward design (Design B) ──
    # Both models receive the same dense task reward.  The augmented model
    # additionally gets intrinsic reward from the meta-controller's selected
    # sub-objective.  This mirrors the biology: the PFC supplements standard
    # reward processing with top-down strategy modulation.
    #
    # The meta-controller receives ONLY sparse negative punishment + intrinsic
    # feedback (no dense reward).  It must learn which sub-objectives are
    # productive purely from whether they avoid failure and generate useful
    # intrinsic signals.
    failure_penalty: float = -1.0      # Given on episode timeout or catastrophic failure
    success_signal: float = 0.0        # No positive sparse signal (meta-controller)

    # ── Meta-controller intrinsic feedback ──
    # The meta-controller also receives a fraction of the selected sub-objective's
    # intrinsic reward as a per-step signal.  This is biologically plausible:
    # the PFC receives dopaminergic feedback about whether its currently selected
    # drive is "producing results," not just end-of-episode punishment.
    # Without this, credit assignment over 300 timesteps is nearly impossible.
    meta_intrinsic_feedback: float = 0.2  # Fraction of intrinsic reward fed to meta

    # ── Intrinsic reward scaling ──
    # Intrinsic rewards must be comparable in magnitude to dense rewards
    # (~0.01/step) to supplement rather than dominate the task signal.
    # Without scaling, intrinsic rewards are ~10x larger than dense,
    # drowning out the task gradient.
    intrinsic_reward_scale: float = 0.1  # Scale intrinsic rewards down by 10x

    # ── Dense reward (baseline model) ──
    # Standard dense reward shaping for comparison.
    baseline_step_penalty: float = -0.01   # Small cost per step (encourages efficiency)
    baseline_goal_reward: float = 1.0      # Reward for reaching the goal
    baseline_collision_penalty: float = -0.5

    # ── Generalizability evaluation ──
    num_transfer_variants: int = 5     # Number of unseen task variants for transfer eval
    few_shot_threshold: float = 0.7    # Success rate threshold for few-shot adaptation


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-OBJECTIVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubObjectiveConfig:
    # Intrinsic reward magnitudes for each sub-objective.
    # These are the "self-generated" reward signals the augmented model uses.
    explore_reward: float = 0.1        # Reward for visiting a new cell
    approach_reward: float = 0.15      # Reward for decreasing distance to salient object
    avoid_reward: float = 0.15         # Reward for increasing distance from threat
    exploit_reward: float = 0.2        # Reward for repeating a previously rewarded action
    memorize_reward: float = 0.1       # Reward for correctly using spatial memory

    # Decay factors
    explore_novelty_decay: float = 0.99  # How quickly visited cells become "boring"


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sub_obj: SubObjectiveConfig = field(default_factory=SubObjectiveConfig)

    seed: int = 42
    device: str = "cpu"   # Override with "cuda" on GPU machines
    log_dir: str = "runs"
    results_dir: str = "results"

    # Which tasks to include in the experiment
    tasks: List[str] = field(default_factory=lambda: [
        "morris_water_maze",
        "visual_foraging",
        "dynamic_obstacles",
        "visual_search",
    ])
