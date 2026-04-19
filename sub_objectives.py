"""
sub_objectives.py — Sub-Objective Library & Intrinsic Reward Computation

This is the heart of the "self-optimizing" paradigm.  The augmented model's
meta-controller selects from a library of sub-objectives at each time step.
Each sub-objective defines its own intrinsic reward function — a self-generated
signal the model uses to learn, independent of any extrinsic task reward.

Biological analogy:
    The prefrontal cortex doesn't get a single "do this" signal.  Instead, it
    selects among competing drives — curiosity, fear, hunger, social reward —
    and the selected drive shapes behavior in the short term.  We replicate
    this with explicit sub-objective modules.

The key experimental property:
    The ACTION POLICY learns from intrinsic rewards (selected by the meta-controller).
    The META-CONTROLLER learns from sparse negative extrinsic signals only.
    This creates a two-level optimization where the meta-controller must discover
    which sub-objectives, in which contexts, lead to avoiding task failure.

Strategy Library Concept:
    Over training, the model develops a "library" of strategies — learned policies
    conditioned on each sub-objective.  When presented with a new task, the
    meta-controller can compose these strategies without relearning from scratch.
    This is analogous to how the PFC can flexibly recombine learned skills.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from config import SubObjectiveConfig


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-OBJECTIVE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Integer IDs for each sub-objective (used as indices into the meta-controller's
# output distribution).
# Reduced from 5 to 3: removed AVOID (consistently <5% usage, only fires in
# threat tasks) and MEMORIZE (0-18% usage, patrol mode too niche).  Fewer
# sub-objectives means each mode gets 1/3 of training data instead of 1/5,
# and meta-controller credit assignment is simpler (3 choices vs 5).
EXPLORE  = 0   # Curiosity-driven exploration of novel states
APPROACH = 1   # Goal-directed movement toward salient targets
EXPLOIT  = 2   # Stay near goal / repeat previously successful patterns

SUB_OBJ_NAMES = ["EXPLORE", "APPROACH", "EXPLOIT"]
NUM_SUB_OBJECTIVES = len(SUB_OBJ_NAMES)


class IntrinsicRewardComputer:
    """
    Computes intrinsic rewards for each sub-objective given environment info.

    This is NOT a neural network — it's a hand-designed reward function library.
    The design choice is deliberate: we want the sub-objectives to have clear,
    interpretable semantics so we can analyze which strategies the meta-controller
    selects in different contexts.

    In a more advanced version, these could be learned reward models.

    Each method returns a scalar intrinsic reward given the step's reward_info dict.
    """

    def __init__(self, cfg: SubObjectiveConfig):
        self.cfg = cfg
        # Track state for stateful sub-objectives
        self.reward_history = {}        # (r,c) → cumulative dense reward at that location
        self.spatial_memory = None      # Will be initialized on first call
        self.grid_size = 0

    def reset(self, grid_size: int):
        """Reset per-episode state for stateful sub-objectives."""
        self.reward_history = {}
        self.grid_size = grid_size
        self.spatial_memory = np.zeros((grid_size, grid_size), dtype=np.float32)

    def compute_all(self, reward_info: Dict) -> np.ndarray:
        """
        Compute intrinsic rewards for ALL sub-objectives.

        Returns: shape (NUM_SUB_OBJECTIVES,) array of intrinsic rewards.
        The meta-controller's selected sub-objective determines which one
        is actually used to train the action policy.
        """
        return np.array([
            self._explore_reward(reward_info),
            self._approach_reward(reward_info),
            self._exploit_reward(reward_info),
        ], dtype=np.float32)

    # ── EXPLORE ──────────────────────────────────────────────────────────────
    # Biological analogue: dopaminergic novelty signal.
    # Rewards visiting cells the agent hasn't been to before.
    # This naturally drives the agent to cover the environment, which is
    # essential for tasks like the Morris water maze where the goal is hidden.

    def _explore_reward(self, info: Dict) -> float:
        if info["is_novel_cell"]:
            return self.cfg.explore_reward
        else:
            # Diminishing returns for revisiting — novelty decays
            visit_count = info["visit_count"]
            return self.cfg.explore_reward * (self.cfg.explore_novelty_decay ** visit_count) * 0.1

    # ── APPROACH ─────────────────────────────────────────────────────────────
    # Biological analogue: goal-directed approach behavior (mesolimbic pathway).
    # Rewards decreasing distance to the nearest salient object (goal/food).
    # Only useful when the agent has some idea where the target is.

    def _approach_reward(self, info: Dict) -> float:
        delta = info.get("approach_delta", 0)
        if delta > 0:
            return self.cfg.approach_reward * min(delta, 1.0)
        else:
            return -self.cfg.approach_reward * 0.3 * min(abs(delta), 1.0)

    # ── EXPLOIT ──────────────────────────────────────────────────────────────
    # Biological analogue: habit formation via dorsal striatum.
    # "Stay and harvest" mode — rewards proximity to the goal/target.
    # Distinct from APPROACH (which rewards movement toward target):
    # EXPLOIT rewards BEING near the target, encouraging the agent to
    # remain and collect rather than keep moving.

    def _exploit_reward(self, info: Dict) -> float:
        dist = info.get("dist_to_goal", float("inf"))
        if dist <= 3.0:
            # Strong reward for being very close — "harvest this area"
            return self.cfg.exploit_reward * (3.0 - dist) / 3.0
        elif dist <= 6.0:
            # Mild reward for being in the neighborhood
            return self.cfg.exploit_reward * 0.15
        return 0.0

class SubObjectiveEmbedding(nn.Module):
    """
    Learnable embeddings for each sub-objective.

    These embeddings are concatenated with the latent state before being
    passed to the action policy.  This conditions the policy on which
    sub-objective is currently active — effectively giving the policy a
    different "mode" for each strategy.

    The embeddings are learned end-to-end, which means the model can develop
    rich internal representations of what each strategy "means" beyond the
    hand-designed intrinsic rewards.
    """

    def __init__(self, num_objectives: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_objectives, embed_dim)

    def forward(self, objective_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            objective_idx: (batch_size,) tensor of sub-objective indices
        Returns:
            (batch_size, embed_dim) tensor of sub-objective embeddings
        """
        return self.embeddings(objective_idx)
