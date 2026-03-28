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
EXPLORE  = 0   # Curiosity-driven exploration of novel states
APPROACH = 1   # Goal-directed movement toward salient targets
AVOID    = 2   # Threat avoidance
EXPLOIT  = 3   # Repeat previously successful patterns
MEMORIZE = 4   # Build and use spatial memory

SUB_OBJ_NAMES = ["EXPLORE", "APPROACH", "AVOID", "EXPLOIT", "MEMORIZE"]
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
        self.exploit_success_map = {}  # (r,c) → cumulative success at that location
        self.spatial_memory = None      # Will be initialized on first call

    def reset(self, grid_size: int):
        """Reset per-episode state for stateful sub-objectives."""
        self.exploit_success_map = {}
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
            self._avoid_reward(reward_info),
            self._exploit_reward(reward_info),
            self._memorize_reward(reward_info),
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

    # ── AVOID ────────────────────────────────────────────────────────────────
    # Biological analogue: amygdala-driven fear/avoidance.
    # Rewards increasing distance from the nearest threat.
    # Critical in foraging and obstacle tasks.

    def _avoid_reward(self, info: Dict) -> float:
        if not info.get("threat_nearby", False):
            return 0.0  # No threat → no avoidance reward needed

        threat_dist = info.get("threat_dist", float("inf"))
        if threat_dist < 3.0:
            # Close threat: strong avoidance reward for increasing distance
            return self.cfg.avoid_reward * (3.0 - threat_dist) / 3.0
        elif threat_dist < 5.0:
            return self.cfg.avoid_reward * 0.3
        return 0.0

    # ── EXPLOIT ──────────────────────────────────────────────────────────────
    # Biological analogue: habit formation via dorsal striatum.
    # Rewards revisiting locations where the agent previously found rewards
    # (food collected, goal reached) or taking actions that led to success.

    def _exploit_reward(self, info: Dict) -> float:
        pos = tuple(info["agent_pos"])

        # Update success map if something good happened
        if info.get("food_collected", False) or info.get("success", False):
            self.exploit_success_map[pos] = self.exploit_success_map.get(pos, 0) + 1.0

        # Reward for being at a previously successful location
        if pos in self.exploit_success_map:
            return self.cfg.exploit_reward * min(self.exploit_success_map[pos], 3.0) / 3.0
        return 0.0

    # ── MEMORIZE ─────────────────────────────────────────────────────────────
    # Biological analogue: hippocampal place cell consolidation.
    # Rewards the agent for systematically building a spatial map and using it.
    # The agent gets rewarded for covering the space evenly (building the map)
    # and for efficiently re-navigating to remembered locations.

    def _memorize_reward(self, info: Dict) -> float:
        pos = info["agent_pos"]
        step = info["step"]

        if self.spatial_memory is None:
            return 0.0

        r, c = pos[0], pos[1]

        # Reward for filling in the spatial memory map
        if self.spatial_memory[r, c] == 0:
            self.spatial_memory[r, c] = step  # Record when this cell was first visited
            # Reward proportional to how much of the map we've covered
            coverage = np.count_nonzero(self.spatial_memory) / self.spatial_memory.size
            return self.cfg.memorize_reward * coverage

        # Small reward for efficiently using memory (revisiting after a gap)
        time_since_visit = step - self.spatial_memory[r, c]
        if time_since_visit > 20:
            self.spatial_memory[r, c] = step
            return self.cfg.memorize_reward * 0.5

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
