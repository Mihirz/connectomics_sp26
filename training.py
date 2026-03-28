"""
training.py — PPO Training Loops for Both Paradigms

This module contains the core training logic.  We implement PPO (Proximal Policy
Optimization) for both models because:
1. PPO is stable and well-understood — we want the training paradigm to be the
   experimental variable, not the RL algorithm.
2. PPO works with discrete action spaces (our grid-world actions).
3. PPO is memory-efficient (no replay buffer), fitting our ≤8 GB VRAM constraint.

KEY DIFFERENCE between the two training loops:

  Baseline:  Standard PPO.  Dense rewards from the environment.  One policy,
             one value head, one optimization target.

  Augmented: Two-level PPO.
             - OUTER LEVEL: The meta-controller is trained with PPO using only
               the sparse negative extrinsic reward.  It learns which sub-objectives
               to select in which states.
             - INNER LEVEL: The action policy is trained with PPO using the
               intrinsic reward from the selected sub-objective.  It learns how
               to execute each strategy.
             Both levels share the CNN encoder, which gets gradients from both.

The two-level structure is what implements "choosing its own optimization function":
the meta-controller picks the objective, and the action policy optimizes for it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from config import ExperimentConfig
from models import AugmentedModel, BaselineModel
from sub_objectives import IntrinsicRewardComputer, NUM_SUB_OBJECTIVES
from environments import make_vectorized_env


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLOUT BUFFER
# ═══════════════════════════════════════════════════════════════════════════════
# Stores transitions from environment rollouts.  Used by PPO to compute
# advantages and update the policy.  Deliberately kept small for VRAM efficiency.

class RolloutBuffer:
    """
    Stores a fixed number of steps from N parallel environments.

    Memory budget: For 16 envs × 128 steps × 32×32×3 obs:
        Observations: 16 × 128 × 3 × 32 × 32 × 4 bytes ≈ 25 MB
        Other tensors: negligible
        Total: ~30 MB — well within budget.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_shape: Tuple, device: str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.pos = 0

        # Pre-allocate tensors
        self.observations = torch.zeros(num_steps, num_envs, *obs_shape, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.action_logprobs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)           # For baseline (dense)
        self.intrinsic_rewards = torch.zeros(num_steps, num_envs, device=device) # For augmented (selected sub-obj)
        self.sparse_rewards = torch.zeros(num_steps, num_envs, device=device)    # For augmented meta-controller
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.meta_values = torch.zeros(num_steps, num_envs, device=device)       # For augmented
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.obj_indices = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.obj_logprobs = torch.zeros(num_steps, num_envs, device=device)

        # Computed during finalize
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.meta_advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.meta_returns = torch.zeros(num_steps, num_envs, device=device)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, key)[self.pos] = val
        self.pos += 1

    def reset(self):
        self.pos = 0

    def compute_gae(self, last_value, last_meta_value, gamma, gae_lambda):
        """
        Compute Generalized Advantage Estimation for both reward streams.

        GAE is used instead of simple discounted returns because it reduces
        variance while maintaining a controllable bias, which is crucial for
        the sparse-reward meta-controller where high variance would make
        learning nearly impossible.
        """
        # ── Action policy GAE (uses intrinsic rewards for augmented, dense for baseline) ──
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]

            # Use intrinsic_rewards for augmented model, rewards for baseline
            # (the caller sets whichever is appropriate)
            delta = self.intrinsic_rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

        # ── Meta-controller GAE (uses sparse rewards) ──
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_meta_value
            else:
                next_value = self.meta_values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.sparse_rewards[t] + gamma * next_value * next_non_terminal - self.meta_values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.meta_advantages[t] = gae
        self.meta_returns = self.meta_advantages + self.meta_values

    def get_batches(self, batch_size: int):
        """Yield random mini-batches for PPO epochs."""
        total = self.num_steps * self.num_envs
        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # Convert flat indices to (step, env) indices
            step_idx = batch_idx // self.num_envs
            env_idx = batch_idx % self.num_envs

            yield {
                "obs": self.observations[step_idx, env_idx],
                "actions": self.actions[step_idx, env_idx],
                "action_logprobs": self.action_logprobs[step_idx, env_idx],
                "obj_indices": self.obj_indices[step_idx, env_idx],
                "obj_logprobs": self.obj_logprobs[step_idx, env_idx],
                "advantages": self.advantages[step_idx, env_idx],
                "meta_advantages": self.meta_advantages[step_idx, env_idx],
                "returns": self.returns[step_idx, env_idx],
                "meta_returns": self.meta_returns[step_idx, env_idx],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTED MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class AugmentedTrainer:
    """
    Trains the augmented (PFC) model using the two-level PPO paradigm.

    Training flow per rollout:
    1. Collect N steps from the environment using the current policy.
    2. For each step, the meta-controller selects a sub-objective.
    3. The selected sub-objective generates an intrinsic reward.
    4. Compute GAE advantages for both the action policy (intrinsic rewards)
       and the meta-controller (sparse rewards).
    5. PPO update on the action policy using intrinsic advantages.
    6. PPO update on the meta-controller using sparse advantages.
    7. Both updates flow gradients through the shared encoder.
    """

    def __init__(self, cfg: ExperimentConfig, task_name: str):
        self.cfg = cfg
        self.device = cfg.device

        # ── Create model ──
        self.model = AugmentedModel(cfg.model, cfg.env).to(self.device)

        # ── Separate optimizers for the two levels ──
        # The meta-controller uses a lower learning rate for stability.
        # Using separate optimizers ensures the two levels don't interfere.
        self.optimizer = optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": cfg.train.lr_policy},
            {"params": self.model.policy_head.parameters(), "lr": cfg.train.lr_policy},
            {"params": self.model.value_head.parameters(), "lr": cfg.train.lr_policy},
            {"params": self.model.obj_embeddings.parameters(), "lr": cfg.train.lr_policy},
            {"params": self.model.meta_controller.parameters(), "lr": cfg.train.lr_meta},
            {"params": self.model.meta_value_head.parameters(), "lr": cfg.train.lr_meta},
        ])

        # ── Intrinsic reward computer ──
        self.reward_computer = IntrinsicRewardComputer(cfg.sub_obj)

        # ── Environment ──
        self.env = make_vectorized_env(
            task_name, cfg.env, cfg.train.num_parallel_envs, base_seed=cfg.seed
        )

        # ── Rollout buffer ──
        obs_shape = (cfg.env.obs_channels, cfg.env.grid_size, cfg.env.grid_size)
        self.buffer = RolloutBuffer(
            cfg.train.rollout_steps, cfg.train.num_parallel_envs, obs_shape, self.device
        )

        # ── Tracking ──
        self.episode_rewards = []
        self.episode_successes = []
        self.obj_selection_counts = np.zeros(NUM_SUB_OBJECTIVES)
        self.total_steps = 0

    def collect_rollout(self) -> Dict:
        """
        Collect one rollout of transitions from the environment.

        This is where the "self-optimization" happens at inference time:
        at each step, the meta-controller picks a sub-objective, and we
        compute the intrinsic reward from that sub-objective.
        """
        self.buffer.reset()
        obs_np, _ = self.env.reset()
        obs = torch.FloatTensor(obs_np).to(self.device)

        # Reset intrinsic reward state for each environment
        reward_computers = [IntrinsicRewardComputer(self.cfg.sub_obj) for _ in range(self.cfg.train.num_parallel_envs)]
        for rc in reward_computers:
            rc.reset(self.cfg.env.grid_size)

        episode_stats = defaultdict(list)

        for step in range(self.cfg.train.rollout_steps):
            with torch.no_grad():
                action, action_lp, obj_idx, obj_lp, value, meta_value = self.model(obs)

            # Execute actions in environment
            actions_np = action.cpu().numpy()
            obj_idx_np = obj_idx.cpu().numpy()
            obs_np, reward_infos, dones, infos = self.env.step(actions_np)

            # ── Compute intrinsic rewards from selected sub-objectives ──
            # THIS IS THE KEY MECHANISM: each environment gets a different
            # intrinsic reward based on what the meta-controller selected.
            intrinsic_rewards = np.zeros(self.cfg.train.num_parallel_envs, dtype=np.float32)
            sparse_rewards = np.zeros(self.cfg.train.num_parallel_envs, dtype=np.float32)

            for i in range(self.cfg.train.num_parallel_envs):
                ri = reward_infos[i]
                all_intrinsic = reward_computers[i].compute_all(ri)
                selected_obj = obj_idx_np[i]
                intrinsic_rewards[i] = all_intrinsic[selected_obj]
                sparse_rewards[i] = ri.get("sparse_reward", 0.0)

                # Track sub-objective usage
                self.obj_selection_counts[selected_obj] += 1

                if dones[i]:
                    episode_stats["success"].append(ri.get("success", False))
                    episode_stats["episode_length"].append(ri.get("step", 0))
                    # Reset this env's reward computer
                    reward_computers[i].reset(self.cfg.env.grid_size)

            # Store transition
            self.buffer.add(
                observations=obs,
                actions=action,
                action_logprobs=action_lp,
                obj_indices=obj_idx,
                obj_logprobs=obj_lp,
                intrinsic_rewards=torch.FloatTensor(intrinsic_rewards).to(self.device),
                sparse_rewards=torch.FloatTensor(sparse_rewards).to(self.device),
                values=value,
                meta_values=meta_value,
                dones=torch.FloatTensor(dones).to(self.device),
                rewards=torch.FloatTensor([ri["dense_reward"] for ri in reward_infos]).to(self.device),
            )

            obs = torch.FloatTensor(obs_np).to(self.device)
            self.total_steps += self.cfg.train.num_parallel_envs

        # Compute last values for GAE bootstrap
        with torch.no_grad():
            _, _, _, _, last_value, last_meta_value = self.model(obs)

        self.buffer.compute_gae(
            last_value, last_meta_value,
            self.cfg.train.gamma, self.cfg.train.gae_lambda
        )

        return dict(episode_stats)

    def update(self) -> Dict[str, float]:
        """
        PPO update on both levels.

        This implements the two-level optimization:
        - Action policy loss uses intrinsic advantages (from selected sub-objectives)
        - Meta-controller loss uses sparse advantages (from extrinsic negative-only signal)
        - Both share the encoder, which gets both gradient signals
        """
        cfg = self.cfg.train
        metrics = defaultdict(float)
        num_updates = 0

        for epoch in range(cfg.ppo_epochs):
            for batch in self.buffer.get_batches(cfg.mini_batch_size):
                # Re-evaluate actions and objectives
                action_lp, obj_lp, values, meta_values, action_ent, obj_ent = \
                    self.model.evaluate_actions(
                        batch["obs"], batch["actions"], batch["obj_indices"]
                    )

                # ── ACTION POLICY LOSS (intrinsic rewards) ──
                # Standard PPO clipped objective using intrinsic advantages.
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(action_lp - batch["action_logprobs"])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()

                # ── META-CONTROLLER LOSS (sparse negative rewards) ──
                # Same PPO structure but using sparse advantages.
                meta_advantages = batch["meta_advantages"]
                if meta_advantages.std() > 1e-8:
                    meta_advantages = (meta_advantages - meta_advantages.mean()) / (meta_advantages.std() + 1e-8)

                meta_ratio = torch.exp(obj_lp - batch["obj_logprobs"])
                meta_surr1 = meta_ratio * meta_advantages
                meta_surr2 = torch.clamp(meta_ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * meta_advantages
                meta_loss = -torch.min(meta_surr1, meta_surr2).mean()

                meta_value_loss = 0.5 * (meta_values - batch["meta_returns"]).pow(2).mean()

                # ── TOTAL LOSS ──
                # Combine both levels.  The entropy bonuses encourage exploration:
                # - Higher meta-entropy → explore different sub-objectives
                # - Standard action entropy → explore different actions
                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    + meta_loss
                    + cfg.value_loss_coef * meta_value_loss
                    - cfg.entropy_coef * action_ent.mean()
                    - cfg.meta_entropy_coef * obj_ent.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["meta_loss"] += meta_loss.item()
                metrics["action_entropy"] += action_ent.mean().item()
                metrics["obj_entropy"] += obj_ent.mean().item()
                num_updates += 1

        return {k: v / max(num_updates, 1) for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineTrainer:
    """
    Standard PPO trainer for the baseline model.

    Uses dense rewards directly from the environment.
    No meta-controller, no sub-objectives, no intrinsic rewards.
    This is the control condition.
    """

    def __init__(self, cfg: ExperimentConfig, task_name: str):
        self.cfg = cfg
        self.device = cfg.device

        self.model = BaselineModel(cfg.model, cfg.env).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.lr_policy)

        self.env = make_vectorized_env(
            task_name, cfg.env, cfg.train.num_parallel_envs, base_seed=cfg.seed + 1000
        )

        obs_shape = (cfg.env.obs_channels, cfg.env.grid_size, cfg.env.grid_size)
        self.buffer = RolloutBuffer(
            cfg.train.rollout_steps, cfg.train.num_parallel_envs, obs_shape, self.device
        )

        self.total_steps = 0

    def collect_rollout(self) -> Dict:
        self.buffer.reset()
        obs_np, _ = self.env.reset()
        obs = torch.FloatTensor(obs_np).to(self.device)

        episode_stats = defaultdict(list)

        for step in range(self.cfg.train.rollout_steps):
            with torch.no_grad():
                action, action_lp, value = self.model(obs)

            actions_np = action.cpu().numpy()
            obs_np, reward_infos, dones, infos = self.env.step(actions_np)

            # Use dense rewards for baseline
            dense_rewards = np.array([ri["dense_reward"] for ri in reward_infos], dtype=np.float32)

            for i in range(self.cfg.train.num_parallel_envs):
                if dones[i]:
                    episode_stats["success"].append(reward_infos[i].get("success", False))
                    episode_stats["episode_length"].append(reward_infos[i].get("step", 0))

            # Store transition (use intrinsic_rewards field for dense rewards in baseline)
            self.buffer.add(
                observations=obs,
                actions=action,
                action_logprobs=action_lp,
                intrinsic_rewards=torch.FloatTensor(dense_rewards).to(self.device),
                sparse_rewards=torch.zeros(self.cfg.train.num_parallel_envs, device=self.device),
                values=value,
                meta_values=torch.zeros(self.cfg.train.num_parallel_envs, device=self.device),
                dones=torch.FloatTensor(dones).to(self.device),
                obj_indices=torch.zeros(self.cfg.train.num_parallel_envs, dtype=torch.long, device=self.device),
                obj_logprobs=torch.zeros(self.cfg.train.num_parallel_envs, device=self.device),
                rewards=torch.FloatTensor(dense_rewards).to(self.device),
            )

            obs = torch.FloatTensor(obs_np).to(self.device)
            self.total_steps += self.cfg.train.num_parallel_envs

        with torch.no_grad():
            _, _, last_value = self.model(obs)

        self.buffer.compute_gae(
            last_value,
            torch.zeros(self.cfg.train.num_parallel_envs, device=self.device),
            self.cfg.train.gamma, self.cfg.train.gae_lambda
        )

        return dict(episode_stats)

    def update(self) -> Dict[str, float]:
        """Standard single-level PPO update."""
        cfg = self.cfg.train
        metrics = defaultdict(float)
        num_updates = 0

        for epoch in range(cfg.ppo_epochs):
            for batch in self.buffer.get_batches(cfg.mini_batch_size):
                action_lp, values, entropy = self.model.evaluate_actions(
                    batch["obs"], batch["actions"]
                )

                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(action_lp - batch["action_logprobs"])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()

                loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["action_entropy"] += entropy.mean().item()
                num_updates += 1

        return {k: v / max(num_updates, 1) for k, v in metrics.items()}
