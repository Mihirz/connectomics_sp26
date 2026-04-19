"""
models.py — Neural Network Architectures

Two models, carefully matched so the ONLY difference is the meta-controller:

1. AugmentedModel (PFC):  CNN encoder + meta-controller + objective-conditioned policy
2. BaselineModel:         CNN encoder + standard policy (same total parameter count)

Both use the same CNN encoder architecture so that any performance difference
is attributable to the training paradigm, not the visual feature extractor.

Architecture choices:
- Small CNN encoder (4 conv layers) because our observations are only 32×32.
- Separate policy and value heads (actor-critic).
- The augmented model additionally has:
    - A meta-controller head (outputs distribution over sub-objectives)
    - Sub-objective embeddings concatenated to the latent state before the policy
    - A meta-value head (estimates value under the meta-controller's sparse reward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional
from config import ModelConfig, EnvConfig


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CNN ENCODER
# ═══════════════════════════════════════════════════════════════════════════════
# Processes raw pixel observations into a compact latent representation.
# Architecture: 4 conv layers with stride 2, reducing 32×32 → 2×2.
#
# Why CNN and not a transformer?  For 32×32 grid worlds, a small CNN is
# perfectly adequate, uses minimal VRAM, and trains fast.  Transformers
# would be overkill and waste our GPU budget.

class CNNEncoder(nn.Module):
    """
    Shared visual encoder for both models.

    Input:  (batch, 3, 32, 32)  — RGB observation
    Output: (batch, latent_dim) — Compact latent state vector

    Layer-by-layer:
        Conv2d(3→16, 3×3, stride=2)  → (16, 16, 16)   + ReLU
        Conv2d(16→32, 3×3, stride=2) → (32, 8, 8)     + ReLU
        Conv2d(32→64, 3×3, stride=2) → (64, 4, 4)     + ReLU  (with padding)
        Conv2d(64→64, 3×3, stride=2) → (64, 2, 2)     + ReLU  (with padding)
        Flatten → (256,)
    """

    def __init__(self, cfg: ModelConfig, obs_channels: int = 3, grid_size: int = 20):
        super().__init__()
        channels = [obs_channels] + cfg.encoder_channels

        layers = []
        for i in range(len(channels) - 1):
            # Use padding=1 for the last two layers to avoid dimension collapse
            padding = 1 if i >= 2 else 0
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=padding))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

        # Compute output size dynamically using actual grid dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, obs_channels, grid_size, grid_size)
            conv_out = self.conv(dummy)
            self.flat_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flat_size, cfg.latent_dim)
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, C, H, W) float tensor
        Returns:
            (batch, latent_dim) latent state
        """
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.norm(x)
        return F.relu(x)


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTED MODEL (PFC PARADIGM)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the experimental model.  It has three key components beyond the encoder:
#
# 1. META-CONTROLLER: A small network that looks at the current latent state
#    and outputs a probability distribution over sub-objectives.  This is the
#    "prefrontal cortex" — it decides WHAT to optimize for right now.
#    Trained with sparse negative-only extrinsic reward.
#
# 2. SUB-OBJECTIVE EMBEDDINGS: Learned vectors for each sub-objective.
#    Concatenated with the latent state, these condition the action policy
#    on the currently selected strategy.
#
# 3. CONDITIONED ACTION POLICY: Takes [latent_state; sub_obj_embedding] and
#    outputs an action distribution.  Trained with the intrinsic reward from
#    the selected sub-objective.

class MetaController(nn.Module):
    """
    The "prefrontal cortex" — selects which sub-objective to pursue.

    Uses a GRU cell to integrate information over time, mirroring how the
    biological PFC maintains working memory of recent experience to guide
    strategy selection.  The GRU hidden state accumulates a summary of
    what the agent has seen and done, enabling decisions like "I've been
    exploring for 40 steps with no progress — switch to approach."

    Training signal: ONLY sparse negative reward on task failure.
    The meta-controller must learn which sub-objectives, in which contexts,
    lead to NOT failing.  It receives zero positive reward for success.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, num_objectives: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
        )

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Args:
            latent: (batch, latent_dim) from CNN encoder
            hidden: (batch, hidden_dim) GRU hidden state
        Returns:
            Categorical distribution over sub-objectives, updated hidden state
        """
        h = self.gru(latent, hidden)
        logits = self.head(h)
        return Categorical(logits=logits), h

    def init_hidden(self, batch_size: int, device: str) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class AugmentedModel(nn.Module):
    """
    The full augmented model with self-optimizing capability.

    Forward pass flow:
        1. Encode observation → latent state
        2. Meta-controller selects sub-objective from latent state
        3. Look up sub-objective embedding
        4. Concatenate [latent; embedding] to condition the policy
        5. Policy head → action distribution
        6. Value head → state value (for intrinsic rewards)
        7. Meta-value head → state value (for sparse extrinsic reward)

    Conditioning uses concatenation: [latent_256; embed_32] = 288-dim input.
    The 32-dim embedding is ~11% of the conditioned vector, large enough
    for the policy to differentiate modes without fragmenting the latent space.
    """

    def __init__(self, model_cfg: ModelConfig, env_cfg: EnvConfig):
        super().__init__()

        self.encoder = CNNEncoder(model_cfg, env_cfg.obs_channels, env_cfg.grid_size)

        # ── Meta-controller (PFC) ──
        self.meta_controller = MetaController(
            model_cfg.latent_dim,
            model_cfg.meta_hidden_dim,
            model_cfg.num_sub_objectives,
        )

        # ── Sub-objective embeddings ──
        self.obj_embeddings = nn.Embedding(model_cfg.num_sub_objectives, model_cfg.objective_embed_dim)

        # ── Conditioned dimension: latent + embedding ──
        conditioned_dim = model_cfg.latent_dim + model_cfg.objective_embed_dim

        # ── Action policy (takes [latent; embedding]) ──
        self.policy_head = nn.Sequential(
            nn.Linear(conditioned_dim, model_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_cfg.hidden_dim, env_cfg.num_actions),
        )

        # ── Value head (also conditioned — value depends on active sub-objective) ──
        self.value_head = nn.Sequential(
            nn.Linear(conditioned_dim, model_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_cfg.hidden_dim, 1),
        )

        # ── Meta-value head (estimates value under sparse extrinsic reward) ──
        # This uses just the latent state (no sub-objective conditioning)
        # because the meta-controller needs to evaluate the state before
        # selecting a sub-objective.
        self.meta_value_head = nn.Sequential(
            nn.Linear(model_cfg.latent_dim, model_cfg.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(model_cfg.meta_hidden_dim, 1),
        )

    def _condition(self, latent: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
        """Concatenate latent state with sub-objective embedding."""
        obj_embed = self.obj_embeddings(obj_idx)
        return torch.cat([latent, obj_embed], dim=-1)

    def init_meta_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize GRU hidden state for the meta-controller."""
        device = next(self.parameters()).device
        return self.meta_controller.init_hidden(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        meta_hidden: torch.Tensor,
        deterministic: bool = False,
        forced_obj_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            obs:             (batch, C, H, W) observation
            meta_hidden:     (batch, meta_hidden_dim) GRU hidden state
            deterministic:   if True, use argmax instead of sampling
            forced_obj_idx:  (batch,) if provided, use this sub-objective instead
                             of sampling from the meta-controller.

        Returns:
            action:         (batch,) selected actions
            action_logprob: (batch,) log probability of selected actions
            obj_idx:        (batch,) selected sub-objective indices
            obj_logprob:    (batch,) log probability of selected sub-objectives
            value:          (batch,) estimated intrinsic value
            meta_value:     (batch,) estimated extrinsic (sparse) value
            meta_hidden:    (batch, meta_hidden_dim) updated GRU hidden state
        """
        # Step 1: Encode observation
        latent = self.encoder(obs)

        # Step 2: Meta-controller with GRU integrates temporal context
        obj_dist, new_hidden = self.meta_controller(latent, meta_hidden)
        if forced_obj_idx is not None:
            obj_idx = forced_obj_idx
        elif deterministic:
            obj_idx = obj_dist.probs.argmax(dim=-1)
        else:
            obj_idx = obj_dist.sample()
        obj_logprob = obj_dist.log_prob(obj_idx)

        # Step 3 & 4: Concatenate latent with sub-objective embedding
        conditioned = self._condition(latent, obj_idx)

        # Step 5: Action distribution
        action_logits = self.policy_head(conditioned)
        action_dist = Categorical(logits=action_logits)
        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        # Step 6 & 7: Value estimates
        value = self.value_head(conditioned).squeeze(-1)
        meta_value = self.meta_value_head(latent).squeeze(-1)

        return action, action_logprob, obj_idx, obj_logprob, value, meta_value, new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        obj_indices: torch.Tensor,
        meta_hidden: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate log probabilities and values for stored transitions.
        Used during PPO update to compute ratios and entropy.

        Args:
            meta_hidden: (batch, meta_hidden_dim) stored GRU hidden states from
                         rollout collection. Required for correct re-evaluation
                         of the recurrent meta-controller.
        """
        latent = self.encoder(obs)

        # Meta-controller (use stored hidden states from rollout)
        if meta_hidden is None:
            meta_hidden = self.init_meta_hidden(latent.size(0))
        obj_dist, _ = self.meta_controller(latent, meta_hidden)
        obj_logprob = obj_dist.log_prob(obj_indices)
        obj_entropy = obj_dist.entropy()

        # Concatenation-conditioned policy
        conditioned = self._condition(latent, obj_indices)
        action_logits = self.policy_head(conditioned)
        action_dist = Categorical(logits=action_logits)
        action_logprob = action_dist.log_prob(actions)
        action_entropy = action_dist.entropy()

        # Values
        value = self.value_head(conditioned).squeeze(-1)
        meta_value = self.meta_value_head(latent).squeeze(-1)

        return action_logprob, obj_logprob, value, meta_value, action_entropy, obj_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE MODEL (FIXED OBJECTIVE)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Standard actor-critic with the same CNN encoder and similar parameter count.
# No meta-controller, no sub-objectives.  Uses a single fixed dense reward.
#
# To keep the comparison fair, we add a few extra hidden units to the policy
# and value heads so the total parameter count roughly matches the augmented model.

class BaselineModel(nn.Module):
    """
    Standard actor-critic baseline.

    Same encoder, same action space, but a single monolithic policy trained
    with dense task-specific rewards.  No strategy selection.

    This is the control condition: it represents the "normal" way of training
    RL agents on a specific task.
    """

    def __init__(self, model_cfg: ModelConfig, env_cfg: EnvConfig):
        super().__init__()

        self.encoder = CNNEncoder(model_cfg, env_cfg.obs_channels, env_cfg.grid_size)

        # 2-layer heads (same depth as augmented model's heads) with a wider
        # hidden dim to compensate for the missing meta-controller, embeddings,
        # and meta-value head.  Target: roughly match augmented param count.
        wider_hidden = model_cfg.hidden_dim + 168  # ~296 to match augmented GRU + concat param count

        self.policy_head = nn.Sequential(
            nn.Linear(model_cfg.latent_dim, wider_hidden),
            nn.ReLU(),
            nn.Linear(wider_hidden, env_cfg.num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(model_cfg.latent_dim, wider_hidden),
            nn.ReLU(),
            nn.Linear(wider_hidden, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:         (batch,) selected actions
            action_logprob: (batch,) log probabilities
            value:          (batch,) estimated values
        """
        latent = self.encoder(obs)

        action_logits = self.policy_head(latent)
        action_dist = Categorical(logits=action_logits)
        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        value = self.value_head(latent).squeeze(-1)

        return action, action_logprob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate for PPO update."""
        latent = self.encoder(obs)

        action_logits = self.policy_head(latent)
        action_dist = Categorical(logits=action_logits)
        action_logprob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        value = self.value_head(latent).squeeze(-1)

        return action_logprob, value, entropy


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER COUNT COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_comparison(model_cfg: ModelConfig, env_cfg: EnvConfig):
    """Print parameter counts for both models to verify fair comparison."""
    augmented = AugmentedModel(model_cfg, env_cfg)
    baseline = BaselineModel(model_cfg, env_cfg)

    aug_params = count_parameters(augmented)
    base_params = count_parameters(baseline)

    print("=" * 60)
    print("MODEL PARAMETER COMPARISON")
    print("=" * 60)
    print(f"  Augmented (PFC) model: {aug_params:,} parameters")
    print(f"  Baseline (Fixed) model: {base_params:,} parameters")
    print(f"  Ratio: {aug_params/base_params:.2f}x")
    print(f"  Difference: {abs(aug_params-base_params):,} params")
    print("=" * 60)

    # Estimate VRAM usage (rough: 4 bytes/param + optimizer states)
    aug_mb = aug_params * 4 * 3 / (1024**2)  # ×3 for param + momentum + variance
    base_mb = base_params * 4 * 3 / (1024**2)
    print(f"  Estimated VRAM (model + optimizer):")
    print(f"    Augmented: ~{aug_mb:.1f} MB")
    print(f"    Baseline:  ~{base_mb:.1f} MB")
    print(f"    Combined:  ~{aug_mb + base_mb:.1f} MB")
    print("=" * 60)
