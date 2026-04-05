"""
environments.py — Biologically-Inspired Grid-World Task Environments

Each environment simulates a classic neuroscience or ethology experiment as a
lightweight 2D grid-world.  Observations are rendered as small RGB images
(3 × grid_size × grid_size) suitable for CNN processing.

All environments follow the same API:
    obs, info = env.reset()
    obs, reward_info, done, info = env.step(action)

`reward_info` is a dict containing BOTH the dense reward (for baseline) and
the sparse negative signal (for augmented model), plus auxiliary info needed
by sub-objectives (distances, novelty, threats, etc.).  This lets us train
both models on the exact same environment rollouts.

Design principle: these tasks are chosen because they *benefit from flexible
strategy selection*.  A model that can dynamically switch between exploring,
exploiting, avoiding, and remembering will outperform one locked into a
single behavioral mode.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from config import EnvConfig


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
# We use distinct colors so the CNN can easily parse the scene.  The "vision"
# challenge comes from partial observability and clutter, not raw pixel decoding.

COLORS = {
    "background":  np.array([0.1, 0.1, 0.15]),   # Dark background
    "agent":       np.array([0.0, 0.8, 0.2]),     # Green — the agent
    "goal":        np.array([1.0, 0.85, 0.0]),    # Gold — goal / platform
    "goal_hidden": np.array([0.1, 0.1, 0.15]),    # Same as background (invisible)
    "wall":        np.array([0.4, 0.4, 0.5]),     # Grey — walls / boundaries
    "water":       np.array([0.15, 0.25, 0.6]),   # Blue — water in Morris maze
    "food":        np.array([0.9, 0.3, 0.1]),     # Orange — food items
    "predator":    np.array([0.8, 0.0, 0.0]),     # Red — threats / predators
    "obstacle":    np.array([0.6, 0.2, 0.6]),     # Purple — moving obstacles
    "distractor":  np.array([0.3, 0.3, 0.3]),     # Dim grey — visual clutter
    "cue_arrow":   np.array([0.0, 0.6, 0.9]),     # Cyan — helpful cue
    "visited":     np.array([0.15, 0.18, 0.22]),   # Slightly lighter — visited cells
}

# Action mapping: 0=up, 1=down, 2=left, 3=right, 4=stay
ACTION_DELTAS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (0, 0),    # stay
}


# ═══════════════════════════════════════════════════════════════════════════════
# BASE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class BaseGridEnv:
    """
    Shared infrastructure for all grid-world tasks.

    Every environment tracks:
    - Agent position
    - Step count
    - Visit counts per cell (for novelty / exploration tracking)
    - A rendered observation image

    The `reward_info` dict returned by step() contains everything both the
    baseline (dense reward) and augmented (sparse + sub-objectives) models need.
    """

    def __init__(self, cfg: EnvConfig, variant_seed: Optional[int] = None):
        self.cfg = cfg
        self.size = cfg.grid_size
        self.max_steps = cfg.max_steps_per_episode
        self.variant_seed = variant_seed  # For generating transfer variants
        self.rng = np.random.RandomState(variant_seed)

        # State
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.float32)
        self.done = False

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def _move_agent(self, action: int) -> np.ndarray:
        """Move agent by action delta; returns new position (clipped to bounds)."""
        dr, dc = ACTION_DELTAS[action]
        new_pos = self.agent_pos + np.array([dr, dc])
        if self._in_bounds(new_pos) and self._is_passable(new_pos):
            self.agent_pos = new_pos
        return self.agent_pos

    def _is_passable(self, pos: np.ndarray) -> bool:
        """Override in subclasses to add walls / boundaries."""
        return True

    def _render(self) -> np.ndarray:
        """Render current state as (3, H, W) float32 image in [0, 1]."""
        raise NotImplementedError

    def _compute_reward_info(self, action: int, old_pos: np.ndarray) -> Dict:
        """Compute reward signals and auxiliary info. Override per task."""
        raise NotImplementedError

    def reset(self) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, Dict, bool, Dict]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: MORRIS WATER MAZE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Biological basis: The Morris water maze is the gold standard for testing
# spatial learning and memory in rodents.  The animal is placed in a circular
# pool of opaque water and must find a hidden platform using spatial cues.
#
# Why it suits our paradigm: Success requires *switching strategies* —
# initially EXPLORE (swim around to find the platform), then MEMORIZE (remember
# where it was), then EXPLOIT (swim directly to it on subsequent trials).
# A model locked into a single strategy (e.g., always explore) will be slow.
# The augmented model's meta-controller should learn this strategy sequence.

class MorrisWaterMaze(BaseGridEnv):

    def __init__(self, cfg: EnvConfig, variant_seed: Optional[int] = None):
        super().__init__(cfg, variant_seed)
        self.pool_center = np.array([self.size // 2, self.size // 2], dtype=np.float32)
        self.pool_radius = cfg.mwm_pool_radius * self.size
        self.platform_radius = cfg.mwm_platform_radius * self.size

        # Pre-compute pool mask (which cells are "water")
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        dist_from_center = np.sqrt((yy - self.pool_center[0])**2 + (xx - self.pool_center[1])**2)
        self.pool_mask = dist_from_center <= self.pool_radius

        self.platform_pos = None
        self.found_platform = False
        self.min_dist_to_platform = float("inf")

    def _is_passable(self, pos: np.ndarray) -> bool:
        """Agent can only move within the circular pool."""
        return bool(self.pool_mask[pos[0], pos[1]])

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        self.done = False
        self.found_platform = False
        self.min_dist_to_platform = float("inf")
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.float32)

        # Place platform at a random position inside the pool
        # (For transfer variants, use variant_seed to shift platform location.)
        while True:
            pr = self.rng.randint(4, self.size - 4)
            pc = self.rng.randint(4, self.size - 4)
            if self.pool_mask[pr, pc]:
                self.platform_pos = np.array([pr, pc])
                break

        # Place agent at a random edge of the pool
        edge_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.pool_mask[r, c]:
                    # Check if any neighbor is outside pool
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if not (0 <= nr < self.size and 0 <= nc < self.size) or not self.pool_mask[nr, nc]:
                            edge_cells.append((r, c))
                            break
        idx = self.rng.randint(len(edge_cells))
        self.agent_pos = np.array(edge_cells[idx])
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] = 1

        return self._render(), {"task": "morris_water_maze"}

    def step(self, action: int) -> Tuple[np.ndarray, Dict, bool, Dict]:
        old_pos = self.agent_pos.copy()
        self.step_count += 1
        self._move_agent(action)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1

        reward_info = self._compute_reward_info(action, old_pos)

        # Track closest approach for distance-scaled failure penalty
        self.min_dist_to_platform = min(self.min_dist_to_platform, reward_info["dist_to_goal"])

        if self.step_count >= self.max_steps:
            self.done = True
            # Scale penalty by how close the agent got during the episode.
            # An agent that nearly found the platform gets a milder penalty (-0.3)
            # than one that stayed far away (-1.0).  This gives the meta-controller
            # a gradient: sub-objectives that move toward the platform produce
            # less negative outcomes, even when the episode times out.
            closeness = 1.0 - min(self.min_dist_to_platform / (self.pool_radius * 2), 1.0)
            reward_info["sparse_reward"] = -1.0 + closeness * 0.7  # range: -1.0 to -0.3
            reward_info["timeout"] = True

        return self._render(), reward_info, self.done, {"task": "morris_water_maze"}

    def _compute_reward_info(self, action: int, old_pos: np.ndarray) -> Dict:
        dist_to_platform = np.linalg.norm(self.agent_pos - self.platform_pos)
        old_dist = np.linalg.norm(old_pos - self.platform_pos)
        on_platform = dist_to_platform <= self.platform_radius

        # ── Dense reward (BASELINE model uses this) ──
        dense_reward = -0.01  # step cost
        if on_platform:
            dense_reward = 1.0
            self.found_platform = True
            self.done = True

        # ── Sparse reward (AUGMENTED model uses this) ──
        # Only negative signal on failure; zero on success.
        sparse_reward = 0.0
        if on_platform:
            self.done = True
            # No positive reward!  The augmented model must learn from intrinsic rewards.

        # ── Auxiliary info for sub-objectives ──
        is_novel = self.visit_counts[self.agent_pos[0], self.agent_pos[1]] <= 1
        approach_delta = old_dist - dist_to_platform  # positive if moving toward platform

        return {
            "dense_reward": dense_reward,
            "sparse_reward": sparse_reward,
            "done": self.done,
            "success": on_platform,
            "timeout": False,
            # Sub-objective auxiliary signals
            "is_novel_cell": is_novel,
            "visit_count": self.visit_counts[self.agent_pos[0], self.agent_pos[1]],
            "dist_to_goal": dist_to_platform,
            "approach_delta": approach_delta,
            "threat_nearby": False,       # No threats in this task
            "threat_dist": float("inf"),
            "agent_pos": self.agent_pos.copy(),
            "step": self.step_count,
        }

    def _render(self) -> np.ndarray:
        img = np.zeros((3, self.size, self.size), dtype=np.float32)

        # Background
        for c in range(3):
            img[c] = COLORS["background"][c]

        # Pool water
        for c in range(3):
            img[c][self.pool_mask] = COLORS["water"][c]

        # ── Proximity gradient (warmth near the platform) ──
        # This gives the CNN a learnable signal without directly revealing the
        # platform location.  Analogous to water temperature gradients that
        # some aquatic animals can sense.  Visible within 70% of pool radius,
        # strong enough for the CNN to detect from several cells away.
        if self.platform_pos is not None:
            yy, xx = np.mgrid[0:self.size, 0:self.size]
            dist_to_plat = np.sqrt((yy - self.platform_pos[0])**2 +
                                   (xx - self.platform_pos[1])**2)
            gradient_radius = self.pool_radius * 0.7
            gradient_mask = (dist_to_plat < gradient_radius) & self.pool_mask
            warmth = 1.0 - (dist_to_plat[gradient_mask] / gradient_radius)
            warmth = warmth * 0.3  # 30% max color shift — clearly detectable
            img[0][gradient_mask] += warmth       # Red/warm tint increases
            img[2][gradient_mask] -= warmth * 0.5  # Blue decreases

        # ── Landmark cues (4 colored markers at pool edges) ──
        # In the real Morris water maze, visual cues on the room walls provide
        # allocentric spatial reference.  We place 4 distinct colored markers
        # at the N/S/E/W extremes of the pool boundary.
        cx, cy = int(self.pool_center[0]), int(self.pool_center[1])
        pr = int(self.pool_radius)
        landmark_colors = [
            np.array([0.9, 0.2, 0.2]),   # North — red
            np.array([0.2, 0.2, 0.9]),   # South — blue
            np.array([0.9, 0.9, 0.2]),   # East — yellow
            np.array([0.2, 0.9, 0.9]),   # West — cyan
        ]
        landmark_positions = [
            (cx - pr, cy),     # North
            (cx + pr, cy),     # South
            (cx, cy + pr),     # East
            (cx, cy - pr),     # West
        ]
        for (lr, lc), color in zip(landmark_positions, landmark_colors):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r, c_idx = lr + dr, lc + dc
                    if 0 <= r < self.size and 0 <= c_idx < self.size:
                        for c in range(3):
                            img[c, r, c_idx] = color[c]

        # Visited cells (subtle shading to help the model track its own path)
        visited_mask = (self.visit_counts > 0) & self.pool_mask
        for c in range(3):
            img[c][visited_mask] = np.maximum(img[c][visited_mask],
                                               COLORS["visited"][c])

        # Platform — HIDDEN unless agent is on it
        if self.found_platform:
            pr, pc = self.platform_pos
            r_lo = max(0, int(pr - self.platform_radius))
            r_hi = min(self.size, int(pr + self.platform_radius) + 1)
            c_lo = max(0, int(pc - self.platform_radius))
            c_hi = min(self.size, int(pc + self.platform_radius) + 1)
            for c in range(3):
                img[c][r_lo:r_hi, c_lo:c_hi] = COLORS["goal"][c]

        # Agent
        ar, ac = self.agent_pos
        for c in range(3):
            img[c, ar, ac] = COLORS["agent"][c]

        return img


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: VISUAL FORAGING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Biological basis: Foraging is one of the most fundamental behaviors across
# all animal species.  It requires balancing exploration (finding new food),
# exploitation (collecting known food), and avoidance (predator evasion).
#
# Why it suits our paradigm: The optimal strategy shifts dynamically.  When
# food is abundant, EXPLOIT; when food is scarce, EXPLORE; when a predator is
# near, AVOID.  The augmented model's meta-controller should learn to switch
# sub-objectives in response to the current context.

class VisualForaging(BaseGridEnv):

    def __init__(self, cfg: EnvConfig, variant_seed: Optional[int] = None):
        super().__init__(cfg, variant_seed)
        self.num_food = cfg.forage_num_food
        self.num_predators = cfg.forage_num_predators
        self.predator_speed = cfg.forage_predator_speed
        self.collect_target = cfg.forage_food_collect_target

        self.food_positions = []
        self.predator_positions = []
        self.predator_directions = []
        self.collected = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        self.done = False
        self.collected = 0
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.float32)

        # Place agent at center
        self.agent_pos = np.array([self.size // 2, self.size // 2])

        # Scatter food randomly
        self.food_positions = []
        for _ in range(self.num_food):
            pos = np.array([self.rng.randint(1, self.size-1), self.rng.randint(1, self.size-1)])
            self.food_positions.append(pos)

        # Place predators at edges with random directions
        self.predator_positions = []
        self.predator_directions = []
        for _ in range(self.num_predators):
            edge = self.rng.randint(4)
            if edge == 0: pos = np.array([0, self.rng.randint(self.size)], dtype=np.float32)
            elif edge == 1: pos = np.array([self.size-1, self.rng.randint(self.size)], dtype=np.float32)
            elif edge == 2: pos = np.array([self.rng.randint(self.size), 0], dtype=np.float32)
            else: pos = np.array([self.rng.randint(self.size), self.size-1], dtype=np.float32)
            direction = self.rng.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.predator_positions.append(pos)
            self.predator_directions.append(direction)

        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] = 1
        return self._render(), {"task": "visual_foraging"}

    def _move_predators(self):
        """Move predators along their directions, bouncing off walls."""
        for i in range(len(self.predator_positions)):
            # Stochastic movement based on speed
            if self.rng.rand() < self.predator_speed:
                new_pos = self.predator_positions[i] + self.predator_directions[i]
                # Bounce off walls
                for dim in range(2):
                    if new_pos[dim] < 0 or new_pos[dim] >= self.size:
                        self.predator_directions[i][dim] *= -1
                        new_pos[dim] = np.clip(new_pos[dim], 0, self.size - 1)
                # Slight random perturbation to direction
                self.predator_directions[i] += self.rng.randn(2) * 0.1
                self.predator_directions[i] /= np.linalg.norm(self.predator_directions[i]) + 1e-8
                self.predator_positions[i] = new_pos

    def step(self, action: int) -> Tuple[np.ndarray, Dict, bool, Dict]:
        old_pos = self.agent_pos.copy()
        self.step_count += 1
        self._move_agent(action)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1
        self._move_predators()

        # Check food collection
        food_collected_this_step = False
        remaining_food = []
        for fp in self.food_positions:
            if np.array_equal(self.agent_pos, fp):
                self.collected += 1
                food_collected_this_step = True
            else:
                remaining_food.append(fp)
        self.food_positions = remaining_food

        # Check predator collision
        predator_caught = False
        min_predator_dist = float("inf")
        for pp in self.predator_positions:
            d = np.linalg.norm(self.agent_pos.astype(float) - pp)
            min_predator_dist = min(min_predator_dist, d)
            if d < 1.5:  # Caught!
                predator_caught = True

        reward_info = self._compute_reward_info(
            action, old_pos, food_collected_this_step, predator_caught, min_predator_dist
        )

        if predator_caught or self.collected >= self.collect_target or self.step_count >= self.max_steps:
            self.done = True
            if self.step_count >= self.max_steps and self.collected < self.collect_target:
                progress = self.collected / max(self.collect_target, 1)
                reward_info["sparse_reward"] = -1.0 + progress * 0.7
                reward_info["timeout"] = True
            if predator_caught:
                reward_info["sparse_reward"] = -1.0

        return self._render(), reward_info, self.done, {"task": "visual_foraging"}

    def _compute_reward_info(self, action, old_pos, food_collected, predator_caught, min_pred_dist):
        # Distance to nearest food
        if len(self.food_positions) > 0:
            dists = [np.linalg.norm(self.agent_pos - fp) for fp in self.food_positions]
            nearest_food_dist = min(dists)
            old_dists = [np.linalg.norm(old_pos - fp) for fp in self.food_positions]
            approach_delta = min(old_dists) - nearest_food_dist
        else:
            nearest_food_dist = 0
            approach_delta = 0

        # Dense reward
        dense_reward = -0.01
        if food_collected:
            dense_reward += 0.3
        if predator_caught:
            dense_reward -= 1.0
        if self.collected >= self.collect_target:
            dense_reward += 1.0

        # Sparse reward
        sparse_reward = 0.0
        # (set by step() on failure conditions)

        is_novel = self.visit_counts[self.agent_pos[0], self.agent_pos[1]] <= 1

        return {
            "dense_reward": dense_reward,
            "sparse_reward": sparse_reward,
            "done": self.done,
            "success": self.collected >= self.collect_target and not predator_caught,
            "timeout": False,
            "is_novel_cell": is_novel,
            "visit_count": self.visit_counts[self.agent_pos[0], self.agent_pos[1]],
            "dist_to_goal": nearest_food_dist,
            "approach_delta": approach_delta,
            "threat_nearby": min_pred_dist < 5.0,
            "threat_dist": min_pred_dist,
            "agent_pos": self.agent_pos.copy(),
            "step": self.step_count,
            "food_collected": food_collected,
            "items_collected": self.collected,
        }

    def _render(self) -> np.ndarray:
        img = np.zeros((3, self.size, self.size), dtype=np.float32)
        for c in range(3):
            img[c] = COLORS["background"][c]

        # Visited cells
        visited_mask = self.visit_counts > 0
        for c in range(3):
            img[c][visited_mask] = COLORS["visited"][c]

        # Food
        for fp in self.food_positions:
            for c in range(3):
                img[c, fp[0], fp[1]] = COLORS["food"][c]

        # Predators (rendered as 3×3 blocks)
        for pp in self.predator_positions:
            pr, pc = int(round(pp[0])), int(round(pp[1]))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r, c_idx = pr + dr, pc + dc
                    if 0 <= r < self.size and 0 <= c_idx < self.size:
                        for c in range(3):
                            img[c, r, c_idx] = COLORS["predator"][c]

        # Agent
        ar, ac = self.agent_pos
        for c in range(3):
            img[c, ar, ac] = COLORS["agent"][c]

        return img


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: DYNAMIC OBSTACLE COURSE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Biological basis: Navigating through a dynamic environment with moving
# hazards is analogous to a fish navigating through currents and predators,
# or a human crossing a busy street.  Requires real-time path planning.
#
# Why it suits our paradigm: The optimal strategy depends on obstacle
# configuration.  Sometimes AVOID is critical; other times a bold APPROACH
# toward the goal through a gap is best.  The model needs moment-to-moment
# strategy switching.

class DynamicObstacleCourse(BaseGridEnv):

    def __init__(self, cfg: EnvConfig, variant_seed: Optional[int] = None):
        super().__init__(cfg, variant_seed)
        self.num_obstacles = cfg.obstacle_num_obstacles
        self.obstacle_speed = cfg.obstacle_speed

        self.goal_pos = None
        self.obstacle_positions = []
        self.obstacle_directions = []

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        self.done = False
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.float32)

        # Agent starts bottom-left, goal is top-right
        self.agent_pos = np.array([self.size - 2, 1])
        self.goal_pos = np.array([1, self.size - 2])

        # Scatter obstacles in the middle band
        self.obstacle_positions = []
        self.obstacle_directions = []
        for _ in range(self.num_obstacles):
            r = self.rng.randint(3, self.size - 3)
            c = self.rng.randint(3, self.size - 3)
            pos = np.array([r, c], dtype=np.float32)
            # Random direction, mostly horizontal or vertical
            if self.rng.rand() > 0.5:
                direction = np.array([0, self.rng.choice([-1, 1])], dtype=np.float32)
            else:
                direction = np.array([self.rng.choice([-1, 1]), 0], dtype=np.float32)
            self.obstacle_positions.append(pos)
            self.obstacle_directions.append(direction)

        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] = 1
        return self._render(), {"task": "dynamic_obstacles"}

    def _move_obstacles(self):
        for i in range(len(self.obstacle_positions)):
            if self.rng.rand() < self.obstacle_speed:
                new_pos = self.obstacle_positions[i] + self.obstacle_directions[i]
                for dim in range(2):
                    if new_pos[dim] < 1 or new_pos[dim] >= self.size - 1:
                        self.obstacle_directions[i][dim] *= -1
                        new_pos[dim] = np.clip(new_pos[dim], 1, self.size - 2)
                self.obstacle_positions[i] = new_pos

    def step(self, action: int) -> Tuple[np.ndarray, Dict, bool, Dict]:
        old_pos = self.agent_pos.copy()
        self.step_count += 1
        self._move_agent(action)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1
        self._move_obstacles()

        # Check collision with obstacles
        collision = False
        min_obs_dist = float("inf")
        for op in self.obstacle_positions:
            d = np.linalg.norm(self.agent_pos.astype(float) - op)
            min_obs_dist = min(min_obs_dist, d)
            if d < 1.5:
                collision = True

        reached_goal = np.array_equal(self.agent_pos, self.goal_pos)

        reward_info = self._compute_reward_info(action, old_pos, collision, reached_goal, min_obs_dist)

        if collision or reached_goal or self.step_count >= self.max_steps:
            self.done = True
            if collision:
                reward_info["sparse_reward"] = -1.0
            if self.step_count >= self.max_steps and not reached_goal:
                closeness = 1.0 - min(reward_info["dist_to_goal"] / (self.size * 1.4), 1.0)
                reward_info["sparse_reward"] = -1.0 + closeness * 0.7
                reward_info["timeout"] = True

        return self._render(), reward_info, self.done, {"task": "dynamic_obstacles"}

    def _compute_reward_info(self, action, old_pos, collision, reached_goal, min_obs_dist):
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        old_dist = np.linalg.norm(old_pos - self.goal_pos)
        approach_delta = old_dist - dist_to_goal

        dense_reward = -0.01
        if reached_goal:
            dense_reward = 1.0
        if collision:
            dense_reward = -0.5

        is_novel = self.visit_counts[self.agent_pos[0], self.agent_pos[1]] <= 1

        return {
            "dense_reward": dense_reward,
            "sparse_reward": 0.0,
            "done": self.done,
            "success": reached_goal,
            "timeout": False,
            "is_novel_cell": is_novel,
            "visit_count": self.visit_counts[self.agent_pos[0], self.agent_pos[1]],
            "dist_to_goal": dist_to_goal,
            "approach_delta": approach_delta,
            "threat_nearby": min_obs_dist < 4.0,
            "threat_dist": min_obs_dist,
            "agent_pos": self.agent_pos.copy(),
            "step": self.step_count,
        }

    def _render(self) -> np.ndarray:
        img = np.zeros((3, self.size, self.size), dtype=np.float32)
        for c in range(3):
            img[c] = COLORS["background"][c]

        # Visited
        visited_mask = self.visit_counts > 0
        for c in range(3):
            img[c][visited_mask] = COLORS["visited"][c]

        # Goal
        gr, gc = self.goal_pos
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, ci = gr+dr, gc+dc
                if 0 <= r < self.size and 0 <= ci < self.size:
                    for c in range(3):
                        img[c, r, ci] = COLORS["goal"][c]

        # Obstacles (2×2 blocks)
        for op in self.obstacle_positions:
            or_, oc = int(round(op[0])), int(round(op[1]))
            for dr in range(2):
                for dc in range(2):
                    r, ci = or_+dr, oc+dc
                    if 0 <= r < self.size and 0 <= ci < self.size:
                        for c in range(3):
                            img[c, r, ci] = COLORS["obstacle"][c]

        # Agent
        ar, ac = self.agent_pos
        for c in range(3):
            img[c, ar, ac] = COLORS["agent"][c]

        return img


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4: VISUAL SEARCH WITH CUES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Biological basis: Visual search — finding a target among distractors — is a
# core function of the visual cortex and attention systems.  Adding unreliable
# cues models how animals must decide whether to trust sensory hints.
#
# Why it suits our paradigm: When cues are present, the optimal strategy is
# APPROACH (follow the cue).  When cues are absent or misleading, EXPLORE
# is better.  The model must decide whether to trust or ignore cues — a form
# of "pseudo free will" in strategy selection.

class VisualSearchWithCues(BaseGridEnv):

    def __init__(self, cfg: EnvConfig, variant_seed: Optional[int] = None):
        super().__init__(cfg, variant_seed)
        self.num_distractors = cfg.search_num_distractors
        self.cue_prob = cfg.search_cue_probability

        self.target_pos = None
        self.distractor_positions = []
        self.has_cue = False
        self.cue_positions = []  # Arrow cells pointing toward target

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        self.done = False
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.float32)

        # Agent at random position
        self.agent_pos = np.array([self.rng.randint(2, self.size-2),
                                    self.rng.randint(2, self.size-2)])

        # Target at random position (far from agent)
        while True:
            self.target_pos = np.array([self.rng.randint(1, self.size-1),
                                         self.rng.randint(1, self.size-1)])
            if np.linalg.norm(self.agent_pos - self.target_pos) > self.size * 0.3:
                break

        # Scatter distractors
        self.distractor_positions = []
        for _ in range(self.num_distractors):
            pos = np.array([self.rng.randint(1, self.size-1), self.rng.randint(1, self.size-1)])
            self.distractor_positions.append(pos)

        # Maybe add cue (arrow trail pointing toward target)
        self.has_cue = self.rng.rand() < self.cue_prob
        self.cue_positions = []
        if self.has_cue:
            # Create a sparse trail of "arrow" cells from a random point toward target
            start = np.array([self.rng.randint(self.size), self.rng.randint(self.size)], dtype=np.float32)
            direction = (self.target_pos.astype(float) - start)
            length = np.linalg.norm(direction)
            if length > 0:
                direction /= length
                for t in range(0, int(length), 3):  # Every 3 cells
                    cp = (start + direction * t).astype(int)
                    if self._in_bounds(cp):
                        self.cue_positions.append(cp.copy())

        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] = 1
        return self._render(), {"task": "visual_search"}

    def step(self, action: int) -> Tuple[np.ndarray, Dict, bool, Dict]:
        old_pos = self.agent_pos.copy()
        self.step_count += 1
        self._move_agent(action)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1

        found_target = np.linalg.norm(self.agent_pos - self.target_pos) < 1.5

        reward_info = self._compute_reward_info(action, old_pos, found_target)

        if found_target or self.step_count >= self.max_steps:
            self.done = True
            if self.step_count >= self.max_steps and not found_target:
                closeness = 1.0 - min(reward_info["dist_to_goal"] / (self.size * 1.4), 1.0)
                reward_info["sparse_reward"] = -1.0 + closeness * 0.7
                reward_info["timeout"] = True

        return self._render(), reward_info, self.done, {"task": "visual_search"}

    def _compute_reward_info(self, action, old_pos, found_target):
        dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        old_dist = np.linalg.norm(old_pos - self.target_pos)
        approach_delta = old_dist - dist_to_target

        dense_reward = -0.01
        if found_target:
            dense_reward = 1.0

        is_novel = self.visit_counts[self.agent_pos[0], self.agent_pos[1]] <= 1

        return {
            "dense_reward": dense_reward,
            "sparse_reward": 0.0,
            "done": self.done,
            "success": found_target,
            "timeout": False,
            "is_novel_cell": is_novel,
            "visit_count": self.visit_counts[self.agent_pos[0], self.agent_pos[1]],
            "dist_to_goal": dist_to_target,
            "approach_delta": approach_delta,
            "threat_nearby": False,
            "threat_dist": float("inf"),
            "agent_pos": self.agent_pos.copy(),
            "step": self.step_count,
            "has_cue": self.has_cue,
        }

    def _render(self) -> np.ndarray:
        img = np.zeros((3, self.size, self.size), dtype=np.float32)
        for c in range(3):
            img[c] = COLORS["background"][c]

        # Visited
        visited_mask = self.visit_counts > 0
        for c in range(3):
            img[c][visited_mask] = COLORS["visited"][c]

        # Distractors
        for dp in self.distractor_positions:
            for c in range(3):
                img[c, dp[0], dp[1]] = COLORS["distractor"][c]

        # Target (hidden — looks like a distractor!)
        tr, tc = self.target_pos
        for c in range(3):
            img[c, tr, tc] = COLORS["distractor"][c]  # Indistinguishable!

        # Cue arrows (if present)
        for cp in self.cue_positions:
            for c in range(3):
                img[c, cp[0], cp[1]] = COLORS["cue_arrow"][c]

        # Agent
        ar, ac = self.agent_pos
        for c in range(3):
            img[c, ar, ac] = COLORS["agent"][c]

        return img


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════
# Runs N copies of an environment in parallel for efficient rollout collection.

class VectorizedEnv:
    """Run N independent copies of an environment for batched rollouts."""

    def __init__(self, env_class, cfg: EnvConfig, num_envs: int, base_seed: int = 0):
        self.envs = [env_class(cfg, variant_seed=base_seed + i) for i in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions):
        obs_list, reward_infos, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if env.done:
                obs, info = env.reset()
                obs_list.append(obs)
                reward_infos.append({"dense_reward": 0, "sparse_reward": 0, "is_novel_cell": False,
                                     "visit_count": 0, "dist_to_goal": 0, "approach_delta": 0,
                                     "threat_nearby": False, "threat_dist": float("inf"),
                                     "agent_pos": env.agent_pos.copy(), "step": 0,
                                     "done": False, "success": False, "timeout": False})
                dones.append(False)
                infos.append(info)
            else:
                obs, ri, done, info = env.step(actions[i])
                obs_list.append(obs)
                reward_infos.append(ri)
                dones.append(done)
                infos.append(info)
        return np.stack(obs_list), reward_infos, dones, infos


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

ENV_REGISTRY = {
    "morris_water_maze": MorrisWaterMaze,
    "visual_foraging": VisualForaging,
    "dynamic_obstacles": DynamicObstacleCourse,
    "visual_search": VisualSearchWithCues,
}


def make_env(task_name: str, cfg: EnvConfig, variant_seed: Optional[int] = None):
    """Create a single environment instance by name."""
    return ENV_REGISTRY[task_name](cfg, variant_seed)


def make_vectorized_env(task_name: str, cfg: EnvConfig, num_envs: int, base_seed: int = 0):
    """Create a vectorized environment by name."""
    return VectorizedEnv(ENV_REGISTRY[task_name], cfg, num_envs, base_seed)
