# Self-Optimizing Training Paradigm — Project Notes

## For Claude Code: Read this first before making any changes.

This document summarizes the design decisions, iteration history, and current
state of the experiment. It was produced after ~10 rounds of iteration between
Mihir and Claude on claude.ai.

---

## Hypothesis

A model that **chooses its own optimization function** during training — via a
meta-controller inspired by the prefrontal cortex — will be more generalizable
across tasks than a model trained with a single fixed objective.

## Architecture

**Augmented model (PFC):** CNN encoder → latent state → meta-controller selects
one of 5 sub-objectives (EXPLORE, APPROACH, AVOID, EXPLOIT, MEMORIZE) → action
policy conditioned on [latent + sub-objective embedding] → action.

**Baseline model:** Same CNN encoder → latent state → action policy → action.
Same parameter count (~185k each, differing by only 22 params).

## Key Design Decisions (arrived at through iteration)

### 1. Design B: Dense + Scaled Intrinsic Rewards
**Both models receive the same dense task reward.** The augmented model
additionally gets intrinsic reward from the selected sub-objective, scaled down.

- Action policy reward = `dense_reward + 0.1 × intrinsic_reward`
- Meta-controller reward = `sparse_failure_penalty + 0.2 × intrinsic_reward`
- The meta-controller does NOT get dense reward

**Why:** The original design (intrinsic-only for the action policy) made the
augmented model solve a much harder optimization problem. The PFC doesn't
replace the reward system, it supplements it. Dense reward is the "VTA →
striatum" pathway; intrinsic reward is the "PFC top-down modulation."

**Critical:** `intrinsic_reward_scale = 0.1` in config.py. Without this,
intrinsic rewards (~0.1/step) drown out dense rewards (~0.01/step) and the
policy can't learn the task.

### 2. Temporal Commitment (meta_decision_interval = 15)
The meta-controller commits to a sub-objective for 15 steps before
reconsidering. Without this:
- Per-step strategy switching creates a feedback loop (meta shifts → policy
  destabilizes → meta re-evaluates incorrectly → repeat)
- Credit assignment over 300 timesteps with sparse reward is impossible
- Biologically: PFC executive control operates on seconds, not milliseconds

### 3. Grid Size 20 (not 32)
The original 32×32 grid made exploration intractable. A 20×20 pool has ~254
cells vs ~616, and 300-step episodes can cover meaningful space.

### 4. Morris Water Maze Enhancements
- **4 landmark cues** at pool edges (N=red, S=blue, E=yellow, W=cyan) —
  matches the real experimental protocol
- **Proximity gradient** (70% pool radius, 30% color shift) — warm tint near
  the hidden platform gives the CNN a learnable signal
- **Distance-scaled failure penalty** — closer approach → milder timeout penalty

### 5. NaN Gradient Protection
When the model reaches high success (very short episodes), reward spikes cause
gradient explosions. Three layers of defense:
- Advantage clamping to [-5, 5]
- Ratio clamping to [0.01, 100]
- **Gradient norm check after backward() but before optimizer.step()** — if
  `clip_grad_norm_` returns NaN/Inf, zero gradients and skip the step
- try/except around entire PPO batch processing

**Current status: The NaN protection may still be insufficient.** The last run
crashed with NaN during rollout collection (meaning weights were corrupted
despite protections). The grad_norm check was the most recent fix and hasn't
been tested yet.

## Current State & What Needs to Happen

### What works:
- Single-task training on Morris water maze: augmented model reaches 96%
  success, slightly outperforming baseline (88%)
- Meta-controller learns meaningful strategy specialization (obj_ent ~0.7-0.9)
- Parameter parity is excellent (22 param difference)

### What's broken:
- **Full experiment crashes with NaN** during sequential multi-task training.
  The model learns morris_water_maze well, then NaN appears when episodes become
  very short (avg_len=8). The grad_norm NaN check was just added and needs testing.
- Three of four tasks (visual_foraging, dynamic_obstacles, visual_search) have
  never been verified to work individually. They may have their own issues.

### Recommended next steps:
1. Run `python run_experiment.py --mode full --episodes 3000 --device cuda`
   and see if the NaN fix holds
2. If NaN persists, consider: reducing learning rate when success is high,
   or adding a learning rate scheduler that decays as performance improves
3. If full experiment completes, examine the evaluation report in
   `results/evaluation_report.json` — the key metrics are multi-task average
   success, zero-shot transfer, and strategy diversity across tasks
4. Test each task individually if needed:
   `python run_experiment.py --mode single_task --task <name> --episodes 2000`
5. The visual_foraging task requires collecting 6/8 food while avoiding 3
   predators — may need tuning (predator speed, food count, etc.)
6. The dynamic_obstacles task had 0% success for the baseline at 3000 episodes
   — the obstacle density or speed may be too high

## File Structure

- `config.py` — All hyperparameters (env, model, training, sub-objectives)
- `environments.py` — 4 grid-world environments (Morris maze, foraging,
  obstacles, visual search)
- `models.py` — AugmentedModel and BaselineModel (shared CNN encoder)
- `sub_objectives.py` — Intrinsic reward library (EXPLORE/APPROACH/AVOID/
  EXPLOIT/MEMORIZE)
- `training.py` — PPO training loops for both paradigms (two-level for
  augmented, standard for baseline)
- `evaluate.py` — 5 generalizability metrics
- `run_experiment.py` — Main entry point (smoke_test / single_task / full)

## Hardware
- NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)
- Both models combined use < 5 MB VRAM
- Single-task runs take ~2-4 minutes, full experiment ~30-40 minutes