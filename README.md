# Self-Optimizing Training Paradigm Experiment

## Hypothesis

Introducing a training paradigm where a model **chooses its own optimization
function** during training — receiving only sparse negative punishment for
violating a broad goal — produces models that are **more generalizable** across
tasks than models trained with a single fixed objective.

**Biological inspiration:** The prefrontal cortex (PFC) is the most
generalizable region of the human brain. It does not optimize for a single
fixed reward signal. Instead, it dynamically selects *what to optimize for*
in the short term while pursuing broader underlying goals. We replicate this
by giving our "augmented" model a **meta-controller** (the PFC analogue) that
selects among a library of sub-objectives, trained only via sparse negative
feedback.

---

## Experimental Design

### Two Models

| Model | Description |
|-------|-------------|
| **Augmented (PFC)** | Has a meta-controller that selects sub-objectives. Trained with sparse negative-only feedback on the broad task goal. |
| **Baseline (Fixed)** | Same backbone. Trained with a single fixed dense reward function per task. |

### Task Suite (Biologically Inspired)

All tasks are lightweight 2D grid-worlds (32×32) rendered as small images.
They are designed to be analogous to neuroscience experiments and to benefit
from flexible, context-dependent strategy selection.

1. **Morris Water Maze** — Spatial navigation to a hidden platform. The agent
   cannot see the goal and must explore, then remember. Directly inspired by
   the classic rodent neuroscience experiment.

2. **Visual Foraging** — Find and collect food items scattered in an
   environment with moving predator zones. Requires balancing exploration,
   exploitation, and threat avoidance.

3. **Dynamic Obstacle Course** — Navigate from start to goal through a field
   of moving obstacles. Requires reactive planning and path adaptation.

4. **Visual Search with Cues** — Locate a hidden target in a cluttered field.
   Partial visual cues (color gradients, arrows) may or may not be present.
   Tests attention and cue integration.

### Generalizability Evaluation

- **Multi-task average** — Performance across all four tasks.
- **Zero-shot transfer** — Performance on unseen task variants (rotated mazes,
  new obstacle patterns, shifted cue locations).
- **Few-shot adaptation** — Episodes needed to reach threshold performance on
  a new variant.
- **Catastrophic forgetting** — Performance on early tasks after training on
  later tasks.
- **Strategy diversity** — Entropy of the meta-controller's sub-objective
  selections (augmented model only).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  AUGMENTED MODEL                     │
│                                                      │
│  Observation ──► CNN Encoder ──► Latent State         │
│                                     │                │
│                          ┌──────────┴──────────┐     │
│                          │  Meta-Controller     │     │
│                          │  (PFC Analogue)      │     │
│                          │  Selects sub-obj k   │     │
│                          └──────────┬──────────┘     │
│                                     │                │
│              Sub-objective k ───► Intrinsic Reward    │
│                                     │                │
│                          ┌──────────┴──────────┐     │
│                          │  Action Policy       │     │
│                          │  π(a|s, obj_k)       │     │
│                          └─────────────────────┘     │
│                                                      │
│  Meta-controller loss: sparse negative only          │
│  Action policy loss:   intrinsic reward from obj_k   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  BASELINE MODEL                      │
│                                                      │
│  Observation ──► CNN Encoder ──► Latent State         │
│                                     │                │
│                          ┌──────────┴──────────┐     │
│                          │  Action Policy       │     │
│                          │  π(a|s)              │     │
│                          └─────────────────────┘     │
│                                                      │
│  Policy loss: dense fixed reward per task            │
└─────────────────────────────────────────────────────┘
```

### Sub-Objective Library

The meta-controller selects from these learned sub-objectives:

| Sub-Objective | Intrinsic Reward | Biological Analogue |
|---------------|-----------------|---------------------|
| **EXPLORE** | Reward for visiting novel states | Curiosity / dopaminergic novelty signal |
| **APPROACH** | Reward for moving toward salient targets | Goal-directed approach behavior |
| **AVOID** | Reward for increasing distance from threats | Fear / amygdala-driven avoidance |
| **EXPLOIT** | Reward for repeating previously rewarded actions | Habit formation / dorsal striatum |
| **MEMORIZE** | Reward for building/using spatial memory | Hippocampal place cell consolidation |

---

## How to Run

### 1. Install dependencies

```bash
pip install torch numpy matplotlib tensorboard
```

### 2. Quick test (< 1 min, CPU)

```bash
python run_experiment.py --mode smoke_test
```

### 3. Single-task comparison

```bash
python run_experiment.py --mode single_task --task morris_water_maze --episodes 2000
```

### 4. Full experiment (all tasks, both models, evaluation)

```bash
python run_experiment.py --mode full --episodes 5000 --device cuda
```

### 5. View results

```bash
tensorboard --logdir runs/
# Or check the generated comparison plots in results/
```

---

## GPU Memory Budget

| Component | Estimated VRAM |
|-----------|---------------|
| CNN encoder (small) | ~10 MB |
| Meta-controller | ~5 MB |
| Action policy + value head | ~10 MB |
| Rollout buffer (256 steps × 64 envs) | ~50 MB |
| Gradient computation | ~100 MB |
| **Total per model** | **< 200 MB** |
| **Both models + overhead** | **< 1 GB** |

Fits comfortably within an 8 GB VRAM budget.

---

## File Structure

```
self_optimizing_experiment/
├── README.md              ← You are here
├── requirements.txt       ← Dependencies
├── config.py              ← All hyperparameters
├── environments.py        ← 4 grid-world task environments
├── models.py              ← Augmented + Baseline architectures
├── sub_objectives.py      ← Sub-objective library & intrinsic rewards
├── training.py            ← PPO training loop for both paradigms
├── evaluate.py            ← Generalizability evaluation metrics
└── run_experiment.py      ← Main entry point
```
