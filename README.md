# Self-Optimizing Training Paradigm Experiment

## Hypothesis

A model equipped with a **meta-controller that selects its own optimization
sub-objective** — receiving supplementary intrinsic rewards on top of standard
task rewards — will develop more generalizable behavior across tasks than a
model trained with a fixed objective alone.

**Biological inspiration:** The prefrontal cortex (PFC) does not replace the
brain's reward system. It *modulates* it. Dopaminergic signals from the VTA
reach both the striatum (basic reward learning) and the PFC (executive
control). The PFC selects what to optimize for in the short term — curiosity,
threat avoidance, exploitation of known rewards — while the underlying reward
circuitry continues to function normally. We replicate this two-level
architecture: a shared reward signal trains both models equally, while the
augmented model's meta-controller adds a supplementary intrinsic signal that
varies by context.

---

## Experimental Design

### Two Models (Design B)

Both models receive **the same dense task reward**. The augmented model
additionally receives a scaled intrinsic reward from its meta-controller's
selected sub-objective.

| Component | Augmented (PFC) | Baseline (Fixed) |
|-----------|----------------|------------------|
| CNN encoder | Shared architecture | Same architecture |
| Dense task reward | ✓ | ✓ |
| Meta-controller | Selects sub-objective | — |
| Intrinsic reward | ✓ (0.1× scale) | — |
| Sub-objective conditioning | Policy sees [latent + obj_embedding] | Policy sees [latent] |
| Parameters | ~185,850 | ~185,830 |

The augmented model's action policy reward is:

```
reward = dense_task_reward + 0.1 × intrinsic_reward(selected_sub_objective)
```

The meta-controller is trained separately with:

```
meta_reward = sparse_failure_penalty + 0.2 × intrinsic_reward(selected_sub_objective)
```

The meta-controller receives **no dense task reward**. It must discover which
sub-objectives are productive purely from whether they avoid failure and
generate useful intrinsic signals.

### Why Design B?

An earlier design (Design A) gave the augmented model *only* intrinsic and
sparse rewards, withholding the dense task reward entirely. This made the
augmented model solve a fundamentally harder optimization problem than the
baseline, producing misleading comparisons. Design B isolates the experimental
variable: both models learn from the same task signal, and the only difference
is the supplementary self-optimization machinery.

---

## Architecture

```
AUGMENTED MODEL                           BASELINE MODEL

Observation ──► CNN Encoder               Observation ──► CNN Encoder
                    │                                          │
              Latent State                              Latent State
               ┌────┴────┐                                    │
    Meta-       Action                                   Action
  Controller    Policy                                   Policy
  (PFC)         π(a|s,k)                                 π(a|s)
       │            │                                        │
  Selects k    Conditioned                               Action
  (every 15    on sub-obj k
   steps)          │
               Action

Meta-ctrl loss: sparse + 0.2×intrinsic
Policy loss:    dense + 0.1×intrinsic     Policy loss: dense only
```

### Temporal Commitment

The meta-controller selects a sub-objective every **15 steps**, not every step.
Between selections, the chosen strategy is held constant. This design choice
has three motivations:

1. **Credit assignment**: Selecting at every step creates 300 decisions per
   episode with sparse feedback — an impossible credit assignment problem.
   At 15-step intervals, there are ~20 decisions per episode.
2. **Policy stability**: Per-step switching destabilizes the action policy,
   which is conditioned on the sub-objective embedding. The policy needs time
   to execute a coherent strategy before the strategy changes.
3. **Biological fidelity**: PFC executive control operates on timescales of
   seconds, not milliseconds.

### Sub-Objective Library

The meta-controller selects from five sub-objectives, each with a hand-designed
intrinsic reward function:

| Sub-Objective | Intrinsic Reward Signal | Biological Analogue |
|---------------|------------------------|---------------------|
| **EXPLORE** | Visiting novel grid cells | Dopaminergic novelty signal |
| **APPROACH** | Decreasing distance to salient targets | Goal-directed approach (mesolimbic) |
| **AVOID** | Increasing distance from threats | Amygdala-driven fear response |
| **EXPLOIT** | Revisiting previously rewarded locations | Habit formation (dorsal striatum) |
| **MEMORIZE** | Building/using spatial memory coverage | Hippocampal place cell consolidation |

These are not learned reward models — they are fixed functions with clear
semantics, enabling analysis of which strategies the meta-controller selects
in different contexts.

---

## Task Suite

All tasks are 20×20 grid-worlds rendered as 3-channel RGB images. They are
designed as lightweight analogues of neuroscience experiments that benefit from
flexible strategy selection.

### 1. Morris Water Maze

The agent is placed at a random edge of a circular pool and must find a hidden
platform. The platform is invisible, but a subtle proximity gradient (warm
color tint) provides a learnable signal. Four colored landmark cues at the pool
edges (N=red, S=blue, E=yellow, W=cyan) provide allocentric spatial reference,
matching the real experimental protocol.

**Why it suits the paradigm**: Success requires switching from EXPLORE (find
the platform) to MEMORIZE (remember where it is) to EXPLOIT (swim directly
there on future trials).

### 2. Visual Foraging

Collect food items scattered across the environment while avoiding moving
predator zones. Requires balancing exploration, exploitation, and threat
avoidance.

**Why it suits the paradigm**: The optimal strategy shifts dynamically — EXPLOIT
when food is nearby, EXPLORE when it's scarce, AVOID when a predator approaches.

### 3. Dynamic Obstacle Course

Navigate from the bottom-left to the top-right through a field of moving
obstacles. Requires reactive path planning.

**Why it suits the paradigm**: Moment-to-moment strategy switching between AVOID
(dodge incoming obstacle) and APPROACH (move toward goal through a gap).

### 4. Visual Search with Cues

Locate a hidden target among distractors. Partial cues (colored arrow trails)
may or may not be present. Tests whether the model can learn to trust or ignore
cues based on context.

**Why it suits the paradigm**: When cues are present, APPROACH is optimal.
When absent, EXPLORE is better. The model must decide which strategy to deploy.

---

## Generalizability Evaluation

The experiment measures five dimensions:

1. **Multi-task performance** — Average success rate across all four tasks.
   The augmented model uses one set of weights for all tasks; the baseline
   trains a separate model per task.

2. **Zero-shot transfer** — Performance on unseen task variants (new platform
   positions, different obstacle patterns) with no additional training.

3. **Few-shot adaptation** — Episodes needed to reach a threshold success rate
   on a novel variant.

4. **Catastrophic forgetting** — Performance degradation on earlier tasks after
   training on later tasks (augmented model only, since it trains sequentially).

5. **Strategy diversity** — Entropy of the meta-controller's sub-objective
   selections per task. Different tasks should produce different strategy
   distributions.

---

## Running the Experiment

### Install

```bash
pip install torch numpy matplotlib
```

### Quick test (~1 min, CPU)

```bash
python run_experiment.py --mode smoke_test
```

### Single-task comparison (~3 min on GPU)

```bash
python run_experiment.py --mode single_task --task morris_water_maze --episodes 2000 --device cuda
```

### Full experiment (~30-40 min on GPU)

```bash
python run_experiment.py --mode full --episodes 3000 --device cuda
```

### View results

Results are saved to `results/`:
- `comparison_<task>.png` — Training curves per task
- `final_comparison.png` — Bar chart of final success rates
- `strategy_distribution.png` — Meta-controller sub-objective usage per task
- `evaluation_report.json` — Full evaluation metrics

---

## GPU Memory Budget

| Component | Estimated VRAM |
|-----------|---------------|
| CNN encoder (4 conv layers) | ~8 MB |
| Meta-controller + embeddings | ~3 MB |
| Action policy + value heads | ~8 MB |
| Rollout buffer (128 steps × 16 envs) | ~5 MB |
| Gradient computation | ~50 MB |
| **Total per model** | **< 100 MB** |
| **Both models + overhead** | **< 500 MB** |

Fits comfortably within an 8 GB VRAM budget.

---

## Numerical Stability

When the model reaches high success rates, episodes become very short (~7
steps), causing reward spikes within 128-step rollouts. Three layers of
protection prevent gradient explosions:

1. **Advantage clamping** to [-5, 5]
2. **Importance ratio clamping** to [0.01, 100]
3. **Gradient norm check** after `backward()` — if `clip_grad_norm_` returns
   NaN or Inf, the gradients are zeroed and `optimizer.step()` is skipped,
   preventing NaN from being written into model weights.

---

## Key Design Decisions & Rationale

| Decision | What | Why |
|----------|------|-----|
| Design B | Both models get dense reward | Isolates the meta-controller as the experimental variable |
| Intrinsic scale 0.1× | `reward = dense + 0.1 × intrinsic` | Raw intrinsic (~0.1/step) drowns out dense (~0.01/step) without scaling |
| Temporal commitment | Meta-controller decides every 15 steps | Credit assignment, policy stability, biological fidelity |
| Grid size 20 | Down from 32 | 32×32 made exploration intractable within 300-step episodes |
| Proximity gradient | 70% pool radius, 30% color shift | CNN needs a learnable visual signal; invisible platform has no gradient |
| Landmark cues | 4 colored markers at pool edges | Provides allocentric spatial reference (matches real Morris maze protocol) |
| Parameter parity | 22-param difference | Ensures performance differences come from the paradigm, not capacity |

---

## File Structure

```
├── README.md              ← This file
├── NOTES.md               ← Iteration history & debugging notes
├── requirements.txt       ← Dependencies
├── config.py              ← All hyperparameters
├── environments.py        ← 4 grid-world task environments
├── models.py              ← AugmentedModel + BaselineModel architectures
├── sub_objectives.py      ← Sub-objective library & intrinsic rewards
├── training.py            ← PPO training loops (two-level + standard)
├── evaluate.py            ← 5 generalizability evaluation metrics
└── run_experiment.py      ← Main entry point (smoke_test / single_task / full)
```

---

## Known Issues & Next Steps

- **Meta-controller entropy collapse**: During sequential multi-task training,
  the meta-controller can lock onto a single sub-objective after the first task
  and lose the ability to select different strategies. A higher entropy floor
  or entropy reset between tasks may be needed.
- **Task difficulty**: Visual foraging, dynamic obstacles, and visual search
  may need parameter tuning (fewer/slower hazards, stronger cues) to be
  learnable within 3000 episodes.
- See `NOTES.md` for full iteration history.
