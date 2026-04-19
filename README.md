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
goal-directed approach, or exploitation of known rewards — while the underlying
reward circuitry continues to function normally. We replicate this two-level
architecture: a shared reward signal trains both models equally, while the
augmented model's meta-controller adds a supplementary intrinsic signal that
varies by context.

---

## Results

The augmented model **consistently outperforms the baseline across all tasks
and all seeds** under strictly fair experimental conditions.

### Multi-task Success Rate (3-seed average)

| Task | Augmented | Baseline | Delta |
|---|---|---|---|
| Morris Water Maze | 0.67 | 0.21 | **+45.7%** |
| Visual Foraging | 0.73 | 0.49 | **+24.3%** |
| Dynamic Obstacles | 0.74 | 0.47 | **+27.8%** |
| Visual Search | 0.14 | 0.08 | **+6.0%** |
| **Overall** | **0.572** | **0.313** | **+25.9%** |

Wins on **12 out of 12** task x seed comparisons. Per-seed deltas: +11.8%,
+36.8%, +29.2%.

### Other Metrics

- **Zero-shot transfer**: Augmented 0.53 avg vs baseline 0.32 avg on unseen
  task variants
- **Few-shot adaptation**: Augmented reaches 70% threshold in 20 episodes;
  baseline takes 40-200 or fails
- **Catastrophic forgetting**: Low for both (augmented avg 2.4%)
- **Strategy diversity**: Entropy 0.6-1.1 across tasks, all 3 sub-objectives
  getting meaningful usage

### Fairness Guarantees

The experiment was subjected to a comprehensive fairness audit. Both models:
- Train with identical episode budgets (20,000 episodes each)
- Use matched parameter counts (231,050 vs 231,590, 0.998x ratio)
- Train multi-task interleaved with shared model and round-robin scheduling
- Use identical seeds, evaluation protocols (`deterministic=True`), and
  best-model checkpoint restoration
- Receive the same dense task reward from environments

The only difference is the augmented model's supplementary intrinsic reward
(the experimental intervention).

---

## Experimental Design

### Two Models (Design B)

Both models receive **the same dense task reward**. The augmented model
additionally receives a scaled intrinsic reward from its meta-controller's
selected sub-objective.

| Component | Augmented (PFC) | Baseline (Fixed) |
|-----------|----------------|------------------|
| CNN encoder | Shared architecture | Same architecture |
| Dense task reward | Yes | Yes |
| Meta-controller | GRU-based, selects sub-objective | -- |
| Intrinsic reward | Yes (0.08x scale) | -- |
| Sub-objective conditioning | Policy sees [latent; obj_embedding] | Policy sees [latent] |
| Parameters | ~231,050 | ~231,590 |

The augmented model's action policy reward is:

```
reward = dense_task_reward + 0.08 * intrinsic_reward(selected_sub_objective)
```

The meta-controller is trained separately with:

```
meta_reward = sparse_failure_penalty + 0.2 * intrinsic_reward(selected_sub_objective)
```

The meta-controller receives **no dense task reward**. It must discover which
sub-objectives are productive purely from whether they avoid failure and
generate useful intrinsic signals.

---

## Architecture

```
AUGMENTED MODEL                           BASELINE MODEL

Observation --> CNN Encoder               Observation --> CNN Encoder
                    |                                          |
              Latent State (256)                        Latent State (256)
               +----+----+                                    |
    Meta-       |         |                              Action Policy
  Controller   Embed(32) Action                          (wider hidden
  (GRU-based)     |      Policy                           for parity)
       |     [latent;embed]                                   |
  Selects k    = 288-dim                                   Action
  (every 16        |
   steps)      Action

Meta-ctrl loss: sparse + 0.2*intrinsic
Policy loss:    dense + 0.08*intrinsic    Policy loss: dense only
```

### GRU-Based Recurrent Meta-Controller

The meta-controller uses a `GRUCell` to integrate information over time,
mirroring how the biological PFC maintains working memory. The GRU hidden
state accumulates a summary of what the agent has seen and done, enabling
decisions like "I've been exploring for 40 steps with no progress -- switch
to approach."

- Hidden state resets on episode boundaries
- Per-step hidden states stored in the rollout buffer for correct PPO
  re-evaluation after mini-batch shuffling

### Temporal Commitment

The meta-controller selects a sub-objective every **16 steps**, not every step.
Between selections, the chosen strategy is held constant. This design choice
has three motivations:

1. **Credit assignment**: Selecting at every step creates 300 decisions per
   episode with sparse feedback -- an impossible credit assignment problem.
   At 16-step intervals, there are ~18 decisions per episode.
2. **Policy stability**: Per-step switching destabilizes the action policy,
   which is conditioned on the sub-objective embedding. The policy needs time
   to execute a coherent strategy before the strategy changes.
3. **Biological fidelity**: PFC executive control operates on timescales of
   seconds, not milliseconds.

### Sub-Objective Library

The meta-controller selects from three sub-objectives, each with a
hand-designed intrinsic reward function:

| Sub-Objective | Intrinsic Reward Signal | Biological Analogue |
|---------------|------------------------|---------------------|
| **EXPLORE** | Visiting novel grid cells | Dopaminergic novelty signal |
| **APPROACH** | Decreasing distance to salient targets | Goal-directed approach (mesolimbic) |
| **EXPLOIT** | Proximity to goal ("stay and harvest") | Habit formation (dorsal striatum) |

Reduced from 5 to 3 sub-objectives (removed AVOID and MEMORIZE) based on
empirical analysis showing they were consistently underused (<5% and <18%
selection rates respectively). Fewer sub-objectives means each mode gets 1/3
of training data instead of 1/5, and the meta-controller's credit assignment
is simpler (3 choices vs 5).

---

## Task Suite

All tasks are 20x20 grid-worlds rendered as 3-channel RGB images. They are
designed as lightweight analogues of neuroscience experiments that benefit from
flexible strategy selection.

### 1. Morris Water Maze

The agent is placed at a random edge of a circular pool and must find a hidden
platform. A subtle proximity gradient provides a learnable signal. Four colored
landmark cues at the pool edges provide allocentric spatial reference, matching
the real experimental protocol.

### 2. Visual Foraging

Collect food items scattered across the environment while avoiding moving
predator zones. Requires balancing exploration and exploitation.

### 3. Dynamic Obstacle Course

Navigate from the bottom-left to the top-right through a field of moving
obstacles. Requires reactive path planning.

### 4. Visual Search with Cues

Locate a hidden target among distractors. Colored arrow trail cues guide
the agent toward the target.

---

## Generalizability Evaluation

The experiment measures five dimensions:

1. **Multi-task performance** -- Average success rate across all four tasks.
   Both models use one shared set of weights trained interleaved across tasks.

2. **Zero-shot transfer** -- Performance on unseen task variants (new platform
   positions, different obstacle patterns) with no additional training.

3. **Few-shot adaptation** -- Episodes needed to reach a threshold success rate
   on a novel variant.

4. **Catastrophic forgetting** -- Performance degradation on earlier tasks after
   continued training on all tasks.

5. **Strategy diversity** -- Entropy of the meta-controller's sub-objective
   selections per task.

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

### Full experiment (~15-20 min, CPU)

```bash
python run_experiment.py --mode full --seed 42
```

### Multi-seed run (for robustness)

```bash
python run_experiment.py --mode full --seed 42 --results-dir results_seed42
python run_experiment.py --mode full --seed 7 --results-dir results_seed7
python run_experiment.py --mode full --seed 123 --results-dir results_seed123
```

### View results

Results are saved to `results/` (or `--results-dir`):
- `comparison_<task>.png` -- Training curves per task
- `final_comparison.png` -- Bar chart of final success rates
- `strategy_distribution.png` -- Meta-controller sub-objective usage per task
- `evaluation_report.json` -- Full evaluation metrics

---

## Key Design Decisions & Rationale

| Decision | What | Why |
|----------|------|-----|
| Design B | Both models get dense reward | Isolates the meta-controller as the experimental variable |
| 3 sub-objectives | EXPLORE, APPROACH, EXPLOIT | Each mode gets 1/3 of data; removed dead AVOID/MEMORIZE |
| GRU meta-controller | Recurrent strategy selection | Temporal integration prevents entropy collapse |
| Intrinsic scale 0.08x | `reward = dense + 0.08 * intrinsic` | Supplements rather than dominates dense signal |
| Temporal commitment | Meta-controller decides every 16 steps | ~18 decisions/episode makes credit assignment tractable |
| Embed dim 32 | 32-dim sub-objective embeddings | 11% of conditioned vector (288-dim), strong enough to differentiate modes |
| Parameter parity | 540-param difference (0.998x ratio) | Ensures performance differences come from the paradigm, not capacity |

---

## File Structure

```
├── README.md              <- This file
├── NOTES.md               <- Iteration history & debugging notes
├── session_log.md         <- Development session narrative
├── config.py              <- All hyperparameters
├── environments.py        <- 4 grid-world task environments
├── models.py              <- AugmentedModel + BaselineModel architectures
├── sub_objectives.py      <- Sub-objective library & intrinsic rewards
├── training.py            <- PPO training loops (two-level + standard)
├── evaluate.py            <- 5 generalizability evaluation metrics
└── run_experiment.py      <- Main entry point (smoke_test / single_task / full)
```
