# Lab Notebook

A dated record of what was tried in this project, what it produced, and what
changed as a result. Entries are in chronological order and each one ends with
the state the repository was left in. Several entries record results that were
later shown to be artifacts; those are kept, because the reason they were wrong
is the most useful part of the record.

Standing design decisions and the evidence behind them live in
[`NOTES.md`](NOTES.md). Configuration values quoted below are the values in
`config.py` at that date, recovered from the commit history.

---

## 2026-03-15 → 2026-03-28 — Initial build

First working version of the pipeline: four grid-world tasks, a shared CNN
encoder, a feedforward meta-controller over **five** sub-objectives (EXPLORE,
APPROACH, AVOID, EXPLOIT, MEMORIZE), and PPO for both models.

- `grid_size = 32`, `objective_embed_dim = 16`, `total_episodes = 5000` per task.
- The action policy was trained on intrinsic reward **only** (later called
  Design A).

**Outcome:** trains end to end, but the augmented model learns almost nothing.
Design A asks it to solve a strictly harder problem than the baseline — the
baseline gets the task reward and the augmented model does not.

---

## 2026-04-05 — Grid size, Design B, and the first Morris Water Maze results

Three changes, all driven by the Design A failure.

**1. Grid 32×32 → 20×20.** A 32×32 pool has ~616 reachable cells; 300-step
episodes cannot cover enough of it for a hidden platform to be found by chance,
so there was no learning signal to bootstrap from. 20×20 gives ~254 cells.

**2. Design A → Design B.** Both models now receive the same dense task reward;
the augmented model *additionally* receives the selected sub-objective's
intrinsic reward:

```
policy reward = dense_task_reward + intrinsic_reward_scale * intrinsic(selected)
meta reward   = sparse_failure_penalty + 0.2 * intrinsic(selected)
```

The reasoning is biological as well as practical: the PFC does not replace the
striatal reward system, it modulates it. `intrinsic_reward_scale = 0.1` was set
because unscaled intrinsic rewards (~0.1/step) are an order of magnitude larger
than dense rewards (~0.01/step) and drown out the task gradient entirely.

**3. Temporal commitment introduced** at `meta_decision_interval = 15`. Per-step
strategy switching produced a feedback loop — the meta-controller shifts, the
conditioned policy destabilizes, the meta-controller re-evaluates against a
policy that has not settled — on top of a credit-assignment problem with ~300
decisions per episode against a sparse signal.

Morris Water Maze support was also fleshed out to match the real protocol: four
coloured landmark cues at the pool edges for allocentric reference, a proximity
gradient (warm tint near the platform) to give the CNN something learnable, and
a distance-scaled timeout penalty.

**Outcome:** single-task Morris Water Maze works — the augmented model reaches
high success and edges out the baseline. The other three tasks were not yet
verified individually.

**Open problem recorded at the time:** the full multi-task run crashed with NaN
once episodes became very short (avg length ~8). Three defenses were added —
advantage clamping to [-5, 5], PPO ratio clamping to [0.01, 100], and a
gradient-norm check between `backward()` and `optimizer.step()` that zeroes
gradients and skips the step if `clip_grad_norm_` returns NaN/Inf. All three are
still in `training.py`. The crash has not recurred since.

---

## 2026-04-12 — GRU meta-controller

Hyperparameter tuning had flattened out at roughly +3–4 pp over baseline and the
sign of the delta depended on the seed. The suspected structural cause: a
feedforward meta-controller makes every decision statelessly, with no memory of
whether the strategy it picked 40 steps ago was working.

Replaced it with a `GRUCell` variant (`GRUCell` rather than `nn.GRU` because
rollouts step one observation at a time):

```python
class MetaController(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_objectives):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, num_objectives))

    def forward(self, latent, hidden):
        h = self.gru(latent, hidden)
        return Categorical(logits=self.head(h)), h
```

This touched the whole pipeline. The subtle part is PPO re-evaluation: mini-batch
shuffling destroys temporal order, so `RolloutBuffer` stores the hidden state
*as it was before* each step's forward pass, and `evaluate_actions()` takes those
stored states as an argument. Hidden state resets on episode boundaries in both
training and evaluation. Baseline `wider_hidden` was re-tuned to hold parameter
parity. `meta_decision_interval` moved 15 → 8 and `intrinsic_reward_scale`
0.1 → 0.08 in the same pass.

**Result as measured that day:**

| Task | Augmented (GRU) | Baseline | Delta (pp) |
|---|---|---|---|
| Morris Water Maze | 1.00 | 0.87 | +13 |
| Visual Foraging | 0.96 | 0.64 | +32 |
| Dynamic Obstacles | 0.87 | 0.89 | -2 |
| Visual Search | 0.35 | 0.06 | +29 |
| **Average** | **0.795** | **0.615** | **+18.0** |

Strategy entropy held at ~1.43–1.47 across tasks rather than collapsing, which
was the mechanism-level effect the GRU was introduced to produce.

**This +18.0 pp number is not real.** See the next entry. The entropy result
survived the audit; the performance result did not.

---

## 2026-04-17 — Fairness audit, and the null result

The GRU result looked too good, which prompted an audit of the whole pipeline
for asymmetries between the two arms rather than a search for more gains.

**Six issues found:**

| # | Severity | Issue |
|---|---|---|
| 1 | Major | Augmented model trained multi-task (one shared model, interleaved over 4 tasks); baseline trained single-task (one model per task). Multi-task and transfer metrics favour multi-task training regardless of the meta-controller. |
| 2 | Moderate | Augmented model evaluated with `deterministic=False`, baseline with `deterministic=True`. Different evaluation protocols. |
| 3 | Moderate | Baseline training envs used `seed + 1000`, augmented used `seed` — different task layouts. |
| 4 | Moderate | Augmented model always restored its best checkpoint; the baseline only restored on an early-stopping trigger, so a baseline that trained to completion kept possibly-degraded final weights. |
| 5 | Minor | Few-shot adaptation was broken: an optimizer was constructed but never used and the loop ran under `torch.no_grad()`, so it silently measured zero-shot performance. |
| 6 | Minor | Dead config parameter `success_signal = 0.5`, referenced nowhere. |

Every asymmetry in the list favoured the augmented model. Each was individually
defensible at the time it was introduced ("two-level optimization needs more
budget", "a stochastic meta-controller preserves strategy diversity"); together
they were a rigged comparison. An earlier fix in this same pass removed a `* 2.0`
budget multiplier that had been giving the augmented model ~10,000 episodes per
task against the baseline's 5,000; on its own that took the delta from +18 pp to
+6.2 pp.

**All six fixed:** baseline trains multi-task interleaved with a shared model and
round-robin scheduling, both models evaluate with `deterministic=True`, both use
`base_seed=cfg.seed`, both always restore the best checkpoint, and the dead
parameter is gone. The few-shot metric is still a fixed-weights measurement and
is labelled as such rather than as adaptation.

**Result with all asymmetries removed:**

| Task | Augmented | Baseline | Delta (pp) |
|---|---|---|---|
| Morris Water Maze | 0.19 | 0.32 | -13 |
| Visual Foraging | 0.59 | 0.75 | -16 |
| Dynamic Obstacles | 0.84 | 0.64 | +20 |
| Visual Search | 0.19 | 0.11 | +8 |
| **Average** | **0.453** | **0.455** | **-0.3** |

A null result. Strategy entropy also fell to 0.6–0.8, with the meta-controller
using essentially only EXPLORE and APPROACH.

**Conclusion recorded at the time:** under matched conditions, at this scale, on
these tasks, the paradigm as then configured gave no measurable advantage, and
the +18 pp figure had been an artifact of compounding experimental bias.

---

## 2026-04-19 — Reduce to three sub-objectives

Starting point: the null result above, plus the observation that two of the five
sub-objectives were barely being selected — AVOID under 5% and MEMORIZE 0–18%
across the four tasks.

Four changes in one pass:

- Sub-objectives 5 → 3 (dropped AVOID and MEMORIZE). Each surviving mode now
  gets ~1/3 of the training data instead of ~1/5, and the meta-controller
  chooses between 3 options instead of 5.
- Conditioning reverted from FiLM back to concatenation. FiLM had been tried in
  the interim and fragmented the latent space; the FiLM runs are the
  `run_film_seed*.log` files in `results/`.
- `objective_embed_dim` 16 → 32, taking the embedding from ~6% to ~11% of the
  288-dim conditioned vector.
- `meta_decision_interval` 8 → 16, roughly 18 decisions per episode instead of
  ~37.

**Result, 3 seeds (42, 7, 123), 20,000 episodes per model:** augmented 0.572 vs
baseline 0.313 mean multi-task success, +25.9 pp, augmented ahead on all 12
task × seed cells. These are the numbers that went on the conference poster and
into the README, and the per-seed reports are in `results_seed42/`,
`results_seed7/` and `results_seed123/`.

**Two caveats that belong with that number, and were not stated at the time:**

1. The reduction from five sub-objectives to three was decided using selection
   rates measured on the same four tasks the result is reported on. There is no
   held-out task. This is tuning on the evaluation set.
2. More generally, this pass searched over four architecture changes against the
   evaluation metric, immediately after that metric had read null. The +25.9 pp
   is a post-selection estimate and should be read as an upper bound.

Both caveats are now stated in `README.md`.

---

## 2026-04-21 — Aggregate figures and poster

`regenerate_plots.py` added, which rebuilds the aggregate figures from the
per-seed `evaluation_report.json` files so that the graphics and the README
table cannot drift apart. Poster prepared for the 2026 California
Neurotechnology Conference.

No experimental changes.

---

## 2026-08-11 — Hardening pass: selection ablation and more seeds

The three-seed +25.9 pp result establishes that the augmented arm beats the
baseline, but it cannot say *why*. The augmented arm differs from the baseline in
two ways at once — it receives intrinsic rewards, and it learns which
sub-objective to pursue — so the result is equally consistent with the intrinsic
rewards acting as ordinary reward shaping while the GRU contributes nothing.

**Changes made:**

- `--meta-mode` added, selecting between `learned` (unchanged default), `random`
  (uniform draw at each 16-step boundary), `fixed-explore` / `fixed-approach` /
  `fixed-exploit`, and `uniform-sum` (all three intrinsic rewards summed, no
  selection). Everything else is held identical across arms: encoder, 0.08
  intrinsic scale, 16-step commitment boundaries, seeds, episode budget and
  evaluation protocol. `random` still feeds a sub-objective embedding to the
  policy, so the conditioned input dimensionality is unchanged.
- **Bug fixed:** evaluation consulted the meta-controller at *every* step, while
  training committed for 16. The two are now consistent, which is a
  precondition for the ablation arms being comparable at all. This changes the
  `learned` numbers slightly relative to the April runs, which is why every arm
  was re-run rather than compared against the committed reports.
- Main comparison and `random` extended from 3 seeds to 10; the four remaining
  arms run on 5 of the same seeds.
- `aggregate_results.py` added: bootstrap 95% CIs and interquartile mean over
  seeds, following Agarwal et al. (2021).

Runs are recorded in `results_main/seed<N>/` and
`results_ablation_<mode>/seed<N>/`. The April three-seed runs are kept unchanged
in `results/`, `results_seed42/`, `results_seed7/` and `results_seed123/` as the
provenance of the poster.

**Outcome:** see the ablation table in `README.md`.
