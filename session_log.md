# Development Session: GRU-Based Recurrent Meta-Controller & Experimental Fairness Audit

## Project Overview

This project implements a biologically-inspired reinforcement learning architecture that models the prefrontal cortex (PFC) as a meta-controller selecting among sub-objectives (EXPLORE, APPROACH, AVOID, EXPLOIT, MEMORIZE). The augmented model is compared against a standard PPO baseline across four grid-world navigation tasks: Morris Water Maze, Visual Foraging, Dynamic Obstacles, and Visual Search.

---

## Session Narrative

### Phase 1: Architectural Improvement — GRU-Based Recurrent Meta-Controller

After multiple rounds of hyperparameter tuning showed diminishing returns (~3-4% average delta over baseline, seed-dependent), I identified a structural limitation: the feedforward meta-controller makes stateless decisions with no memory of what worked earlier in an episode. A recurrent architecture should let the meta-controller track strategy effectiveness over time, mirroring how the biological PFC maintains working memory.

**Prompt:**
> The feedforward meta-controller is making each strategy decision in isolation. Implement a GRU-based recurrent meta-controller that carries hidden state across timesteps, so it can integrate temporal context when selecting sub-objectives. This requires changes across the full pipeline — models, training, and evaluation.

**Implementation (across 4 files):**

`models.py` — Replaced feedforward MetaController with GRU variant:

```python
class MetaController(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_objectives: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
        )

    def forward(self, latent, hidden):
        h = self.gru(latent, hidden)
        logits = self.head(h)
        return Categorical(logits=logits), h
```

Key design decisions:
- `GRUCell` (not full `GRU`) since we step one observation at a time during rollouts
- `evaluate_actions()` accepts stored per-step hidden states so PPO can correctly re-evaluate after shuffling mini-batches
- Hidden state resets on episode boundaries during both training and evaluation
- Baseline `wider_hidden` adjusted from +80 to +160 to maintain parameter parity (0.998x ratio, 362 param difference)

`training.py` — RolloutBuffer stores per-step `meta_hiddens`; `collect_rollout()` carries and resets GRU state:

```python
# Store hidden state BEFORE forward pass (what PPO needs for re-evaluation)
step_meta_hidden = meta_hidden.clone()

action, action_lp, new_obj, obj_lp, value, meta_value, new_hidden = \
    self.model(obs, meta_hidden)
meta_hidden = new_hidden

# Reset on episode boundary
if dones[i]:
    meta_hidden[i] = 0.0
```

`evaluate.py` — Hidden state initialized per episode and carried through evaluation.

**Initial Results (before fairness audit):**

| Task | Augmented (GRU) | Baseline | Delta |
|---|---|---|---|
| Morris Water Maze | 1.00 | 0.87 | +0.13 |
| Visual Foraging | 0.96 | 0.64 | +0.32 |
| Dynamic Obstacles | 0.87 | 0.89 | -0.02 |
| Visual Search | 0.35 | 0.06 | +0.29 |
| **Average** | **0.795** | **0.615** | **+0.180** |

Strategy entropy remained stable at ~1.43-1.47 across all tasks, confirming the GRU prevents the entropy collapse that plagued longer training runs with the feedforward variant.

---

### Phase 2: Identifying Episode Budget Asymmetry

After reviewing the training configuration, I noticed the augmented model was receiving 2x more training episodes than the baseline via a budget multiplier:

```python
total_interleaved_eps = int(cfg.train.total_episodes * len(cfg.tasks) * 2.0)
```

This gave the augmented model ~10,000 episodes per task vs 5,000 for the baseline.

**Prompt:**
> The augmented model is getting twice the training episodes as the baseline — the 2x multiplier on interleaved training means ~10k episodes per task vs 5k for the baseline. That's not a fair comparison. Equalize the budgets.

**Fix:** Removed the 2x multiplier. Results with equal budgets: +6.2% delta (down from +18%).

---

### Phase 3: Comprehensive Fairness Audit

The episode budget fix prompted a deeper question: what other asymmetries might be inflating the results?

**Prompt:**
> Audit the entire experimental pipeline for fairness — check training structure, evaluation conditions, environment seeds, early stopping, reward signals, everything. I want to make sure we're not p-hacking.

**Systematic audit found 6 issues:**

1. **Training structure confound (MAJOR):** Augmented model trained multi-task (one shared model, interleaved across 4 tasks). Baseline trained single-task (one dedicated model per task). Transfer and multi-task metrics inherently favor the multi-task trained model regardless of the meta-controller.

2. **Evaluation determinism asymmetry (MODERATE):** Augmented model evaluated with `deterministic=False` (stochastic actions + stochastic meta-controller). Baseline evaluated with `deterministic=True` (greedy actions). Different evaluation protocols invalidate the comparison.

3. **Environment seed offset (MODERATE):** Baseline used `seed + 1000`, augmented used `seed`. Different task layouts during training introduce a confound.

4. **Best-model restoration asymmetry (MODERATE):** Augmented model always restored best checkpoint. Baseline only restored on early stopping trigger — if it trained to completion, it kept potentially degraded final weights.

5. **Few-shot evaluation was broken (MINOR):** Optimizer created but never used, `torch.no_grad()` prevented any learning. The metric was silently measuring zero-shot performance, not few-shot adaptation.

6. **Dead config parameter (MINOR):** `success_signal = 0.5` defined but never referenced anywhere in the codebase.

**All fixes applied:**

```python
# 1. Baseline now trains multi-task interleaved (same as augmented)
baseline_model = BaselineModel(cfg.model, cfg.env).to(cfg.device)
base_optimizer = optim.Adam(baseline_model.parameters(), lr=cfg.train.lr_policy)
# ... interleaved round-robin training loop matching augmented structure ...

# 2. Both models use deterministic=True during evaluation
action, _, obj_idx, _, _, _, meta_hidden = model(obs_tensor, meta_hidden, deterministic=True)

# 3. Same seeds
self.env = make_vectorized_env(task_name, cfg.env, cfg.train.num_parallel_envs, base_seed=cfg.seed)

# 4. Both models always restore best checkpoint
if not early_stopped and best_model_state is not None and evals_since_best > 0:
    trainer.model.load_state_dict(best_model_state)
```

---

### Phase 4: Fair Results

With all asymmetries removed:

| Task | Augmented | Baseline | Delta |
|---|---|---|---|
| Morris Water Maze | 0.19 | 0.32 | -0.13 |
| Visual Foraging | 0.59 | 0.75 | -0.16 |
| Dynamic Obstacles | 0.84 | 0.64 | +0.20 |
| Visual Search | 0.19 | 0.11 | +0.08 |
| **Average** | **0.453** | **0.455** | **-0.003** |

The delta collapsed to effectively zero. Strategy entropy also collapsed (0.6-0.8 vs ~1.4 before), with the meta-controller only using EXPLORE and APPROACH. Under fair conditions, the self-optimizing paradigm provides no measurable advantage — the previous +18% result was an artifact of compounding experimental asymmetries.

---

## Key Takeaways

- Architectural improvements (GRU meta-controller) showed genuine gains on the mechanism level — stable entropy, temporal integration — but these don't translate to task performance advantages when the experimental comparison is properly controlled.
- The original experiment had at least 6 independent sources of bias favoring the augmented model, each individually defensible ("two-level optimization needs more budget," "stochastic meta-controller preserves diversity") but collectively amounting to a rigged comparison.
- The honest result is a null finding: the meta-controller paradigm does not outperform standard PPO under matched conditions on these tasks at this scale.

---

## Files Modified

| File | Changes |
|---|---|
| `models.py` | GRU-based MetaController, updated forward/evaluate signatures, parameter parity |
| `training.py` | Per-step hidden state storage in RolloutBuffer, GRU state management in rollouts, seed fix |
| `evaluate.py` | GRU hidden state in evaluation loops, determinism fix, few-shot eval cleanup |
| `run_experiment.py` | Equal episode budgets, multi-task baseline training, best-model restoration for both |
| `config.py` | Removed dead `success_signal` parameter |
