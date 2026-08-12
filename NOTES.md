# Design Decisions and the Evidence Behind Them

Every non-obvious choice in this codebase, why it is set the way it is, and what
happens if it is changed. This is the standing reference; the dated record of how
the project arrived here — including the results that turned out to be artifacts
— is in [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md).

Values quoted are the current values in `config.py`. Changing any of them
invalidates the committed results in `results_main/` and
`results_ablation_*/`, which is why the entries below say what each one is
holding in place.

---

## Reward structure — "Design B"

```
policy reward = dense_task_reward + 0.08 * intrinsic(selected_sub_objective)
meta reward   = sparse_failure_penalty + 0.2 * intrinsic(selected_sub_objective)
```

**Both models receive the same dense task reward.** The augmented model gets the
intrinsic term on top; the baseline does not. The meta-controller never sees the
dense reward — it has to infer which sub-objectives are productive from failure
avoidance plus per-step intrinsic feedback.

The earlier alternative ("Design A") gave the action policy intrinsic reward
*only*. That is not a fair comparison: it hands the augmented model a strictly
harder problem than the baseline, and it does not match the biology either. The
PFC modulates striatal reward learning rather than substituting for it.

**`intrinsic_reward_scale = 0.08`** is load-bearing. Raw intrinsic rewards run
~0.1/step against dense rewards at ~0.01/step. Unscaled, the task gradient is
drowned out and the policy learns to farm intrinsic reward instead of solving the
task.

**`meta_intrinsic_feedback = 0.2`** exists because credit assignment from a
single end-of-episode failure signal over ~300 timesteps is close to hopeless.
It is the per-step "is this drive producing anything" signal.

## Temporal commitment — `meta_decision_interval = 16`

The meta-controller chooses once every 16 steps and the choice is held in
between, giving ~18 decisions per episode instead of ~300. Three reasons, in
descending order of how much they actually mattered:

1. **Credit assignment.** 300 sparse-feedback decisions per episode is not a
   learnable problem at this scale.
2. **Policy stability.** The action policy is conditioned on the sub-objective
   embedding. Switching every step creates a feedback loop: the meta-controller
   shifts, the policy destabilizes, and the meta-controller then evaluates a
   strategy against a policy that has not settled into it.
3. **Biological fidelity.** PFC executive control operates on seconds, not
   milliseconds.

Evaluation applies the same 16-step commitment as training. It did not until
2026-08-11 — evaluation used to re-decide every step — and that inconsistency
had to be fixed before the selection-mechanism ablation could mean anything.

## Sub-objective library — three, not five

EXPLORE, APPROACH, EXPLOIT. AVOID and MEMORIZE were removed after they were
selected under 5% and 0–18% of the time respectively.

**This reduction was decided on selection rates measured on the four evaluation
tasks, with no held-out task.** It is a form of tuning on the test set and the
README says so. All results currently reported were produced after the
reduction.

APPROACH and EXPLOIT are deliberately different signals: APPROACH rewards
*decreasing* distance to a salient target, EXPLOIT rewards *being* near one. The
distinction is what lets "travel to the food" and "stay and harvest" be separate
strategies.

## Conditioning — concatenation at `objective_embed_dim = 32`

The policy sees `[latent_256; embed_32]` = 288 dims. FiLM conditioning was tried
and reverted: it fragmented the latent space. 32 dims puts the embedding at ~11%
of the conditioned vector, enough for the policy to differentiate modes without
the latent state losing its share of the input.

## GRU meta-controller

`GRUCell`, not `nn.GRU`, because rollouts step one observation at a time. The
non-obvious part is PPO re-evaluation: mini-batch shuffling destroys temporal
order, so `RolloutBuffer` stores the hidden state *as it was before* each step's
forward pass and `evaluate_actions()` takes those stored states as an argument.
Re-running the GRU from a zero state during the update would compute log-probs
for decisions the model never made.

Hidden state resets on episode boundaries, in training and evaluation both.

The GRU's demonstrated effect is on the mechanism, not on the score: it keeps
selection entropy off the floor where the feedforward version collapsed
(0.30–1.08 across task × seed, though the low end is near-collapse). Whether it
improves task performance is what the `--meta-mode random` ablation was built
to test, and **the answer is no**: over 10 seeds, `random` is 7.2 pp below
`learned` with a 95% CI of [-17.8, +3.6], and `uniform-sum` — no selection at
all — beats `learned` by 15.9 pp [3.6, 29.8].

Keep this in mind before spending effort on the meta-controller. The
sub-objectives themselves matter (`fixed-exploit` is 24.3 pp worse than
`learned`, well outside its interval), but nothing so far shows that *learning
which one to use* beats picking a reasonable one and sticking with it.

## Parameter parity

231,050 (augmented) vs 231,590 (baseline), a 540-parameter gap, 0.998x. The
baseline's policy and value heads are widened (`hidden_dim + 168`) to absorb the
capacity the augmented model spends on the meta-controller, the embeddings and
the meta-value head. Without this the comparison measures capacity, not paradigm.

Note that in every `--meta-mode` other than `learned` the GRU meta-controller and
meta-value head are instantiated but never run, so those arms train and act with
152,518 parameters. `count_active_parameters()` in `models.py` reports this, and
the README quotes it rather than the constructor's total.

## Numerical stability

High success produces very short episodes, which produce reward spikes, which
produced NaN weights and a crashed run in April. Four defenses, all still in
`training.py`:

- advantage clamping to [-5, 5]
- PPO ratio clamping to [0.01, 100]
- a gradient-norm check between `backward()` and `optimizer.step()` — if
  `clip_grad_norm_` returns NaN/Inf, zero the gradients and skip the step
- `try/except (ValueError, RuntimeError)` around each PPO mini-batch, since NaN
  weights surface as a `Categorical` constructor error

The crash has not recurred since the gradient-norm check was added.

## Entropy floor on the meta-controller

`meta_entropy_coef = 0.15`, `meta_entropy_floor = 0.4`. Below the floor the
entropy bonus ramps smoothly from 1x to 5x rather than switching on at a
threshold; a hard cliff destabilized training. Without any floor the
meta-controller collapses onto a single sub-objective within a few thousand
episodes, which makes the whole architecture pointless.

## Environment sizing

**`grid_size = 20`, not 32.** A 32×32 pool has ~616 reachable cells and 300-step
episodes cannot cover enough of it to find a hidden platform by chance, so there
is no signal to bootstrap from. 20×20 gives ~254.

**Morris Water Maze** carries four coloured landmark cues at the pool edges
(allocentric reference, as in the real protocol), a proximity gradient rendered
as a warm tint so the CNN has a learnable cue, and a distance-scaled timeout
penalty so a near miss is punished less than a wander.

## Known limitations

- **Visual Search is not solved by either model** (0.14 vs 0.08). The delta is
  noise between two failures and should not be counted as support for the
  hypothesis.
- **Evaluation environments are unseeded.** `make_env(..., variant_seed=None)`
  produces a fresh `RandomState`, so repeated evaluation of identical weights
  gives slightly different numbers. Across-seed CIs absorb this, but a single
  run is not bit-reproducible.
- **Few-shot adaptation does not fine-tune.** It measures how quickly rolling
  success reaches threshold with fixed weights. It is reported as such.
- **Four tasks, one family.** All four are 20×20 grid-worlds with the same
  action space and observation format. Nothing here speaks to transfer beyond
  that family.
