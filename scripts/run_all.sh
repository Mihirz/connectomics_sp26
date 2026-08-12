#!/usr/bin/env bash
# Reproduces every number in the README's results tables.
#
# Each job is one process pinned to a single BLAS thread: these models are tiny,
# and on a laptop CPU intra-op threading costs more in contention than it buys
# (a single rollout takes 2.2s at 1 thread vs 4.4s at 6).  Throughput comes from
# running jobs in parallel instead.  Set WORKERS to the number of performance
# cores you can spare.
#
#   ./scripts/run_all.sh                       # everything, ~100 CPU-hours
#   WORKERS=4 ./scripts/run_all.sh             # leave cores for other work
#   DEVICE=cuda WORKERS=4 ./scripts/run_all.sh # on the GPU box
#
# Budget the memory as well as the cores: each worker holds ~150 MB resident,
# and on a machine that is already swapping, more workers make the whole queue
# slower rather than faster.
#
# Jobs are ordered by priority: the main learned-vs-baseline comparison first,
# then the decisive `random` ablation, then the remaining arms.  Completed runs
# are skipped, so the script is safe to re-run after an interruption.

set -u
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-7}"
DEVICE="${DEVICE:-cpu}"               # DEVICE=cuda on the GPU box
EPISODES="${EPISODES:-5000}"          # 5000 per task x 4 tasks = 20,000 per model
MAIN_SEEDS="${MAIN_SEEDS:-42 7 123 0 1 2 3 4 5 6}"
SECONDARY_SEEDS="${SECONDARY_SEEDS:-42 7 123 0 1}"

JOBS="$(mktemp)"
trap 'rm -f "$JOBS"' EXIT

emit() {  # emit <results-dir> <args...>
  local dir="$1"; shift
  if [ -f "$dir/evaluation_report.json" ]; then
    echo "skip (done): $dir" >&2
    return
  fi
  mkdir -p "$dir"
  # -u so a run in progress can be watched; these jobs take hours each.
  printf '%s\n' "python3 -u run_experiment.py --episodes $EPISODES --device $DEVICE --results-dir $dir $* > $dir/run.log 2>&1"
}

# 1. Main comparison: learned meta-controller vs. baseline, 10 seeds.
for s in $MAIN_SEEDS; do
  emit "results_main/seed$s" --mode full --seed "$s"
done >> "$JOBS"

# 2. The decisive ablation: same intrinsic rewards, selection made at random.
for s in $MAIN_SEEDS; do
  emit "results_ablation_random/seed$s" --mode ablation --meta-mode random --seed "$s"
done >> "$JOBS"

# 3. Remaining arms, on a subset of the same seed set.
for mode in fixed-explore fixed-approach fixed-exploit uniform-sum; do
  for s in $SECONDARY_SEEDS; do
    emit "results_ablation_$mode/seed$s" --mode ablation --meta-mode "$mode" --seed "$s"
  done
done >> "$JOBS"

echo "$(wc -l < "$JOBS") jobs, $WORKERS workers"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
xargs -P "$WORKERS" -I CMD sh -c CMD < "$JOBS"
echo "all jobs finished"
