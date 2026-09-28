#!/bin/bash
#SBATCH --job-name=magicc_run
#SBATCH --partition=generic
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# Sized from observed shard runtimes on the real ~1955-scenario/20-shard production
# run (job 3072453): every shard completed in 2h26m-3h13m for ~98 scenarios/shard, so
# 6h gives ~3h of headroom over the slowest observed shard. This used to be
# 2-00:00:00 ("to be safe"), which backfired: SLURM's backfill scheduler favors jobs
# with realistic (shorter) requested walltimes when slotting into gaps, so an
# oversized request can queue far longer than the job itself takes to run - a scoped
# 784-scenario/8-shard rerun (same ~98 scenarios/shard) sat pending for an estimated
# 14h under the old 2-day request. If a future batch changes the scenarios/shard
# ratio substantially (much larger BATCH_SIZE, or far fewer/more shards than ~98
# scenarios each), override with `sbatch --time=...` rather than assuming this default
# still fits.
#SBATCH --exclude=compute5
# compute5 has a broken /apps mount (module loads fail with "unknown module" there
# even for modules that exist fine on every other generic-partition node) - confirmed
# 2026-09-09. Remove this exclusion once that's fixed cluster-side.
#SBATCH --output=logging/slurm_%A_%a.out
#SBATCH --error=logging/slurm_%A_%a.err
#
# Runs 302_running_helpers.py as a SLURM array job. Each array task processes one
# shard of the batch file produced by 300_prepare_scenarios.ipynb; sharding (which
# rows this task covers) is auto-detected from SLURM_ARRAY_TASK_ID/_MIN/_MAX/_STEP
# unless overridden with an explicit --num-shards/--shard-index below.
#
# GCCcore/13.2.0 (the cluster default) has a measurably higher MAGICC ensemble-member
# failure rate than GCCcore/14.3.0 for reasons still under investigation (a compiler/
# runtime-build sensitivity - see notes/ and the upstream GitLab issue); 14.3.0 is the
# best confirmed option so far, hence pinned explicitly below rather than relying on
# whatever the login/interactive-session default happens to be.
#
# Usage:
#
#   Small validation test (recommended before a real production run) - 4 array tasks,
#   each covering only ~2-3 scenarios out of a forced 700-way split (the same
#   tiny-shard trick used throughout manual smoke-testing), now via a genuine
#   sbatch/array submission rather than an interactive srun session - this is what
#   actually exercises module loading and n_jobs>1 concurrency in a real batch
#   context, neither of which had been tested before:
#
#     sbatch --array=0-3 run_magicc.sbatch 1 4 4 --num-shards 700
#
#   Real production run - auto-shards the whole batch across the array size, e.g.
#   --array=0-19 gives 20 shards of ~98 scenarios each for a ~1955-row batch:
#
#     sbatch --array=0-19 run_magicc.sbatch 5 4 4
#
# Positional args: BATCH_SIZE N_JOBS N_PROCESSES (passed straight to
# 302_running_helpers.py - see its own module docstring for what these mean and how
# to size them). Any additional arguments (e.g. --num-shards/--shard-index) are
# passed straight through too.

set -euo pipefail

module load GCCcore/14.3.0

# Deliberately no `cd "$(dirname "$0")"` here: sbatch copies the submitted script
# into a per-job spool directory before running it, so $0 points there, not at this
# file's real location. SLURM already preserves the submission directory as the
# working directory by default (unless overridden with --chdir), which is what we
# want, so there's nothing to do here.

BATCH_SIZE="$1"
N_JOBS="$2"
N_PROCESSES="$3"
shift 3

uv run python 302_running_helpers.py "$BATCH_SIZE" "$N_JOBS" "$N_PROCESSES" "$@"
