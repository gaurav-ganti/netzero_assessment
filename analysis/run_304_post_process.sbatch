#!/bin/bash
#SBATCH --job-name=304_post_process
#SBATCH --partition=generic
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=compute5
# compute5 has a broken /apps mount (module loads fail with "unknown module" there
# even for modules that exist fine on every other generic-partition node) - confirmed
# 2026-09-09. Remove this exclusion once that's fixed cluster-side.
#SBATCH --output=logging/slurm_304_%j.out
#SBATCH --error=logging/slurm_304_%j.err
#
# Runs 304_post_process_gsat.ipynb as a proper SLURM job, independent of the
# JupyterHub single-user server's lifecycle - same rationale as run_500_metrics.sbatch
# / run_500c_ody.sbatch. Unlike those, this is a single-threaded sequential loop over
# every (model, scenario, flag) in the batch file (~1,955 scenarios x 3 flags) - not
# parallelized via ProcessPoolExecutor, so --cpus-per-task is modest here. If this
# turns out too slow in practice, parallelizing the loop would be the natural next
# step, but this rerun is deliberately unchanged from the already-validated original
# logic (plus one small additive change: also saving
# individual_runs_quantiles/{model}_{scenario}_{flag}.csv, gcages' own per-year
# quantile trajectory, previously computed but discarded).
#
# Usage:
#
#   sbatch run_304_post_process.sbatch

set -euo pipefail

module load GCCcore/14.3.0

# Deliberately no `cd "$(dirname "$0")"` here - sbatch copies the submitted script
# into a per-job spool directory before running it, so $0 points there, not at this
# file's real location. SLURM already preserves the submission directory as the
# working directory by default (unless overridden with --chdir), which is what we
# want, so there's nothing to do here.

mkdir -p logging

uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ar7_netzero \
  --ExecutePreprocessor.timeout=21600 \
  304_post_process_gsat.ipynb
