#!/bin/bash
#SBATCH --job-name=500c_ody
#SBATCH --partition=generic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=compute5
# compute5 has a broken /apps mount (module loads fail with "unknown module" there
# even for modules that exist fine on every other generic-partition node) - confirmed
# 2026-09-09. Remove this exclusion once that's fixed cluster-side.
#SBATCH --output=logging/slurm_500c_%j.out
#SBATCH --error=logging/slurm_500c_%j.err
#
# Runs 500c_calculate_ody.py as a proper SLURM job, independent of the JupyterHub
# single-user server's lifecycle - same rationale as run_500_metrics.sbatch.
#
# Reads the full annual GSAT timeseries from individual_runs/*_ANTHROPOGENIC.csv (one
# file per scenario, not per-flag like 500's pass 1) and computes overshoot
# degree-years per (model, scenario, run_id) - parallelized across --cpus-per-task via
# ProcessPoolExecutor, so N_WORKERS below should match --cpus-per-task.
#
# Usage:
#
#   sbatch run_500c_ody.sbatch regenerated_manifest.csv 8 101_netzero_timings.csv
#
# Positional args: MANIFEST_FILENAME N_WORKERS NETZERO_TIMINGS_FILENAME.

set -euo pipefail

module load GCCcore/14.3.0

# Deliberately no `cd "$(dirname "$0")"` here - sbatch copies the submitted script
# into a per-job spool directory before running it, so $0 points there, not at this
# file's real location. SLURM already preserves the submission directory as the
# working directory by default (unless overridden with --chdir), which is what we
# want, so there's nothing to do here.

MANIFEST_FILENAME="${1:-regenerated_manifest.csv}"
N_WORKERS="${2:-8}"
NETZERO_TIMINGS_FILENAME="${3:-101_netzero_timings.csv}"

mkdir -p logging

uv run python 500c_calculate_ody.py "$MANIFEST_FILENAME" "$N_WORKERS" "$NETZERO_TIMINGS_FILENAME"
