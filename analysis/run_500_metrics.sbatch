#!/bin/bash
#SBATCH --job-name=500_metrics
#SBATCH --partition=generic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=compute5
# compute5 has a broken /apps mount (module loads fail with "unknown module" there
# even for modules that exist fine on every other generic-partition node) - confirmed
# 2026-09-09. Remove this exclusion once that's fixed cluster-side.
#SBATCH --output=logging/slurm_500_%j.out
#SBATCH --error=logging/slurm_500_%j.err
#
# Runs 500_calculate_peak_2100_warming.py as a proper SLURM job, independent of the
# JupyterHub single-user server's lifecycle (unlike running it directly in a Jupyter
# terminal, which is subject to whatever idle-culling policy the hub has configured -
# a real risk for a job that can run 1-2 hours).
#
# Two passes, run back-to-back in one process (see the script's own __main__ block):
#   1. compile_gsat_metrics_from_manifest - parallelized across --cpus-per-task via
#      ProcessPoolExecutor, so N_WORKERS below should match --cpus-per-task.
#   2. compile_component_metrics_from_manifest - single-threaded regardless of
#      N_WORKERS (no parallelism in that pass currently).
#
# Usage:
#
#   sbatch run_500_metrics.sbatch regenerated_manifest.csv 8 101_netzero_timings.csv
#
# Positional args: MANIFEST_FILENAME N_WORKERS NETZERO_TIMINGS_FILENAME (last one
# optional - omit it to skip pass 2 and only produce 500_gsat_metrics_*.csv). Defaults
# below match the values used in this project's most recent run if none are given.

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

uv run python 500_calculate_peak_2100_warming.py "$MANIFEST_FILENAME" "$N_WORKERS" "$NETZERO_TIMINGS_FILENAME"
