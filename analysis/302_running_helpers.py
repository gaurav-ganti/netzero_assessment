"""
Parallel MAGICC scenario runner.

This script runs MAGICC climate model simulations in parallel by executing
Jupyter notebooks via papermill. It batches scenarios and distributes them
across multiple processes.

The batch file (300_all_batches_v0.6.csv) has one row per scenario/variant, not per
(scenario, flag): 301_run_magicc.ipynb always computes all three MAGICC_FLAGS
("ANTHROPOGENIC", "CO2", "GHG") for a scenario in a single execution and saves one
output CSV per flag - see 301_run_magicc.ipynb's own documentation.

Usage:
    python 302_running_helpers.py <batch_size> <n_jobs> <n_processes> [--num-shards N] [--shard-index I]

Arguments:
    batch_size   : Number of scenarios per batch
    n_jobs       : Number of notebooks to run in parallel
    n_processes  : Number of MAGICC processes per notebook
    --num-shards : Split the full batch CSV into this many shards, each run by a
                   separate invocation (e.g. one per SLURM array task). Default 1
                   (no sharding). Rows are distributed round-robin across shards.
    --shard-index: Which shard (0-indexed) this invocation should process. Default 0.
                   If SLURM_ARRAY_TASK_ID/_MIN/_MAX/_STEP are set and these flags are
                   not passed explicitly, the shard index/count are derived from them
                   automatically - so a SLURM array job needs no extra flags at all.

Example:
    # Run with batches of 10 scenarios, 4 parallel notebooks, 2 MAGICC processes each
    python 302_running_helpers.py 10 4 2

    # As shard 3 of 20 (e.g. one SLURM array task out of --array=0-19)
    python 302_running_helpers.py 10 4 2 --num-shards 20 --shard-index 3

Note:
    Total CPU usage = n_jobs * n_processes. For an 8-core machine, good options:
    - n_jobs=4, n_processes=2 (8 total)
    - n_jobs=2, n_processes=4 (8 total)
    - n_jobs=8, n_processes=1 (maximize notebook parallelism)

    Sharding only splits work across separate invocations of this script (e.g.
    separate SLURM array tasks/nodes) - it does not itself parallelize within one
    invocation, which is what n_jobs/n_processes are for.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import concurrent.futures
import datetime
import logging
import os
import pathlib
import time
from itertools import islice

import dotenv
import pandas as pd
import papermill as pm
from tqdm import tqdm

from utils import sanitize_label

# Load environment variables
dotenv.load_dotenv()

# Set up output folder
OUTPUT_FOLDER = pathlib.Path(os.environ["OUTPUT_FOLDER"])
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Set up papermill output folder
PAPERMILL_OUTPUT_FOLDER = OUTPUT_FOLDER / "papermill_output"
PAPERMILL_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Set up logging folder
LOGGING_FOLDER = OUTPUT_FOLDER / "logging"
LOGGING_FOLDER.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# Batch configuration file produced by 300_prepare_scenarios.ipynb
BATCH_FILE = OUTPUT_FOLDER / "300_all_batches_v0.6.csv"

# 301_run_magicc.ipynb always produces one output file per scenario for each of these
# flags in a single execution (see its own documentation) - there is no MAGICC_FLAG
# column in the batch file any more.
MAGICC_FLAGS = ("ANTHROPOGENIC", "CO2", "GHG")

# Session timestamp used for both logging and manifest files
SESSION_DATETIME = None
# Distinguishes concurrent invocations (e.g. separate SLURM array tasks) sharing the
# same OUTPUT_FOLDER, so their log/manifest/failed-run files don't collide.
RUN_LABEL = None
MANIFEST_ENTRIES = []


def get_shard_from_slurm_env():
    """Derive (shard_index, num_shards) from SLURM_ARRAY_TASK_* env vars, if set.

    Returns (None, None) if not running as a SLURM array task, so callers can fall
    back to explicit CLI flags or the single-shard default.
    """
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_min = os.environ.get("SLURM_ARRAY_TASK_MIN")
    task_max = os.environ.get("SLURM_ARRAY_TASK_MAX")
    task_step = os.environ.get("SLURM_ARRAY_TASK_STEP", "1")
    if task_id is None or task_min is None or task_max is None:
        return None, None
    shard_index = (int(task_id) - int(task_min)) // int(task_step)
    num_shards = (int(task_max) - int(task_min)) // int(task_step) + 1
    return shard_index, num_shards


def setup_logging(run_label):
    """Set up logging - should only be called from main process."""
    global SESSION_DATETIME
    logging.basicConfig(level=logging.INFO)

    SESSION_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    fh = logging.FileHandler(LOGGING_FOLDER / f'{SESSION_DATETIME}_{run_label}_302_runs.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


def save_manifest(run_label):
    """Save the session manifest file with all completed runs."""
    if not MANIFEST_ENTRIES:
        return

    manifest_dir = OUTPUT_FOLDER / "individual_runs/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f'{SESSION_DATETIME}_{run_label}_manifest.csv'
    manifest_df = pd.DataFrame(MANIFEST_ENTRIES)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}")


def check_failed_runs(input_df, run_label):
    """
    Cross-reference completed runs against input to identify failures.

    A row is only counted as fully completed if all of MAGICC_FLAGS were produced for
    it - 301_run_magicc.ipynb always tries to produce all three per scenario in one
    execution, but a partial failure inside that notebook (e.g. one QEXTRA pass
    erroring out after another succeeded) would otherwise go unnoticed.

    Args:
        input_df: DataFrame with the original input scenarios (MODEL, SCENARIO,
            ENSEMBLE_MEMBERS, FILE columns, as produced by
            300_prepare_scenarios.ipynb) - already restricted to this shard.
        run_label: Distinguishes this invocation's output files from other
            concurrent shards sharing the same OUTPUT_FOLDER.

    Returns:
        DataFrame of failed runs (empty if all succeeded)
    """
    if not MANIFEST_ENTRIES:
        logger.warning("No manifest entries found - all runs may have failed")
        failed_df = input_df.copy()
        failed_df["missing_flags"] = ",".join(sorted(MAGICC_FLAGS))
    else:
        # Count completed flags per (model, scenario) - a row is only "done" if it has
        # an entry for every flag in MAGICC_FLAGS.
        completed_flags = {}
        for entry in MANIFEST_ENTRIES:
            key = (entry["model"], entry["scenario"])
            completed_flags.setdefault(key, set()).add(entry["magicc_flag"])

        # Find rows missing one or more expected flags
        failed_runs = []
        for row in input_df.itertuples():
            key = (row.MODEL, row.SCENARIO)
            if completed_flags.get(key, set()) != set(MAGICC_FLAGS):
                failed_runs.append({
                    "MODEL": row.MODEL,
                    "SCENARIO": row.SCENARIO,
                    "ENSEMBLE_MEMBERS": int(row.ENSEMBLE_MEMBERS),
                    "FILE": row.FILE,
                    "missing_flags": ",".join(sorted(set(MAGICC_FLAGS) - completed_flags.get(key, set()))),
                })

        failed_df = pd.DataFrame(failed_runs)

    n_failed = len(failed_df)
    n_total = len(input_df)
    n_completed = n_total - n_failed

    logger.info(f"Run completion summary: {n_completed}/{n_total} succeeded, {n_failed} failed")

    if n_failed > 0:
        logger.warning(f"Failed runs ({n_failed}):")
        for _, row in failed_df.iterrows():
            logger.warning(f"  - {row['MODEL']} | {row['SCENARIO']} (missing: {row['missing_flags']})")

        # Save failed runs to CSV for re-running
        failed_output_path = OUTPUT_FOLDER / f"302_failed_runs_{SESSION_DATETIME}_{run_label}.csv"
        failed_df.to_csv(failed_output_path, index=False)
        logger.info(f"Failed runs saved to: {failed_output_path}")
    else:
        logger.info("All runs completed successfully!")

    return failed_df

current_path=pathlib.Path().resolve()

def construct_and_batch_configs(input_data, batch_size, n_processes):
    """
    Build configuration dicts from input data and yield them in batches.

    Args:
        input_data: DataFrame with columns MODEL, SCENARIO, ENSEMBLE_MEMBERS, FILE
            (as produced by 300_prepare_scenarios.ipynb) - one row per scenario, not
            per (scenario, flag)
        batch_size: Number of configurations per batch
        n_processes: Number of MAGICC processes to use per notebook

    Yields:
        Lists of configuration dicts, each list containing up to batch_size items
    """
    logger.info(
        f"Step 1: Batching {len(input_data)} into batches of size {batch_size}"
    )
    configs = [
        {
            "MODEL": row.MODEL,
            "SCENARIO": row.SCENARIO,
            "ENSEMBLE_MEMBERS": int(row.ENSEMBLE_MEMBERS),
            "FILE": row.FILE,
            "N_PROCESSES": n_processes
        }
        for row in input_data.itertuples()
    ]
    iterator = iter(configs)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def run_papermill_notebook(config):
    """
    Execute the MAGICC notebook with the given configuration.

    301_run_magicc.ipynb always produces one output file per scenario for each flag
    in MAGICC_FLAGS in a single execution (see its own documentation).

    Args:
        config: Dict containing MODEL, SCENARIO, ENSEMBLE_MEMBERS, FILE, and
                N_PROCESSES parameters for the notebook

    Returns:
        List of dicts with run metadata for the manifest, one per flag
    """
    import warnings
    warnings.filterwarnings("ignore")

    OUTPUT_MODEL = sanitize_label(config['MODEL'])
    OUTPUT_SCENARIO = sanitize_label(config['SCENARIO'])
    pm.execute_notebook(
        pathlib.Path(current_path / "301_run_magicc.ipynb"),
        PAPERMILL_OUTPUT_FOLDER / f"302_{OUTPUT_MODEL}_{OUTPUT_SCENARIO}_{config['ENSEMBLE_MEMBERS']}.ipynb",
        parameters=config
    )

    # Return one manifest entry per flag - the notebook saves one CSV per flag
    return [
        {
            "model": config["MODEL"],
            "scenario": config["SCENARIO"],
            "ensemble_members": config["ENSEMBLE_MEMBERS"],
            "magicc_flag": flag,
            "n_processes": config["N_PROCESSES"],
            "output_filename": f"{OUTPUT_MODEL}_{OUTPUT_SCENARIO}_{flag}.csv",
        }
        for flag in MAGICC_FLAGS
    ]

def parallel_process(conf_batch, n_jobs=16, front_num=3):
    """
    Execute notebook configurations in parallel with progress tracking.

    Runs a few jobs sequentially first to catch errors early, then parallelizes
    the remaining jobs using ProcessPoolExecutor.

    Args:
        conf_batch: List of configuration dicts to process
        n_jobs: Maximum number of parallel workers (default: 16)
        front_num: Number of jobs to run sequentially first for error checking (default: 3)

    Returns:
        List of results (None for success, Exception for failures)

    Reference:
        https://danshiebler.com/2016-09-14-parallel-progress-bar/
    """
    results = []

    # Run a few jobs sequentially first to catch errors early
    # Use more MAGICC processes for sequential jobs since they don't compete for resources
    if front_num > 0:
        for conf in conf_batch[:front_num]:
            try:
                conf["N_PROCESSES"] = 10
                result = run_papermill_notebook(conf)  # list of per-flag manifest entries
                results.append(result)
                MANIFEST_ENTRIES.extend(result)
            except Exception as e:
                results.append(e)

    # Run remaining jobs in parallel
    remaining = conf_batch[front_num:]
    if not remaining:
        return results

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_papermill_notebook, conf) for conf in remaining]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()  # list of per-flag manifest entries
                results.append(result)
                MANIFEST_ENTRIES.extend(result)
            except Exception as e:
                results.append(e)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MAGICC for the scenario/flag batch produced by 300_prepare_scenarios.ipynb."
    )
    parser.add_argument("batch_size", type=int, help="Number of scenarios per batch")
    parser.add_argument("n_jobs", type=int, help="Number of notebooks to run in parallel")
    parser.add_argument("n_processes", type=int, help="Number of MAGICC processes per notebook")
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Split the full batch CSV into this many shards (default: 1, or "
             "auto-detected from SLURM_ARRAY_TASK_* if set)",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="Which shard (0-indexed) this invocation processes (default: 0, or "
             "auto-detected from SLURM_ARRAY_TASK_* if set)",
    )
    args = parser.parse_args()

    slurm_shard_index, slurm_num_shards = get_shard_from_slurm_env()
    num_shards = args.num_shards if args.num_shards is not None else (slurm_num_shards or 1)
    shard_index = args.shard_index if args.shard_index is not None else (slurm_shard_index or 0)
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index ({shard_index}) must be in [0, {num_shards})")

    run_label = f"shard{shard_index}of{num_shards}" if num_shards > 1 else "full"
    setup_logging(run_label)

    batch_size, n_jobs, n_processes = args.batch_size, args.n_jobs, args.n_processes

    start_time = datetime.datetime.now().isoformat()
    logger.info(f'Starting runs at: {start_time}')
    logger.info(f'Configuration: batch_size={batch_size}, n_jobs={n_jobs}, n_processes={n_processes}')
    logger.info(f'Shard: {shard_index} of {num_shards} (run_label={run_label})')
    mod_scens = pd.read_csv(BATCH_FILE, index_col=0)
    if num_shards > 1:
        # Round-robin (not contiguous) slicing: every row costs the same (all three
        # MAGICC_FLAGS are always computed together), but the batch file is still
        # concatenated as unequal-size variant blocks (original/NZCO2/NZGHG) -
        # round-robin gives near-equal shard sizes regardless of how those blocks
        # divide by num_shards, which contiguous chunking wouldn't guarantee.
        mod_scens = mod_scens.iloc[shard_index::num_shards]
        logger.info(f'This shard covers {len(mod_scens)} of the full batch file\'s rows')
    confs = construct_and_batch_configs(mod_scens, batch_size, n_processes)
    logger.info("Config batching successful")
    counter = 1
    logger.info(
        f"Step 2: Running batches of scenarios through MAGICC"
    )
    for c in confs:
        logger.info(f"Running batch no: {counter}")
        parallel_process(
            c,
            n_jobs=n_jobs,
            front_num=1
        )
        batch_finish = datetime.datetime.now().isoformat()
        logger.info(f"Completed batch no: {counter} at {batch_finish}")
        counter += 1
    runs_finish = datetime.datetime.now().isoformat()
    logger.info(f"Completed runs at: {runs_finish}")
    save_manifest(run_label)
    check_failed_runs(mod_scens, run_label)
