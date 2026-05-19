"""
Parallel MAGICC scenario runner.

This script runs MAGICC climate model simulations in parallel by executing
Jupyter notebooks via papermill. It batches scenarios and distributes them
across multiple processes.

Usage:
    python 301_running_helpers.py <batch_size> <n_jobs> <n_processes>

Arguments:
    batch_size   : Number of scenarios per batch
    n_jobs       : Number of notebooks to run in parallel
    n_processes  : Number of MAGICC processes per notebook

Example:
    # Run with batches of 10 scenarios, 4 parallel notebooks, 2 MAGICC processes each
    python 301_running_helpers.py 10 4 2

Note:
    Total CPU usage = n_jobs * n_processes. For an 8-core machine, good options:
    - n_jobs=4, n_processes=2 (8 total)
    - n_jobs=2, n_processes=4 (8 total)
    - n_jobs=8, n_processes=1 (maximize notebook parallelism)
"""

import warnings
warnings.filterwarnings("ignore")

import concurrent.futures
import datetime
import logging
import os
import pathlib
import sys
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

# Session timestamp used for both logging and manifest files
SESSION_DATETIME = None
MANIFEST_ENTRIES = []


def setup_logging():
    """Set up logging - should only be called from main process."""
    global SESSION_DATETIME
    logging.basicConfig(level=logging.INFO)

    SESSION_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    fh = logging.FileHandler(LOGGING_FOLDER / f'{SESSION_DATETIME}_301_runs.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


def save_manifest():
    """Save the session manifest file with all completed runs."""
    if not MANIFEST_ENTRIES:
        return

    manifest_dir = OUTPUT_FOLDER / "individual_runs/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f'{SESSION_DATETIME}_manifest.csv'
    manifest_df = pd.DataFrame(MANIFEST_ENTRIES)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}")


def check_failed_runs(input_df):
    """
    Cross-reference completed runs against input to identify failures.

    Args:
        input_df: DataFrame with the original input scenarios

    Returns:
        DataFrame of failed runs (empty if all succeeded)
    """
    if not MANIFEST_ENTRIES:
        logger.warning("No manifest entries found - all runs may have failed")
        failed_df = input_df.copy()
    else:
        # Create sets for comparison using (model, scenario, magicc_flag) as key
        completed = {
            (entry["model"], entry["scenario"], entry["magicc_flag"])
            for entry in MANIFEST_ENTRIES
        }

        # Find runs that were in input but not in completed manifest
        failed_runs = []
        for row in input_df.itertuples():
            key = (row.model, row.scenario, row.magicc_flag)
            if key not in completed:
                failed_runs.append({
                    "MODEL": row.model,
                    "SCENARIO": row.scenario,
                    "ENSEMBLE_MEMBERS": int(row.ensemble),
                    "FILE": row.file,
                    "MAGICC_FLAG": row.magicc_flag
                })

        failed_df = pd.DataFrame(failed_runs)

    n_failed = len(failed_df)
    n_total = len(input_df)
    n_completed = n_total - n_failed

    logger.info(f"Run completion summary: {n_completed}/{n_total} succeeded, {n_failed} failed")

    if n_failed > 0:
        logger.warning(f"Failed runs ({n_failed}):")
        for _, row in failed_df.iterrows():
            logger.warning(f"  - {row['MODEL']} | {row['SCENARIO']} | {row['MAGICC_FLAG']}")

        # Save failed runs to CSV for re-running
        failed_output_path = OUTPUT_FOLDER / f"301_failed_runs_{SESSION_DATETIME}.csv"
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
        input_data: DataFrame with columns: model, scenario, ensemble, file, magicc_flag
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
            "MODEL": row.model,
            "SCENARIO": row.scenario,
            "ENSEMBLE_MEMBERS": int(row.ensemble),
            "FILE": row.file,
            "MAGICC_FLAG": row.magicc_flag,
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

    Args:
        config: Dict containing MODEL, SCENARIO, ENSEMBLE_MEMBERS, FILE,
                MAGICC_FLAG, and N_PROCESSES parameters for the notebook

    Returns:
        Dict with run metadata for the manifest
    """
    import warnings
    warnings.filterwarnings("ignore")

    OUTPUT_MODEL = sanitize_label(config['MODEL'])
    OUTPUT_SCENARIO = sanitize_label(config['SCENARIO'])
    pm.execute_notebook(
        pathlib.Path(current_path / "200_run_magicc.ipynb"),
        PAPERMILL_OUTPUT_FOLDER / f"301_{OUTPUT_MODEL}_{OUTPUT_SCENARIO}_{config['ENSEMBLE_MEMBERS']}_{config['MAGICC_FLAG']}.ipynb",
        parameters=config
    )

    # Return manifest entry for this run
    return {
        "model": config["MODEL"],
        "scenario": config["SCENARIO"],
        "ensemble_members": config["ENSEMBLE_MEMBERS"],
        "magicc_flag": config["MAGICC_FLAG"],
        "n_processes": config["N_PROCESSES"],
        "output_filename": f"{OUTPUT_MODEL}_{OUTPUT_SCENARIO}_{config['MAGICC_FLAG']}.csv",
    }

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
                result = run_papermill_notebook(conf)
                results.append(result)
                MANIFEST_ENTRIES.append(result)
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
                result = future.result()
                results.append(result)
                MANIFEST_ENTRIES.append(result)
            except Exception as e:
                results.append(e)

    return results

if __name__ == "__main__":
    setup_logging()

    if len(sys.argv) < 4:
        logger.error("Usage: python 301_running_helpers.py <batch_size> <n_jobs> <n_processes>")
        sys.exit(1)

    try:
        batch_size = int(sys.argv[1])
        n_jobs = int(sys.argv[2])
        n_processes = int(sys.argv[3])
    except ValueError:
        logger.error("batch_size, n_jobs, and n_processes must be integers")
        sys.exit(1)

    start_time = datetime.datetime.now().isoformat()
    logger.info(f'Starting runs at: {start_time}')
    logger.info(f'Configuration: batch_size={batch_size}, n_jobs={n_jobs}, n_processes={n_processes}')
    mod_scens = pd.read_csv(
        #OUTPUT_FOLDER / "300_trial_batches_v0.1.csv",
        OUTPUT_FOLDER / "300_trial_batches_AR6WG3IMP_v0.1.csv",
        header=0,
        names=["idx", "model", "scenario", "ensemble", "file", "magicc_flag"]
    )
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
    save_manifest()
    check_failed_runs(mod_scens)
