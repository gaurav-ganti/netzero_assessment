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

import papermill as pm
import pandas as pd
import sys

import pathlib
import time
import os

from tqdm import tqdm
from itertools import islice
import concurrent.futures

import logging
import datetime

from utils import sanitize_label

logger = logging.getLogger(__name__)

# Session timestamp used for both logging and manifest files
SESSION_DATETIME = None
MANIFEST_ENTRIES = []


def setup_logging():
    """Set up logging - should only be called from main process."""
    global SESSION_DATETIME
    logging.basicConfig(level=logging.INFO)

    SESSION_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_dir = pathlib.Path("logging")
    log_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(log_dir / f'{SESSION_DATETIME}_301_runs.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


def save_manifest():
    """Save the session manifest file with all completed runs."""
    if not MANIFEST_ENTRIES:
        return

    manifest_dir = pathlib.Path("output/individual_runs/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f'{SESSION_DATETIME}_manifest.csv'
    manifest_df = pd.DataFrame(MANIFEST_ENTRIES)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}")

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
        f"papermill_output/301_{OUTPUT_MODEL}_{OUTPUT_SCENARIO}_{config["ENSEMBLE_MEMBERS"]}_{config["MAGICC_FLAG"]}.ipynb",
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
        pathlib.Path("output/300_trial_batches_v0.1.csv"),
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
