"""
New metrics functions for the AR7 net-zero assessment.

This module contains additional metric extraction and analysis functions
that extend the base metrics.py functionality.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
import dotenv
import pandas as pd
from tqdm import tqdm


def _process_single_file(args):
    """
    Process a single output file and extract GSAT metrics.

    This is a helper function for parallel processing.
    """
    output_file, model, scenario, magicc_flag = args

    if not output_file.exists():
        return []

    # Read CSV directly with pandas (much faster than pyam)
    df = pd.read_csv(output_file)

    # Filter for GSAT variable
    gsat_var = "Surface Temperature (GSAT)"
    df_gsat = df[df["variable"] == gsat_var]

    if df_gsat.empty:
        return []

    # Get year columns (numeric columns)
    year_cols = [c for c in df_gsat.columns if c.isdigit()]

    metrics = []
    for _, row in df_gsat.iterrows():
        run_id = row.get("run_id", 0)

        # Get values for all years
        values = row[year_cols].astype(float)

        # Calculate peak GSAT and year
        gsat_peak = values.max()
        gsat_peak_year = int(year_cols[values.argmax()])

        # Get 2100 value
        gsat_2100 = row.get("2100", float("nan"))
        if pd.isna(gsat_2100):
            gsat_2100 = float("nan")

        metrics.append({
            "model": model,
            "scenario": scenario,
            "magicc_flag": magicc_flag,
            "run_id": run_id,
            "gsat_peak": round(gsat_peak, 3),
            "gsat_peak_year": gsat_peak_year,
            "gsat_2100": round(float(gsat_2100), 3),
            "delta_warming": round(gsat_peak - float(gsat_2100), 3),
        })

    return metrics


def compile_gsat_metrics_from_manifest(
    manifest_filename: str,
    n_workers: int = 8
) -> pd.DataFrame:
    """
    Load all runs from a manifest file and compile peak GSAT and 2100 GSAT metrics.

    This function reads the environment file to find the output location,
    loads the manifest file from the individual_runs/manifests folder,
    reads all the run output files referenced in the manifest, and compiles
    peak GSAT and 2100 GSAT for all ensemble members into a single DataFrame.

    Parameters
    ----------
    manifest_filename : str
        Name of the manifest file (e.g., "2026-05-07_15-23_manifest.csv")
    n_workers : int
        Number of parallel workers for processing files (default: 8)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - model: Model name
        - scenario: Scenario name
        - magicc_flag: Run type (ALL, CO2, GHG)
        - run_id: Ensemble member ID
        - gsat_peak: Maximum GSAT value across all years
        - gsat_peak_year: Year of peak GSAT
        - gsat_2100: GSAT value at year 2100
        - delta_warming: Difference between peak and 2100 warming (gsat_peak - gsat_2100)
    """
    # Load environment variables
    dotenv.load_dotenv()

    # Set up paths
    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_dir = output_folder / "individual_runs" / "manifests"
    individual_runs_dir = output_folder / "individual_runs"

    # Read manifest file
    manifest_path = manifest_dir / manifest_filename
    manifest = pd.read_csv(manifest_path)

    # Prepare arguments for parallel processing
    args_list = [
        (
            individual_runs_dir / row["output_filename"],
            row["model"],
            row["scenario"],
            row["magicc_flag"]
        )
        for _, row in manifest.iterrows()
    ]

    # Process files in parallel
    all_metrics = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_file, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing runs"):
            try:
                metrics = future.result()
                all_metrics.extend(metrics)
            except Exception as e:
                args = futures[future]
                print(f"Error processing {args[0]}: {e}")

    # Compile into DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    return metrics_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics_new.py <manifest_filename> [n_workers]")
        print("Example: python metrics_new.py 2026-05-07_15-23_manifest.csv 8")
        sys.exit(1)

    manifest_filename = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    print(f"Processing manifest: {manifest_filename}")
    print(f"Using {n_workers} workers")
    metrics_df = compile_gsat_metrics_from_manifest(manifest_filename, n_workers)

    # Load environment to get output folder
    dotenv.load_dotenv()
    output_folder = Path(os.environ["OUTPUT_FOLDER"])

    # Extract identifier from manifest filename (remove _manifest.csv suffix)
    manifest_id = manifest_filename.replace("_manifest.csv", "")

    # Save compiled metrics
    output_path = output_folder / f"500_gsat_metrics_{manifest_id}.csv"
    metrics_df.to_csv(output_path, index=False)

    print(f"\nCompiled {len(metrics_df)} metric rows")
    print(f"Unique scenarios: {metrics_df.groupby(['model', 'scenario']).ngroups}")
    print(f"Output saved to: {output_path}")
