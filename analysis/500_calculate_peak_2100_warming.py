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
import numpy as np
from tqdm import tqdm


def _process_single_file(args):
    """
    Process a single output file and extract GSAT metrics.

    This is a helper function for parallel processing.
    """
    output_file, model, scenario, magicc_flag = args

    if not output_file.exists():
        return []

    df = pd.read_csv(output_file)

    gsat_var = "Surface Temperature (GSAT)"
    df_gsat = df[df["variable"] == gsat_var]

    if df_gsat.empty:
        return []

    year_cols = [c for c in df_gsat.columns if c.isdigit()]

    metrics = []
    for _, row in df_gsat.iterrows():
        run_id = row.get("run_id", 0)

        values = row[year_cols].astype(float)

        gsat_peak = values.max()
        gsat_peak_year = int(year_cols[values.argmax()])

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

    (Unchanged from before — produces the long-format ANTHROPOGENIC/CO2/GHG per-run_id metrics
    that 501 and 502 currently rely on.)
    """
    dotenv.load_dotenv()

    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_dir = output_folder / "individual_runs" / "manifests"
    individual_runs_dir = output_folder / "individual_runs"

    manifest_path = manifest_dir / manifest_filename
    manifest = pd.read_csv(manifest_path)

    args_list = [
        (
            individual_runs_dir / row["output_filename"],
            row["model"],
            row["scenario"],
            row["magicc_flag"]
        )
        for _, row in manifest.iterrows()
    ]

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

    metrics_df = pd.DataFrame(all_metrics)

    return metrics_df


def _load_timeseries(filepath: Path) -> pd.DataFrame:
    """Load timeseries with run_id index and integer year columns."""
    df = pd.read_csv(filepath)
    year_cols = [c for c in df.columns if c.isdigit()]
    ts = df.set_index("run_id")[year_cols].astype(float)
    ts.columns = ts.columns.astype(int)
    return ts


def _get_file_lookup_keys(model: str, scenario: str) -> dict:
    """
    Determine file lookup keys for a scenario's component runs.

    Returns dict with keys: ANTHROPOGENIC, CO2, GHG, NZCO2 (value is the manifest lookup key).
    "ANTHROPOGENIC" is this module's internal/output-facing label (kept as-is throughout, for
    downstream compatibility) - the manifest's actual magicc_flag value it maps to is
    "ANTHROPOGENIC".
    """
    keys = {
        "ANTHROPOGENIC": f"{model}|{scenario}|ANTHROPOGENIC",
        "CO2": f"{model}|{scenario}|CO2",
        "GHG": f"{model}|{scenario}|GHG",
    }

    if "_NZCO2" in scenario:
        keys["NZCO2"] = f"{model}|{scenario}|CO2"
    elif "_NZKyoto" in scenario:
        base_scenario = scenario.replace("_NZKyoto", "")
        keys["NZCO2"] = f"{model}|{base_scenario}_NZCO2|CO2"
    else:
        keys["NZCO2"] = f"{model}|{scenario}_NZCO2|CO2"

    return keys


def compile_component_metrics_from_manifest(
    manifest_filename: str,
    netzero_timings_path: str | Path,
) -> pd.DataFrame:
    """
    Compile component-contribution GSAT metrics (CO2_NZCO2, CO2_Negative, NonCO2_GHG,
    Residual) for all "ANTHROPOGENIC"-flagged scenarios in a manifest, extracted at BOTH the
    per-run_id peak-GSAT year and the scenario-level net-zero-CO2 year.

    This produces a wide-format DataFrame, one row per (model, scenario, run_id),
    with columns suffixed "|peak_gsat" and "|netzero_co2" for each component, so
    downstream notebooks (501, 502, 503) can choose which extraction basis to use
    without needing to re-run extraction.

    Parameters
    ----------
    manifest_filename : str
        Name of the manifest file (e.g., "2026-05-27_09-04_manifest.csv").
    netzero_timings_path : str or Path
        Path to the net-zero timings CSV (e.g. "101_netzero_timings.csv"), which
        must have columns: model, scenario, year_netzero_co2.

    Returns
    -------
    pd.DataFrame
        Columns: model, scenario, run_id, peak_year,
        GSAT|ANTHROPOGENIC|peak, GSAT|ANTHROPOGENIC|2100,
        GSAT|{component}|peak_gsat, GSAT|{component}|netzero_co2,
        GSAT|{component}|delta_peak_gsat, GSAT|{component}|delta_netzero_co2
        for component in [CO2_NZCO2, CO2_Negative, NonCO2_GHG, Residual].
    """
    dotenv.load_dotenv()

    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_path = output_folder / "individual_runs" / "manifests" / manifest_filename
    individual_runs_dir = output_folder / "individual_runs"

    manifest = pd.read_csv(manifest_path)
    manifest["key"] = manifest["model"] + "|" + manifest["scenario"] + "|" + manifest["magicc_flag"]
    file_lookup = manifest.set_index("key")["output_filename"].to_dict()

    netzero_timings = pd.read_csv(netzero_timings_path)
    netzero_lookup = netzero_timings.set_index(["model", "scenario"])["year_netzero_co2"].to_dict()

    all_scenarios = manifest[manifest["magicc_flag"] == "ANTHROPOGENIC"][["model", "scenario"]].drop_duplicates()

    results = []
    skipped = []

    for _, row in tqdm(all_scenarios.iterrows(), total=len(all_scenarios), desc="Processing component metrics"):
        model, scenario = row["model"], row["scenario"]

        keys = _get_file_lookup_keys(model, scenario)
        files = {k: file_lookup.get(keys[k]) for k in ["ANTHROPOGENIC", "CO2", "GHG", "NZCO2"]}

        missing = [k for k in ["ANTHROPOGENIC", "CO2", "GHG"] if files[k] is None]
        if missing:
            skipped.append({"model": model, "scenario": scenario, "missing": missing})
            continue

        has_nzco2 = files["NZCO2"] is not None
        if not has_nzco2:
            skipped.append({"model": model, "scenario": scenario, "missing": ["NZCO2"]})

        ts = {k: _load_timeseries(individual_runs_dir / files[k]) for k in ["ANTHROPOGENIC", "CO2", "GHG"]}
        if has_nzco2:
            ts["NZCO2"] = _load_timeseries(individual_runs_dir / files["NZCO2"])

        ts["NonCO2_GHG"] = ts["GHG"] - ts["CO2"]
        ts["Residual"] = ts["ANTHROPOGENIC"] - ts["GHG"]
        if has_nzco2:
            ts["CO2_Negative"] = ts["CO2"] - ts["NZCO2"]

        # Per-run_id peak GSAT|ANTHROPOGENIC year and value
        all_arr = ts["ANTHROPOGENIC"].values
        all_years = ts["ANTHROPOGENIC"].columns.values
        run_ids = ts["ANTHROPOGENIC"].index.values
        peak_idx = all_arr.argmax(axis=1)
        peak_years = all_years[peak_idx]
        peak_values = all_arr[np.arange(len(all_arr)), peak_idx]

        # Net-zero CO2 year (scenario-level, broadcast across run_ids)
        base_scenario = scenario.replace("_NZCO2", "").replace("_NZKyoto", "")
        nz_year = netzero_lookup.get((model, base_scenario))
        has_nz_year = nz_year is not None and not pd.isna(nz_year)
        nz_year_int = int(nz_year) if has_nz_year else None
        
        # Per-scenario GSAT|ANTHROPOGENIC in NZCO2 year (same year for all run_ids)
        if has_nz_year and nz_year_int in all_years:
            nz_col_idx = np.where(all_years == nz_year_int)[0][0]
            nz_values = all_arr[:, nz_col_idx]
        else:
            nz_values = np.full(len(all_arr), np.nan)
            
        result = pd.DataFrame({
            "model": model,
            "scenario": scenario,
            "run_id": run_ids,
            "peak_year": peak_years,
            "year_netzero_co2": nz_year_int,
            "GSAT|ANTHROPOGENIC|peak": peak_values,
            "GSAT|ANTHROPOGENIC|netzero_co2": nz_values,
            "GSAT|ANTHROPOGENIC|2100": ts["ANTHROPOGENIC"][2100].values,
        })

        components = [("NonCO2_GHG", "NonCO2_GHG"), ("Residual", "Residual")]
        if has_nzco2:
            components += [("CO2_NZCO2", "NZCO2"), ("CO2_Negative", "CO2_Negative")]

        for comp_name, ts_key in components:
            comp_ts = ts[ts_key]

            # Extraction at per-run_id peak GSAT year
            comp_arr = comp_ts.reindex(columns=all_years).values
            peak_vals = comp_arr[np.arange(len(comp_arr)), peak_idx]
            result[f"GSAT|{comp_name}|peak_gsat"] = peak_vals
            result[f"GSAT|{comp_name}|2100"] = comp_ts[2100].values
            result[f"GSAT|{comp_name}|delta_peak_gsat"] = comp_ts[2100].values - peak_vals

            # Extraction at scenario-level net-zero CO2 year
            if has_nz_year and nz_year_int in comp_ts.columns:
                nz_vals = comp_ts[nz_year_int].values
                result[f"GSAT|{comp_name}|netzero_co2"] = nz_vals
                result[f"GSAT|{comp_name}|delta_netzero_co2"] = comp_ts[2100].values - nz_vals
            else:
                result[f"GSAT|{comp_name}|netzero_co2"] = np.nan
                result[f"GSAT|{comp_name}|delta_netzero_co2"] = np.nan

        results.append(result)

    component_metrics_df = pd.concat(results, ignore_index=True)

    if skipped:
        print(f"Skipped/partial for {len(skipped)} scenarios (see returned skipped list).")

    return component_metrics_df, pd.DataFrame(skipped)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics_new.py <manifest_filename> [n_workers] [netzero_timings_filename]")
        print("Example: python metrics_new.py 2026-05-07_15-23_manifest.csv 8 101_netzero_timings.csv")
        print()
        print("If netzero_timings_filename is omitted, only the original ANTHROPOGENIC/CO2/GHG")
        print("peak/2100 metrics are compiled (500_gsat_metrics_*.csv).")
        print("If provided, component-contribution metrics (CO2_NZCO2, CO2_Negative,")
        print("NonCO2_GHG, Residual) are additionally compiled and saved to")
        print("500b_component_metrics_*.csv, with both peak-GSAT-year and")
        print("net-zero-CO2-year extraction bases included.")
        sys.exit(1)

    manifest_filename = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    netzero_timings_filename = sys.argv[3] if len(sys.argv) > 3 else None

    dotenv.load_dotenv()
    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_id = manifest_filename.replace("_manifest.csv", "")

    print(f"Processing manifest: {manifest_filename}")
    print(f"Using {n_workers} workers")
    metrics_df = compile_gsat_metrics_from_manifest(manifest_filename, n_workers)

    output_path = output_folder / f"500_gsat_metrics_{manifest_id}.csv"
    metrics_df.to_csv(output_path, index=False)

    print(f"\nCompiled {len(metrics_df)} metric rows")
    print(f"Unique scenarios: {metrics_df.groupby(['model', 'scenario']).ngroups}")
    print(f"Output saved to: {output_path}")

    if netzero_timings_filename is not None:
        netzero_timings_path = output_folder / netzero_timings_filename
        print(f"\nProcessing component metrics using net-zero timings: {netzero_timings_path}")
        component_metrics_df, skipped_df = compile_component_metrics_from_manifest(
            manifest_filename, netzero_timings_path
        )

        component_output_path = output_folder / f"500b_component_metrics_{manifest_id}.csv"
        component_metrics_df.to_csv(component_output_path, index=False)

        print(f"\nCompiled {len(component_metrics_df)} component metric rows")
        print(f"Unique scenarios: {component_metrics_df.groupby(['model', 'scenario']).ngroups}")
        print(f"Output saved to: {component_output_path}")

        if not skipped_df.empty:
            skipped_output_path = output_folder / f"500b_skipped_{manifest_id}.csv"
            skipped_df.to_csv(skipped_output_path, index=False)
            print(f"Skipped/partial scenarios ({len(skipped_df)}) saved to: {skipped_output_path}")