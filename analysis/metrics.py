"""
Helper functions for compiling temperature metrics from MAGICC output.

This module provides functions for loading scenario data and extracting
temperature metrics at key points (net-zero CO2, 2100, etc.).
"""

import pandas as pd
import numpy as np
import pyam

from pathlib import Path

from utils import get_value_at_netzero, get_value_at_year


def load_runs_manifest(manifest_path):
    """
    Load and pivot a runs manifest file into a format suitable for load_scenario_data.

    This function reads a manifest CSV file containing MAGICC run information,
    separates base runs from NZCO2 runs, and pivots the data so that each
    model-scenario combination has columns for each run type's output filename.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Path to the manifest CSV file. The file should contain columns:
        - model: Model name
        - scenario: Scenario name (NZCO2 runs have '_NZCO2' suffix)
        - magicc_flag: Run type (ALL, CO2, GHG)
        - output_filename: Name of the output file
        - ensemble_members: Number of ensemble members
        - n_processes: Number of processes used

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by (model, scenario) with columns:
        - ensemble_members_ALL, ensemble_members_CO2, ensemble_members_GHG
        - n_processes_ALL, n_processes_CO2, n_processes_GHG
        - output_filename_ALL, output_filename_CO2, output_filename_GHG
        - output_filename_NZCO2
    """
    # Load manifest
    runs = pd.read_csv(Path(manifest_path))

    # Extract NZCO2 suffix info
    runs["is_nzco2"] = runs["scenario"].str.endswith("_NZCO2")
    runs["scenario_base"] = runs["scenario"].str.replace("_NZCO2$", "", regex=True)

    # Split into base runs and nzco2 runs
    base_runs = runs[~runs["is_nzco2"]].copy()
    nzco2_runs = runs[runs["is_nzco2"]][["model", "scenario_base", "output_filename"]].copy()
    nzco2_runs = nzco2_runs.rename(columns={"output_filename": "output_filename_NZCO2"})

    # Pivot base runs
    runs_pivoted = base_runs.pivot_table(
        index=["model", "scenario_base"],
        columns="magicc_flag",
        values=["ensemble_members", "n_processes", "output_filename"],
        aggfunc="first"
    )

    # Flatten column names
    runs_pivoted.columns = ["_".join(col).strip() for col in runs_pivoted.columns]
    runs_pivoted = runs_pivoted.reset_index()

    # Merge NZCO2 output filename
    runs_pivoted = runs_pivoted.merge(
        nzco2_runs.drop_duplicates(),
        on=["model", "scenario_base"],
        how="left"
    )

    # Rename scenario_base back to scenario
    runs_pivoted = runs_pivoted.rename(columns={"scenario_base": "scenario"})
    runs_pivoted = runs_pivoted.set_index(["model", "scenario"])

    return runs_pivoted


def load_scenario_data(model, scenario, runs_pivoted, output_dir):
    """
    Load all temperature data for a single model-scenario combination.

    Parameters
    ----------
    model : str
        Model name
    scenario : str
        Scenario name
    runs_pivoted : pandas.DataFrame
        DataFrame indexed by (model, scenario) with columns containing
        output filenames for each run type (ALL, CO2, GHG, NZCO2)
    output_dir : pathlib.Path
        Directory containing the output CSV files

    Returns
    -------
    pyam.IamDataFrame
        DataFrame with variables:
        - Surface Temperature (GSAT)|ALL
        - Surface Temperature (GSAT)|CO2
        - Surface Temperature (GSAT)|GHG
        - Surface Temperature (GSAT)|CO2 [NZCO2]
        - Surface Temperature (GSAT)|Non-CO2 GHG (derived)
        - Surface Temperature (GSAT)|CO2 [Negative] (derived)
        - Surface Temperature (GSAT)|Residual (derived)
    """
    ix = (model, scenario)
    dfs = []

    # Load ALL, CO2, GHG runs
    for v in ["ALL", "CO2", "GHG"]:
        file_name = runs_pivoted.loc[ix, f"output_filename_{v}"]
        _df = pyam.IamDataFrame(output_dir / file_name)
        _df.rename(
            variable={"Surface Temperature (GSAT)": f"Surface Temperature (GSAT)|{v}"},
            inplace=True
        )
        dfs.append(_df)

    # Load NZCO2 run
    nzco2_file = runs_pivoted.loc[ix, "output_filename_NZCO2"]
    if pd.notna(nzco2_file):
        df_nzco2 = pyam.IamDataFrame(output_dir / nzco2_file)
        df_nzco2.rename(
            variable={"Surface Temperature (GSAT)": "Surface Temperature (GSAT)|CO2 [NZCO2]"},
            inplace=True
        )
        # Remove _NZCO2 suffix from scenario name
        df_nzco2.rename(
            scenario={df_nzco2.scenario[0]: df_nzco2.scenario[0].replace("_NZCO2", "")},
            inplace=True
        )
        dfs.append(df_nzco2)

    # Concatenate all data
    df = pyam.concat(dfs)

    # Compute derived variables
    # Non-CO2 GHG = GHG - CO2
    df.subtract(
        a="Surface Temperature (GSAT)|GHG",
        b="Surface Temperature (GSAT)|CO2",
        name="Surface Temperature (GSAT)|Non-CO2 GHG",
        append=True
    )

    # CO2 [Negative] = CO2 - CO2 [NZCO2]
    if pd.notna(nzco2_file):
        df.subtract(
            a="Surface Temperature (GSAT)|CO2",
            b="Surface Temperature (GSAT)|CO2 [NZCO2]",
            name="Surface Temperature (GSAT)|CO2 [Negative]",
            append=True
        )

    # Residual = ALL - GHG
    df.subtract(
        a="Surface Temperature (GSAT)|ALL",
        b="Surface Temperature (GSAT)|GHG",
        name="Surface Temperature (GSAT)|Residual",
        append=True
    )

    return df


def extract_metrics(df, variables, var_short_names):
    """
    Extract temperature metrics at net-zero CO2 and 2100 for all variables.

    Parameters
    ----------
    df : pyam.IamDataFrame
        Input dataframe containing temperature timeseries data.
        Should have 'year_netzero_co2' in metadata for net-zero extraction.
    variables : list of str
        List of variable names to extract metrics for
    var_short_names : dict
        Mapping from full variable names to short names for output columns

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by (model, scenario, run_id) with columns for
        each variable at each time point:
        - {short_name}|NZCO2 - value at net-zero CO2 year
        - {short_name}|2100 - value at year 2100
        - GSAT|ALL|max - maximum GSAT|ALL value across all years
    """
    metrics_dfs = []

    for var in variables:
        short_name = var_short_names[var]

        # Skip if variable not present (e.g., CO2 [Negative] when no NZCO2 run)
        if var not in df.variable:
            continue

        # Value at net-zero CO2
        if "year_netzero_co2" in df.meta.columns:
            try:
                metrics_nz = get_value_at_netzero(
                    df=df,
                    variable=var,
                    netzero_meta_column="year_netzero_co2",
                    output_column=f"{short_name}|NZCO2",
                )
                metrics_dfs.append(metrics_nz)
            except Exception:
                pass

        # Value at 2100
        try:
            metrics_2100 = get_value_at_year(
                df=df,
                variable=var,
                year=2100,
                output_column=f"{short_name}|2100",
            )
            metrics_dfs.append(metrics_2100)
        except Exception:
            pass

    # Maximum GSAT|ALL across time
    if "Surface Temperature (GSAT)|ALL" in df.variable:
        try:
            ts = df.filter(variable="Surface Temperature (GSAT)|ALL").timeseries()
            max_vals = ts.max(axis=1)
            max_df = pd.DataFrame({
                "model": max_vals.index.get_level_values("model"),
                "scenario": max_vals.index.get_level_values("scenario"),
                "run_id": max_vals.index.get_level_values("run_id"),
                "GSAT|ALL|max": max_vals.values,
            }).set_index(["model", "scenario", "run_id"])
            metrics_dfs.append(max_df)
        except Exception:
            pass

    # Combine all metrics
    if metrics_dfs:
        return pd.concat(metrics_dfs, axis=1)
    else:
        return pd.DataFrame()


# Default variable configurations
DEFAULT_VARIABLES = [
    "Surface Temperature (GSAT)|ALL",
    "Surface Temperature (GSAT)|CO2 [NZCO2]",
    "Surface Temperature (GSAT)|CO2 [Negative]",
    "Surface Temperature (GSAT)|Non-CO2 GHG",
    "Surface Temperature (GSAT)|Residual",
]

DEFAULT_VAR_SHORT_NAMES = {
    "Surface Temperature (GSAT)|ALL": "GSAT|ALL",
    "Surface Temperature (GSAT)|CO2 [NZCO2]": "GSAT|CO2_NZCO2",
    "Surface Temperature (GSAT)|CO2 [Negative]": "GSAT|CO2_Negative",
    "Surface Temperature (GSAT)|Non-CO2 GHG": "GSAT|NonCO2_GHG",
    "Surface Temperature (GSAT)|Residual": "GSAT|Residual",
}
