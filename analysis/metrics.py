"""
Helper functions for compiling temperature metrics from MAGICC output.

This module provides functions for loading scenario data and extracting
temperature metrics at key points (net-zero CO2, 2100, etc.).
"""

import pandas as pd
import numpy as np
import pyam
import re 

from pathlib import Path

from utils import get_value_at_netzero, get_value_at_year


def load_runs_manifest(manifest_path):
    runs = pd.read_csv(Path(manifest_path))

    # Classify each row by variant and extract base scenario name
    def classify(scenario):
        if "_NZCO2" in scenario:
            return "NZCO2", re.sub(r"_NZCO2$", "", scenario)
        elif "_NZKyoto" in scenario:
            return "NZKyoto", re.sub(r"_NZKyoto$", "", scenario)
        else:
            return "base", scenario

    runs[["variant", "scenario_base"]] = pd.DataFrame(
        runs["scenario"].apply(classify).tolist(), index=runs.index
    )

    base_runs    = runs[runs["variant"] == "base"].copy()
    nzco2_runs   = runs[(runs["variant"] == "NZCO2") & (runs["magicc_flag"] == "CO2")].copy()
    nzkyoto_runs = runs[(runs["variant"] == "NZKyoto") & (runs["magicc_flag"] == "CO2")].copy()

    # Pivot base runs
    runs_pivoted = base_runs.pivot_table(
        index=["model", "scenario_base"],
        columns="magicc_flag",
        values=["ensemble_members", "output_filename"],
        aggfunc="first"
    )
    runs_pivoted.columns = ["_".join(col).strip() for col in runs_pivoted.columns]
    runs_pivoted = runs_pivoted.reset_index()

    # Merge NZCO2 (CO2-flag only)
    nzco2_merge = nzco2_runs[["model", "scenario_base", "output_filename"]].rename(
        columns={"output_filename": "output_filename_NZCO2_CO2"}
    )
    runs_pivoted = runs_pivoted.merge(nzco2_merge, on=["model", "scenario_base"], how="left")

    # Merge NZKyoto (CO2-flag only)
    nzkyoto_merge = nzkyoto_runs[["model", "scenario_base", "output_filename"]].rename(
        columns={"output_filename": "output_filename_NZKyoto_CO2"}
    )
    runs_pivoted = runs_pivoted.merge(nzkyoto_merge, on=["model", "scenario_base"], how="left")

    runs_pivoted = runs_pivoted.rename(columns={"scenario_base": "scenario"})
    runs_pivoted = runs_pivoted.set_index(["model", "scenario"])

    return runs_pivoted
    

def load_magicc_csv(path):
    meta_cols = ["model", "scenario", "region", "variable", "unit", "run_id", "climate_model"]
    raw = pd.read_csv(path)
    year_cols = [c for c in raw.columns if c not in meta_cols]
    long = raw.melt(
        id_vars=[c for c in meta_cols if c in raw.columns],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].astype(int)
    if "climate_model" in long.columns:
        long = long.drop(columns=["climate_model"])
    
    return pyam.IamDataFrame(long, index=["model", "scenario", "run_id"])
    
    
def load_scenario_data(model, scenario, runs_pivoted, output_dir):
    ix = (model, scenario)

    def load_ts(col):
        file_name = runs_pivoted.loc[ix, col]
        if pd.isna(file_name):
            return None
        raw = pd.read_csv(output_dir / file_name)
        meta_cols = ["model", "scenario", "region", "variable", "unit", "run_id", "climate_model"]
        year_cols = [c for c in raw.columns if c not in meta_cols]
        # Return wide format indexed by run_id
        return raw.set_index("run_id")[year_cols].astype(float)

    # "ANTHROPOGENIC" is this module's internal/output-facing label (kept as-is below and in
    # extract_metrics for downstream compatibility) - "ANTHROPOGENIC" is the actual
    # magicc_flag value the manifest pivot produces that column from.
    ts_all = load_ts("output_filename_ANTHROPOGENIC")
    ts_co2 = load_ts("output_filename_CO2")
    ts_ghg = load_ts("output_filename_GHG")
    ts_nzco2 = load_ts("output_filename_NZCO2_CO2")

    if ts_all is None or ts_co2 is None or ts_ghg is None:
        raise ValueError(f"Missing required ANTHROPOGENIC/CO2/GHG run for {model} / {scenario}")

    results = {
        "ANTHROPOGENIC":          ts_all,
        "CO2":          ts_co2,
        "GHG":          ts_ghg,
        "Non-CO2 GHG":  ts_ghg - ts_co2,
        "Residual":     ts_all - ts_ghg,
    }
    if ts_nzco2 is not None:
        results["CO2 [NZCO2]"]    = ts_nzco2
        results["CO2 [Negative]"] = ts_co2 - ts_nzco2

    # Stack into long format with variable as column
    dfs = []
    for var_name, ts in results.items():
        ts = ts.copy()
        ts.index.name = "run_id"
        ts = ts.reset_index()
        ts = ts.melt(id_vars="run_id", var_name="year", value_name="value")
        ts["variable"] = f"Surface Temperature (GSAT)|{var_name}"
        ts["model"] = model
        ts["scenario"] = scenario
        dfs.append(ts)

    return pd.concat(dfs, ignore_index=True)


def extract_metrics(df, variables, var_short_names, year_netzero=None):
    df["year"] = df["year"].astype(int)
    metrics_list = []

    for var in variables:
        short_name = var_short_names.get(var)
        if short_name is None or var not in df["variable"].unique():
            continue

        var_df = df[df["variable"] == var]
        wide = var_df.pivot_table(index="run_id", columns="year", values="value")

        # Value at net-zero CO2 year
        if year_netzero is not None and not pd.isna(year_netzero):
            nz_year = int(year_netzero)
            if nz_year in wide.columns:
                s = wide[nz_year].rename(f"{short_name}|NZCO2")
                metrics_list.append(s)

        # Value at 2100
        if 2100 in wide.columns:
            s = wide[2100].rename(f"{short_name}|2100")
            metrics_list.append(s)

        # Max across all years (ANTHROPOGENIC only)
        if var == "Surface Temperature (GSAT)|ANTHROPOGENIC":
            s = wide.max(axis=1).rename("GSAT|ANTHROPOGENIC|max")
            metrics_list.append(s)

    if not metrics_list:
        return pd.DataFrame()

    result = pd.concat(metrics_list, axis=1)
    result.index.name = "run_id"
    model = df["model"].iloc[0]
    scenario = df["scenario"].iloc[0]
    result.insert(0, "model", model)
    result.insert(1, "scenario", scenario)
    result = result.reset_index().set_index(["model", "scenario", "run_id"])

    return result


# Default variable configurations
DEFAULT_VARIABLES = [
    "Surface Temperature (GSAT)|ANTHROPOGENIC",
    "Surface Temperature (GSAT)|CO2 [NZCO2]",
    "Surface Temperature (GSAT)|CO2 [Negative]",
    "Surface Temperature (GSAT)|Non-CO2 GHG",
    "Surface Temperature (GSAT)|Residual",
]

DEFAULT_VAR_SHORT_NAMES = {
    "Surface Temperature (GSAT)|ANTHROPOGENIC": "GSAT|ANTHROPOGENIC",
    "Surface Temperature (GSAT)|CO2 [NZCO2]": "GSAT|CO2_NZCO2",
    "Surface Temperature (GSAT)|CO2 [Negative]": "GSAT|CO2_Negative",
    "Surface Temperature (GSAT)|Non-CO2 GHG": "GSAT|NonCO2_GHG",
    "Surface Temperature (GSAT)|Residual": "GSAT|Residual",
}
