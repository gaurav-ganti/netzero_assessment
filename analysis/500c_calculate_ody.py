"""
500c: Overshoot Degree-Years (ODY) extraction.

Computes, per (model, scenario, run_id), the integral of GSAT above that member's own
NZCO2-year level, from the net-zero-CO2 year to 2100:

    ODY = sum over years t in [year_netzero_co2, 2100] of max(0, GSAT(t) - GSAT(year_netzero_co2))

Baseline is per-MEMBER (that specific ensemble member's own GSAT value at the
scenario's net-zero year), not a scenario-wide average - consistent with how
delta_netzero_co2 is computed elsewhere in this pipeline. The max(0, ...) makes this a
one-sided "excess warming" measure (standard "overshoot degree-years" convention,
e.g. as used for 1.5C overshoot in the literature) - years spent below the baseline
don't offset years spent above it. Rectangular (annual-step) integration, not
trapezoidal - negligible difference at annual resolution, simpler to reason about.

Unlike delta_netzero_co2 (which only reflects the *endpoint* difference between the
net-zero year and 2100), ODY reflects the *entire post-NZCO2 trajectory* - a member
that overshoots substantially for decades and only returns to baseline right at 2100
looks identical to one that never overshot at all under delta_netzero_co2, but has a
large ODY. It also gives members that never return by 2100 (the
"no_decline_after_nzco2" category elsewhere in this pipeline) a real, continuous
magnitude instead of just a binary flag.

Computed once per (model, scenario, run_id) here - all downstream statistics (median,
mean, percentiles, grouping by scenario_group, the two "NZCO2-only-not-NZGHG"
set-difference groups planned for later, etc.) happen locally afterward, same as the
rest of the 500-series.

Requires the full annual timeseries from individual_runs/ (calibrated GSAT), not just
the 500b component metrics' three scalar extractions (peak/netzero_co2/2100) - this is
why this needs its own extraction pass rather than being derivable from 500b alone.

Usage:
    python 500c_calculate_ody.py <manifest_filename> [n_workers] [netzero_timings_filename]
    python 500c_calculate_ody.py regenerated_manifest.csv 8 101_netzero_timings.csv
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_timeseries(filepath: Path) -> pd.DataFrame:
    """Load timeseries with run_id index and integer year columns. Identical to
    500_calculate_peak_2100_warming.py's helper of the same name."""
    df = pd.read_csv(filepath)
    year_cols = [c for c in df.columns if c.isdigit()]
    ts = df.set_index("run_id")[year_cols].astype(float)
    ts.columns = ts.columns.astype(int)
    return ts


def _process_single_scenario(args):
    """Compute ODY for every run_id in one scenario's ANTHROPOGENIC file."""
    individual_runs_dir, output_filename, model, scenario, nz_year = args

    if nz_year is None or pd.isna(nz_year):
        return []

    nz_year = int(nz_year)
    filepath = individual_runs_dir / output_filename
    if not filepath.exists():
        return []

    ts = _load_timeseries(filepath)

    year_cols = [y for y in ts.columns if nz_year <= y <= 2100]
    if nz_year not in ts.columns or not year_cols:
        return []

    window = ts[year_cols]
    baseline = ts[nz_year]  # per-run_id, that member's own GSAT at the net-zero year
    excess = window.sub(baseline, axis=0).clip(lower=0)
    ody = excess.sum(axis=1)

    return [
        {"model": model, "scenario": scenario, "run_id": run_id, "year_netzero_co2": nz_year, "ODY": val}
        for run_id, val in ody.items()
    ]


def compile_ody_from_manifest(manifest_filename: str, netzero_timings_path, n_workers: int = 8) -> pd.DataFrame:
    dotenv.load_dotenv()
    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_dir = output_folder / "individual_runs" / "manifests"
    individual_runs_dir = output_folder / "individual_runs"

    manifest = pd.read_csv(manifest_dir / manifest_filename)
    anthropogenic = manifest[manifest["magicc_flag"] == "ANTHROPOGENIC"]

    netzero_timings = pd.read_csv(netzero_timings_path)
    netzero_lookup = netzero_timings.set_index(["model", "scenario"])["year_netzero_co2"].to_dict()

    args_list = []
    for _, row in anthropogenic.iterrows():
        model, scenario = row["model"], row["scenario"]
        base_scenario = scenario.replace("_NZCO2", "").replace("_NZKyoto", "")
        nz_year = netzero_lookup.get((model, base_scenario))
        args_list.append((individual_runs_dir, row["output_filename"], model, scenario, nz_year))

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_scenario, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing ODY"):
            all_rows.extend(future.result())

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 500c_calculate_ody.py <manifest_filename> [n_workers] [netzero_timings_filename]")
        print("Example: python 500c_calculate_ody.py regenerated_manifest.csv 8 101_netzero_timings.csv")
        sys.exit(1)

    manifest_filename = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    netzero_timings_filename = sys.argv[3] if len(sys.argv) > 3 else "101_netzero_timings.csv"

    dotenv.load_dotenv()
    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_id = manifest_filename.replace("_manifest.csv", "")
    netzero_timings_path = output_folder / netzero_timings_filename

    print(f"Processing manifest: {manifest_filename}")
    print(f"Using {n_workers} workers")
    ody_df = compile_ody_from_manifest(manifest_filename, netzero_timings_path, n_workers)

    output_path = output_folder / f"500c_ody_{manifest_id}.csv"
    ody_df.to_csv(output_path, index=False)

    print(f"\nCompiled {len(ody_df)} ODY rows")
    print(f"Unique scenarios: {ody_df.groupby(['model', 'scenario']).ngroups}")
    print(f"Output saved to: {output_path}")
