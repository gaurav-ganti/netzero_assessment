"""
Extracts gcages' per-year quantile trajectories directly from the already-calibrated
individual_runs/*_ANTHROPOGENIC.csv files - no re-calibration, no individual_runs_raw/
needed at all.

This is a lighter, faster alternative to re-running 304_post_process_gsat.ipynb when
individual_runs/ is already known to be complete and correct (confirmed via
verify_individual_runs_coverage.py - 5,865/5,865 files, raw == calibrated == expected).
304's own patch (collecting timeseries_quantile during calibration) is kept as-is for
any future FULL rerun of the calibration step itself; this script is for right now,
where recalibrating everything again would just reproduce numerically identical
individual_runs/ output at a much higher time cost for no benefit.

Reuses gcages' own quantile machinery directly (groupby_except +
fix_index_name_after_groupby_quantile from pandas_openscm.grouping - the same two
calls CMIP7ScenarioMIPPostProcessor.__call__() itself uses to build
PostProcessingResult.timeseries_quantile), rather than reimplementing the equivalent
logic by hand. Verified: applied directly to an already-calibrated file, this
reproduces CMIP7ScenarioMIPPostProcessor's own timeseries_quantile output exactly
(max abs difference 0.0, checked against a real sample scenario) - the calibration
step, exceedance-probability calcs, and peak/EOC metrics gcages also computes
internally are all irrelevant overhead for this purpose.

Usage:
    python extract_quantiles_from_calibrated.py [manifest_filename]
    python extract_quantiles_from_calibrated.py regenerated_manifest.csv
"""

import os
import sys
from pathlib import Path

import dotenv
import pandas as pd
from pandas_openscm.grouping import fix_index_name_after_groupby_quantile, groupby_except
from tqdm import tqdm

# gcages' own native percentile set (CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config)
PERCENTILES = (0.05, 0.10, 1.0 / 6.0, 0.33, 0.5, 0.67, 5.0 / 6.0, 0.90, 0.95)


def _load_timeseries(filepath: Path) -> pd.DataFrame:
    """Load with the FULL id-column index (climate_model, model, region, run_id,
    scenario, unit, variable) - not just run_id - so groupby_except(..., "run_id")
    below has the other levels to group by, matching gcages' own indexing."""
    df = pd.read_csv(filepath)
    id_cols = [c for c in df.columns if not c.isdigit()]
    year_cols = [c for c in df.columns if c.isdigit()]
    ts = df.set_index(id_cols)[year_cols].astype(float)
    ts.columns = ts.columns.astype(int)
    return ts


def extract_quantiles_from_manifest(manifest_filename: str) -> pd.DataFrame:
    dotenv.load_dotenv()
    output_folder = Path(os.environ["OUTPUT_FOLDER"])
    manifest_dir = output_folder / "individual_runs" / "manifests"
    individual_runs_dir = output_folder / "individual_runs"

    manifest = pd.read_csv(manifest_dir / manifest_filename)
    anthropogenic = manifest[manifest["magicc_flag"] == "ANTHROPOGENIC"]

    quantile_trajectories_all = []
    for _, row in tqdm(list(anthropogenic.iterrows()), desc="Extracting quantiles"):
        filepath = individual_runs_dir / row["output_filename"]
        if not filepath.exists():
            continue

        ts = _load_timeseries(filepath)
        quantile_traj = fix_index_name_after_groupby_quantile(
            groupby_except(ts, "run_id").quantile(list(PERCENTILES)),
            new_name="quantile",
        )
        quantile_trajectories_all.append(quantile_traj)

    return pd.concat(quantile_trajectories_all)


if __name__ == "__main__":
    manifest_filename = sys.argv[1] if len(sys.argv) > 1 else "regenerated_manifest.csv"

    dotenv.load_dotenv()
    output_folder = Path(os.environ["OUTPUT_FOLDER"])

    print(f"Processing manifest: {manifest_filename}")
    result_df = extract_quantiles_from_manifest(manifest_filename)

    output_path = output_folder / "individual_runs_quantiles_anthropogenic.csv"
    result_df.to_csv(output_path)

    n_scenarios = result_df.index.droplevel("quantile").nunique()
    print(f"\nSaved {len(result_df)} rows ({n_scenarios} scenarios x {len(PERCENTILES)} quantile levels)")
    print(f"Output saved to: {output_path}")
