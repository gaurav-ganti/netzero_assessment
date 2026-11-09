"""One-off local backfill: rerun missing members for the _NZCO2 rerun only.

Adapts 303_run_magicc-missing.ipynb's Part B harvest+QEXTRA logic, but does NOT
merge into individual_runs_raw/ (those files only exist on the cluster, not
locally). Instead writes recovered rows to a dedicated local folder,
`output_revision/backfill_recovered_nzco2_rerun/`, matching the
`{sanitized_model}_{sanitized_scenario}_{flag}.csv` naming convention used by
individual_runs_raw/, ready to be transferred to the cluster and merged there
per notes/magicc-unicc-cluster-runs-skill.md section 5.

Scoped to scenario names ending in "_NZCO2" only - the manifest also contains
7214 leftover rows from pre-existing tracking files that predate this rerun
(missing_members/ on the cluster wasn't actually empty beforehand), which must
NOT be backfilled here.
"""
import multiprocessing as _mp
try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass

import os
import platform
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import logging
logging.getLogger("pint.util").setLevel(logging.ERROR)

import dotenv
import pandas as pd
import pandas_indexing as pix
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.scm_running import CMIP7ScenarioMIPSCMRunner

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))
from utils import sanitize_label, write_magicc_extra_rf_file

dotenv.load_dotenv()
OUTPUT_REVISION = ANALYSIS_DIR / "output_revision"
RECOVERED_DIR = OUTPUT_REVISION / "backfill_recovered_nzco2_rerun"
RECOVERED_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_REVISION / "missing_members_manifest_nzco2rerun.csv"
BATCH_PATH = OUTPUT_REVISION / "300_all_batches_v0.6.csv"

# --- MAGICC exe / config setup (same as 303 cell 8) ---
BASE_DIR = ANALYSIS_DIR.parent.parent / "gcages/tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS = BASE_DIR / "history_cmip7_scenariomip.csv"
MAGICC_EXE_PATH = BASE_DIR / "magicc-v7.6.0a3" / "bin"
MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE = (
    BASE_DIR / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

if platform.system() == "Darwin" and platform.processor() == "arm":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc-darwin-arm64"
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/Cellar/gcc/15.2.0_1/lib/gcc/current"
elif platform.system() == "Linux":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc"
else:
    raise RuntimeError(f"Unhandled platform: {platform.system()}")

print(f"Using MAGICC_EXE: {MAGICC_EXE}")

N_PROCESSES = 8

CO2_EFFRF_COMPONENTS = ("Effective Radiative Forcing|CO2",)
KYOTO_EFFRF_COMPONENTS = (
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
)
QEXTRA_BASKETS = {"CO2": CO2_EFFRF_COMPONENTS, "GHG": KYOTO_EFFRF_COMPONENTS}
BASKET_DESCRIPTION = {
    "CO2": "CO2-only",
    "GHG": "Kyoto-gases-only (CO2+CH4+N2O+F-Gases, no Montreal halogens)",
}


def build_scm_runner_restricted(output_variables, run_ids):
    runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        n_processes=N_PROCESSES,
        magicc_exe_path=MAGICC_EXE,
        magicc_prob_distribution_path=MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE,
        historical_emissions_path=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS,
        output_variables=output_variables,
        batch_size_scenarios=15,
    )
    cfgs_by_id = {c["run_id"]: c for c in runner.climate_models_cfgs["MAGICC7"]}
    runner.climate_models_cfgs["MAGICC7"] = [cfgs_by_id[rid] for rid in sorted(run_ids)]
    return runner


def sum_effrf_components(erf_results, components):
    summed = (
        erf_results
        .loc[pix.isin(variable=list(components))]
        .groupby(level="run_id")
        .sum(numeric_only=True)
    )
    year_columns = [c for c in summed.columns if isinstance(c, int)]
    return summed[year_columns]


def main():
    missing_manifest = pd.read_csv(MANIFEST_PATH)
    missing_manifest = missing_manifest[
        missing_manifest["scenario"].astype(str).str.endswith("_NZCO2")
    ].reset_index(drop=True)
    print(f"Scoped to _NZCO2 scenarios: {len(missing_manifest)} missing rows across "
          f"{missing_manifest[['model', 'scenario']].drop_duplicates().shape[0]} scenarios")

    batch_df = pd.read_csv(BATCH_PATH, index_col=0)
    batch_lookup = batch_df.set_index(["MODEL", "SCENARIO"])

    backfill_summary = []

    groups = list(missing_manifest.groupby(["model", "scenario"]))
    for gi, ((model, scenario), group) in enumerate(groups):
        sanitized_model = sanitize_label(model)
        sanitized_scenario = sanitize_label(scenario)

        file_row = batch_lookup.loc[(model, scenario)]
        scenario_file = Path(file_row["FILE"])

        union_run_ids = sorted(group["run_id"].unique())
        missing_by_flag = {flag: set(sub) for flag, sub in group.groupby("flag")["run_id"]}

        print(f"\n=== [{gi + 1}/{len(groups)}] {model} | {scenario} - backfilling "
              f"{len(union_run_ids)} member(s), needed for flags: {sorted(missing_by_flag)} ===")

        start = load_timeseries_csv(
            scenario_file,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_columns_type=int,
            out_columns_name="year",
        )
        scenario_to_run = start.loc[pd.IndexSlice[model, scenario, :, :, :], :]

        erf_runner = build_scm_runner_restricted(
            output_variables=("Surface Air Temperature Change", *KYOTO_EFFRF_COMPONENTS),
            run_ids=union_run_ids,
        )
        erf_runner.climate_models_cfgs["MAGICC7"] = [
            {**c, "rf_total_runmodus": "ANTHROPOGENIC"}
            for c in erf_runner.climate_models_cfgs["MAGICC7"]
        ]
        erf_results = erf_runner(scenario_to_run)
        anthropogenic_gsat = erf_results.loc[pix.isin(variable=["Surface Air Temperature Change"])]

        survived_after_harvest = set(anthropogenic_gsat.index.get_level_values("run_id"))
        still_missing = set(union_run_ids) - survived_after_harvest
        if still_missing:
            print(f"  WARNING: {len(still_missing)} member(s) failed again even locally: "
                  f"{sorted(still_missing)} - will remain missing, needs manual follow-up.")

        qextra_effrf_by_flag = {
            flag: sum_effrf_components(erf_results, components)
            for flag, components in QEXTRA_BASKETS.items()
        }

        results_by_flag = {"ANTHROPOGENIC": anthropogenic_gsat}
        for flag, qextra_effrf in qextra_effrf_by_flag.items():
            qextra_forcing_dir = (
                OUTPUT_REVISION / "qextra_forcing" / f"{sanitized_model}_{sanitized_scenario}_{flag}_backfill"
            )
            for run_id, row in qextra_effrf.iterrows():
                write_magicc_extra_rf_file(
                    qextra_forcing_dir / f"member_{run_id}.IN",
                    years=list(row.index),
                    values=list(row.values),
                    description=(
                        f"{BASKET_DESCRIPTION[flag]} ERF backfill: {model} | {scenario}, "
                        f"ensemble member {run_id}"
                    ),
                )
            qextra_runner = build_scm_runner_restricted(
                output_variables=("Surface Air Temperature Change",),
                run_ids=list(qextra_effrf.index),
            )
            qextra_runner.climate_models_cfgs["MAGICC7"] = [
                {
                    **c,
                    "file_extra_rf": str(qextra_forcing_dir / f"member_{c['run_id']}.IN"),
                    "rf_extra_read": 1,
                    "rf_total_runmodus": "QEXTRA",
                }
                for c in qextra_runner.climate_models_cfgs["MAGICC7"]
            ]
            results_by_flag[flag] = qextra_runner(scenario_to_run)

        for flag, gsat_results in results_by_flag.items():
            if flag not in missing_by_flag:
                continue

            new_rows = gsat_results.loc[pix.isin(variable=["Surface Air Temperature Change"])]
            new_rows = new_rows.loc[
                new_rows.index.get_level_values("run_id").isin(missing_by_flag[flag])
            ]

            recovered_path = RECOVERED_DIR / f"{sanitized_model}_{sanitized_scenario}_{flag}.csv"
            new_rows.sort_index(level="run_id").to_csv(recovered_path)

            n_recovered = len(new_rows)
            n_still_missing = len(missing_by_flag[flag]) - n_recovered
            print(f"  [{flag}] recovered {n_recovered}/{len(missing_by_flag[flag])} member(s) "
                  f"-> {recovered_path}" + (f" ({n_still_missing} still missing)" if n_still_missing else ""))

            backfill_summary.append({
                "model": model, "scenario": scenario, "flag": flag,
                "needed": len(missing_by_flag[flag]), "recovered": n_recovered,
            })

    summary_df = pd.DataFrame(backfill_summary, columns=["model", "scenario", "flag", "needed", "recovered"])
    summary_path = RECOVERED_DIR / "_backfill_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    if summary_df.empty:
        print("\nNothing was backfilled (manifest was empty after _NZCO2 scoping).")
    else:
        print(f"\nBackfill complete: {summary_df['recovered'].sum()}/{summary_df['needed'].sum()} "
              f"member(s) recovered across {len(summary_df)} (scenario, flag) group(s).")
        print(f"Summary saved to: {summary_path}")
        incomplete = summary_df[summary_df["recovered"] < summary_df["needed"]]
        if not incomplete.empty:
            print(f"\n{len(incomplete)} (scenario, flag) group(s) still incomplete after local rerun:")
            print(incomplete.to_string(index=False))


if __name__ == "__main__":
    main()
