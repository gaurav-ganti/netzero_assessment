"""Merge locally-recovered _NZCO2 backfill rows into the cluster's individual_runs_raw/.

Run this ON THE CLUSTER, after transferring the local
`output_revision/backfill_recovered_nzco2_rerun/` folder there (e.g. into
`$OUTPUT_FOLDER/backfill_recovered_nzco2_rerun/`).

Implements notes/magicc-unicc-cluster-runs-skill.md section 5's merge logic:
dedup by run_id (keep="first" - the recovered/backfilled row wins only when no
existing row for that run_id was already present), handling both an existing
raw file to merge into and a totally-failed unit with no existing raw file at
all.

Usage (from the analysis/ directory on the cluster):
    uv run python scripts/merge_303_backfill_nzco2.py
"""
import os
from pathlib import Path

import dotenv
import pandas as pd

dotenv.load_dotenv()
OUTPUT_FOLDER = Path(os.environ["OUTPUT_FOLDER"])

RECOVERED_DIR = OUTPUT_FOLDER / "backfill_recovered_nzco2_rerun"
RAW_DIR = OUTPUT_FOLDER / "individual_runs_raw"


def load_and_index(path):
    df = pd.read_csv(path)
    index_cols = [c for c in df.columns if not c.isdigit()]
    df = df.set_index(index_cols)
    df.columns = df.columns.astype(int)
    return df


def main():
    recovered_paths = sorted(RECOVERED_DIR.glob("*_NZCO2_*.csv"))
    print(f"Found {len(recovered_paths)} recovered file(s) under {RECOVERED_DIR}")

    n_new_files = 0
    n_merged = 0
    n_rows_added = 0

    for recovered_path in recovered_paths:
        new_rows = load_and_index(recovered_path)
        output_path = RAW_DIR / recovered_path.name

        if not output_path.exists():
            new_rows.sort_index(level="run_id").to_csv(output_path)
            n_new_files += 1
            n_rows_added += len(new_rows)
            print(f"  [NEW FILE] {recovered_path.name}: {len(new_rows)} row(s) -> {output_path}")
            continue

        existing = load_and_index(output_path)
        before = len(existing)
        combined = pd.concat([existing, new_rows])
        combined = combined[~combined.index.get_level_values("run_id").duplicated(keep="first")]
        combined = combined.sort_index(level="run_id")
        combined.to_csv(output_path)

        added = len(combined) - before
        n_merged += 1
        n_rows_added += added
        print(f"  [MERGED] {recovered_path.name}: {before} -> {len(combined)} row(s) "
              f"(+{added}, {len(new_rows) - added} already present)")

    print(f"\nDone: {n_merged} file(s) merged into existing raw output, "
          f"{n_new_files} new file(s) created (totally-failed units), "
          f"{n_rows_added} row(s) added in total.")
    print("\nNext: re-run 304_post_process_gsat.ipynb (calibration) and "
          "305_create_manifest_from_output.ipynb, then recompile the 500-series "
          "metrics for all four scenario groups.")


if __name__ == "__main__":
    main()
