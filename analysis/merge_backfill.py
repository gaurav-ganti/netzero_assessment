"""
Merge locally-backfilled ensemble members (see 303_run_magicc-missing.ipynb's Part B,
run on a Mac for its far lower MAGICC failure rate) into the cluster's raw
individual_runs_raw/ output.

Usage (run on the cluster, after untarring backfill_recovered.tar.gz into OUTPUT_FOLDER):
    uv run python merge_backfill.py
"""
import os
from pathlib import Path

import dotenv
import pandas as pd

dotenv.load_dotenv()
OUTPUT_FOLDER = Path(os.environ["OUTPUT_FOLDER"])

recovered_dir = OUTPUT_FOLDER / "backfill_recovered"
raw_dir = OUTPUT_FOLDER / "individual_runs_raw"

recovered_files = sorted(recovered_dir.glob("*.csv"))
print(f"Found {len(recovered_files)} recovered-row file(s) to merge")

merged_count = 0
skipped_count = 0

for recovered_path in recovered_files:
    output_path = raw_dir / recovered_path.name
    if not output_path.exists():
        print(f"  SKIP (no matching raw file): {recovered_path.name}")
        skipped_count += 1
        continue

    new_rows = pd.read_csv(recovered_path)
    id_cols = [c for c in new_rows.columns if not c.isdigit()]
    new_rows = new_rows.set_index(id_cols)
    new_rows.columns = new_rows.columns.astype(int)

    existing = pd.read_csv(output_path)
    existing = existing.set_index([c for c in existing.columns if not c.isdigit()])
    existing.columns = existing.columns.astype(int)

    combined = pd.concat([existing, new_rows])
    combined = combined[~combined.index.get_level_values("run_id").duplicated(keep="first")]
    combined = combined.sort_index(level="run_id")
    combined.to_csv(output_path)

    merged_count += 1

print(f"\nMerged {merged_count} file(s), skipped {skipped_count}")
