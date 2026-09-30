"""
Sanity check before re-running 304: confirms individual_runs_raw/ and individual_runs/
have matching file coverage, and both match what 300_all_batches_v0.6.csv expects.

Purely diagnostic - reads and counts, changes nothing. Run this BEFORE re-running 304
(so a rerun can't yet have "fixed" any gap it would otherwise have reported), and
again AFTER, to confirm individual_runs/'s coverage is unchanged (still the same file
set - 304's rerun should be additive/identical, not fill in gaps that didn't already
get filled some other way).

Usage:
    python verify_individual_runs_coverage.py
"""

import os
from pathlib import Path
from collections import Counter

import dotenv
import pandas as pd

from utils import sanitize_label

dotenv.load_dotenv()
OUTPUT_FOLDER = Path(os.environ["OUTPUT_FOLDER"])
MAGICC_FLAGS = ("ANTHROPOGENIC", "CO2", "GHG")

raw_dir = OUTPUT_FOLDER / "individual_runs_raw"
calibrated_dir = OUTPUT_FOLDER / "individual_runs"

raw_files = {f.name for f in raw_dir.glob("*.csv")}
calibrated_files = {f.name for f in calibrated_dir.glob("*.csv")}

print(f"individual_runs_raw/:  {len(raw_files)} files")
print(f"individual_runs/:      {len(calibrated_files)} files")
print()

only_in_raw = raw_files - calibrated_files
only_in_calibrated = calibrated_files - raw_files

print(f"In raw but not calibrated: {len(only_in_raw)}")
if only_in_raw:
    for f in sorted(only_in_raw)[:20]:
        print(f"    {f}")
    if len(only_in_raw) > 20:
        print(f"    ... and {len(only_in_raw) - 20} more")

print(f"In calibrated but not raw: {len(only_in_calibrated)}")
if only_in_calibrated:
    for f in sorted(only_in_calibrated)[:20]:
        print(f"    {f}")
    if len(only_in_calibrated) > 20:
        print(f"    ... and {len(only_in_calibrated) - 20} more")

# Cross-reference against what the batch file actually expects
batch_path = OUTPUT_FOLDER / "300_all_batches_v0.6.csv"
if batch_path.exists():
    batch_df = pd.read_csv(batch_path, index_col=0)
    expected_files = set()
    for row in batch_df.itertuples():
        sanitized_model = sanitize_label(row.MODEL)
        sanitized_scenario = sanitize_label(row.SCENARIO)
        for flag in MAGICC_FLAGS:
            expected_files.add(f"{sanitized_model}_{sanitized_scenario}_{flag}.csv")

    print()
    print(f"300_all_batches_v0.6.csv expects: {len(batch_df)} scenarios x {len(MAGICC_FLAGS)} flags = {len(expected_files)} files")

    missing_from_raw = expected_files - raw_files
    missing_from_calibrated = expected_files - calibrated_files
    unexpected_in_raw = raw_files - expected_files
    unexpected_in_calibrated = calibrated_files - expected_files

    print(f"Expected but missing from raw:        {len(missing_from_raw)}")
    print(f"Expected but missing from calibrated:  {len(missing_from_calibrated)}")
    print(f"Present in raw but not expected:       {len(unexpected_in_raw)}")
    print(f"Present in calibrated but not expected: {len(unexpected_in_calibrated)}")

    for label, s in [
        ("Expected but missing from raw", missing_from_raw),
        ("Expected but missing from calibrated", missing_from_calibrated),
        ("Present in raw but not expected", unexpected_in_raw),
        ("Present in calibrated but not expected", unexpected_in_calibrated),
    ]:
        if s:
            print(f"\n{label} (first 20):")
            for f in sorted(s)[:20]:
                print(f"    {f}")
else:
    print(f"\n300_all_batches_v0.6.csv not found at {batch_path} - skipping expected-file cross-reference")

print()
if raw_files == calibrated_files and (not batch_path.exists() or (raw_files == expected_files == calibrated_files)):
    print("ALL CLEAR: raw, calibrated, and expected file sets are identical.")
else:
    print("MISMATCH FOUND - see details above before re-running 304.")
