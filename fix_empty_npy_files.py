#!/usr/bin/env python3
"""
Fix empty .npy files by re-converting corresponding .pickle files.

This script reads a list of empty .npy files and re-converts them from
the corresponding .pickle files in _raw_datasets directory.

Usage:
    python fix_empty_npy_files.py [--dry-run]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

# List of empty .npy files that need to be re-converted
EMPTY_NPY_FILES = [
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/05.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/14.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/09.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/20.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/13.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/19.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/16.npy",
    "datasets_npy/HQ/training_set/2021-03-16_091349_FINISHED_Head/21.npy",
]

RAW_ROOT = Path("_raw_datasets")
NPY_ROOT = Path("datasets_npy")


def npy_to_pickle_path(npy_path: Path) -> Path:
    """
    Convert npy path to corresponding pickle path.

    datasets_npy/HQ/training_set/case/XX.npy
    -> _raw_datasets/HQ/training_set/case/XX.pickle
    """
    rel_path = npy_path.relative_to(NPY_ROOT)
    pickle_path = RAW_ROOT / rel_path.with_suffix(".pickle")
    return pickle_path


def load_pickle(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data)
    if arr.dtype == object:
        raise ValueError(f"{path} contains object dtype; cannot safely save as .npy")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Fix empty .npy files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    print(f"Processing {len(EMPTY_NPY_FILES)} empty .npy files...")

    success = 0
    failed = 0

    for npy_str in EMPTY_NPY_FILES:
        npy_path = Path(npy_str)
        pickle_path = npy_to_pickle_path(npy_path)

        print(f"\n[{npy_str}]")
        print(f"  Source: {pickle_path}")

        if not pickle_path.exists():
            print(f"  ERROR: Pickle file not found!")
            failed += 1
            continue

        if args.dry_run:
            print(f"  [DRY-RUN] Would convert and save to {npy_path}")
            success += 1
            continue

        try:
            arr = load_pickle(pickle_path)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, arr)
            print(f"  OK: Saved array with shape {arr.shape}, dtype {arr.dtype}")
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Summary: {success} success, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
