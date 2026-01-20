#!/usr/bin/env python3
import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np


def _find_raw_root(in_dir: Path):
    for parent in (in_dir,) + tuple(in_dir.parents):
        if parent.name == "_raw_datasets":
            return parent
    return None


def _default_out_root(in_dir: Path) -> Path:
    raw_root = _find_raw_root(in_dir)
    if raw_root is not None:
        return raw_root.parent / "datasets_npy"
    parent = in_dir.parent
    if parent.name in ("", "."):
        return parent / f"{in_dir.name}_npy"
    return parent.parent / f"{parent.name}_npy"


def _normalize_split_name(name: str) -> str:
    if name == "valiadation_set":
        return "validation_set"
    return name


def _map_out_rel_path(rel_path: Path) -> Path:
    parts = rel_path.parts
    split_names = {"training_set", "validation_set", "valiadation_set", "test_set"}
    if len(parts) >= 2:
        if parts[0] in ("HQ", "LQ") and parts[1] in split_names:
            split = _normalize_split_name(parts[1])
            return Path(parts[0]) / split / Path(*parts[2:])
        if parts[0] in split_names and parts[1] in ("HQ", "LQ"):
            split = _normalize_split_name(parts[0])
            return Path(parts[1]) / split / Path(*parts[2:])
    return rel_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CBCT pickle files to .npy format.",
    )
    parser.add_argument(
        "--in-dir",
        default="_raw_datasets",
        help=(
            "Input directory containing .pickle/.pkl files "
            "(default: _raw_datasets)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output root for .npy files (default: datasets_npy mirroring _raw_datasets)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List conversions without writing .npy files",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Log progress every N files (default: 200)",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help=(
            "Limit to N case directories per split (training/validation/test). "
            "Default: no limit"
        ),
    )
    parser.add_argument(
        "--case-suffix",
        type=str,
        default="_FINISHED_Head",
        help="Case directory name suffix filter (default: _FINISHED_Head)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of files to convert (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sampling when --fraction < 1",
    )
    return parser.parse_args()


def _load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _as_array(obj, path: Path) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.dtype == object:
        raise ValueError(f"{path} contains object dtype; cannot safely save as .npy")
    return arr


def _iter_pickle_paths(root: Path):
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith((".pkl", ".pickle")):
                yield Path(dirpath) / name


def _list_split_dirs(root: Path):
    split_names = {"training_set", "validation_set", "valiadation_set", "test_set"}
    if root.name in split_names:
        return [root]
    split_dirs = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        for split_name in split_names.intersection(dirnames):
            split_dirs.append(Path(dirpath) / split_name)
    return sorted(set(split_dirs))


def _select_case_dirs(root: Path, limit: int, suffix: str):
    split_dirs = _list_split_dirs(root)
    if not split_dirs:
        raise SystemExit(
            f"No split directories found under {root} (expected training_set/validation_set/test_set)"
        )

    selected = []
    for split_dir in split_dirs:
        candidates = [
            p
            for p in split_dir.iterdir()
            if p.is_dir() and (not suffix or p.name.endswith(suffix))
        ]
        candidates = sorted(candidates)
        if not candidates:
            raise SystemExit(f"No matching case directories found under: {split_dir}")
        selected.extend(candidates[:limit])
    return selected


def main() -> int:
    args = _parse_args()
    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir) if args.out_dir else _default_out_root(in_dir)
    raw_root = _find_raw_root(in_dir)

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    if args.limit_per_split is not None:
        if args.limit_per_split <= 0:
            raise SystemExit("--limit-per-split must be a positive integer")
        selected_dirs = _select_case_dirs(in_dir, args.limit_per_split, args.case_suffix)
        pkl_files = sorted(
            {path for case_dir in selected_dirs for path in _iter_pickle_paths(case_dir)}
        )
        print(
            f"Selected {len(selected_dirs)} case directories with suffix "
            f"'{args.case_suffix}' (limit per split={args.limit_per_split})."
        )
    else:
        pkl_files = sorted(_iter_pickle_paths(in_dir))
    if not pkl_files:
        raise SystemExit(f"No .pkl/.pickle files found under: {in_dir}")

    if args.log_every <= 0:
        raise SystemExit("--log-every must be a positive integer")

    if not (0 < args.fraction <= 1.0):
        raise SystemExit("--fraction must be within (0, 1]")

    total = len(pkl_files)
    if args.fraction < 1.0:
        sample_size = max(1, int(total * args.fraction))
        rng = random.Random(args.seed)
        pkl_files = rng.sample(pkl_files, sample_size)
        print(f"Sampling {len(pkl_files)}/{total} files (fraction={args.fraction}).")
        total = len(pkl_files)
    print(f"Found {total} .pkl/.pickle files under {in_dir}.")
    print(f"Saving .npy files to: {out_root}")

    converted = 0
    skipped = 0
    processed = 0

    def log_progress() -> None:
        if processed % args.log_every == 0 or processed == total:
            status = "planned" if args.dry_run else "done"
            print(
                f"{processed}/{total} files processed "
                f"({status}={converted}, skipped={skipped})"
            )

    for pkl_path in pkl_files:
        rel_path = pkl_path.relative_to(raw_root or in_dir)
        rel_path = _map_out_rel_path(rel_path)
        out_path = out_root / rel_path
        out_path = out_path.with_suffix(".npy")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            processed += 1
            log_progress()
            continue

        if args.dry_run:
            converted += 1
            processed += 1
            log_progress()
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = _as_array(_load_pkl(pkl_path), pkl_path)
        np.save(out_path, arr)
        converted += 1
        processed += 1
        log_progress()

    if args.dry_run:
        print(f"Dry run complete. planned={converted}, skipped={skipped}.")
    else:
        print(f"Conversion complete. done={converted}, skipped={skipped}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
