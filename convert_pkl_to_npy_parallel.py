#!/usr/bin/env python3
import argparse
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _find_raw_root(in_root: Path):
    for parent in (in_root,) + tuple(in_root.parents):
        if parent.name == "_raw_datasets":
            return parent
    return None


def _default_out_root(in_root: Path) -> Path:
    raw_root = _find_raw_root(in_root)
    if raw_root is not None:
        return raw_root.parent / "datasets_npy"
    return in_root.parent / f"{in_root.name}_npy"


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
        description="Convert pickle datasets to .npy format in parallel.",
    )
    parser.add_argument(
        "--in-root",
        default="_raw_datasets",
        help="Root directory containing .pkl files (default: _raw_datasets)",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root for .npy files (default: datasets_npy mirroring _raw_datasets)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    parser.add_argument(
        "--allow-object",
        action="store_true",
        help="Allow object arrays (uses numpy pickle in .npy)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Log progress every N files (default: 200)",
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


def _as_array(obj, path: Path, allow_object: bool) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.dtype == object and not allow_object:
        raise ValueError(f"{path} contains object dtype; use --allow-object to save")
    return arr


def _convert_one(
    pkl_path: Path,
    in_root: Path,
    out_root: Path,
    overwrite: bool,
    allow_object: bool,
):
    raw_root = _find_raw_root(in_root)
    rel_path = pkl_path.relative_to(raw_root or in_root)
    rel_path = _map_out_rel_path(rel_path)
    out_path = (out_root / rel_path).with_suffix(".npy")
    if out_path.exists() and not overwrite:
        return "skip", str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = _as_array(_load_pkl(pkl_path), pkl_path, allow_object)
    np.save(out_path, arr, allow_pickle=allow_object)
    return "ok", str(out_path)


def main() -> int:
    args = _parse_args()
    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else _default_out_root(in_root)

    if not in_root.exists():
        raise SystemExit(f"Input root not found: {in_root}")

    pkl_files = sorted(set(in_root.rglob("*.pkl")) | set(in_root.rglob("*.pickle")))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found under: {in_root}")

    if not (0 < args.fraction <= 1.0):
        raise SystemExit("--fraction must be within (0, 1]")

    if args.fraction < 1.0:
        sample_size = max(1, int(len(pkl_files) * args.fraction))
        rng = random.Random(args.seed)
        pkl_files = rng.sample(pkl_files, sample_size)

    workers = args.workers or (os.cpu_count() or 1)
    total = len(pkl_files)
    print(
        f"Found {total} .pkl files under {in_root}. "
        f"Saving .npy to {out_root}. Workers={workers}."
    )

    start = time.perf_counter()
    done = 0
    skipped = 0
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _convert_one,
                pkl_path,
                in_root,
                out_root,
                args.overwrite,
                args.allow_object,
            )
            for pkl_path in pkl_files
        ]

        for future in as_completed(futures):
            try:
                status, payload = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                errors.append(str(exc))
            else:
                if status == "skip":
                    skipped += 1
                elif status == "ok":
                    done += 1
                else:
                    errors.append(payload)

            processed = done + skipped + len(errors)
            if processed % args.log_every == 0 or processed == total:
                elapsed = time.perf_counter() - start
                rate = processed / elapsed if elapsed else 0.0
                print(
                    f"{processed}/{total} files processed "
                    f"(done={done}, skipped={skipped}, errors={len(errors)}) "
                    f"[{rate:.2f} files/s]"
                )

    if errors:
        print("Conversion finished with errors. First 5 errors:")
        for msg in errors[:5]:
            print(f"  - {msg}")
        return 1

    print("Conversion finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
