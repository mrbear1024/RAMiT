#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np


def _default_out_dir(in_dir: Path) -> Path:
    parent = in_dir.parent
    if parent.name in ("", "."):
        return parent / f"{in_dir.name}_npy"
    return parent.parent / f"{parent.name}_npy" / in_dir.name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CBCT pickle files to .npy format.",
    )
    parser.add_argument(
        "--in-dir",
        default="datasets/CBCT",
        help="Input directory containing .pkl files (default: datasets/CBCT)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for .npy files (default: <parent>_npy/<in-dir-name>)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
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


def main() -> int:
    args = _parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(in_dir)

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    pkl_files = sorted(in_dir.rglob("*.pkl"))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found under: {in_dir}")

    for pkl_path in pkl_files:
        rel_path = pkl_path.relative_to(in_dir)
        out_path = (out_dir / rel_path) if out_dir else pkl_path
        out_path = out_path.with_suffix(".npy")
        if out_path.exists() and not args.overwrite:
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = _as_array(_load_pkl(pkl_path), pkl_path)
        np.save(out_path, arr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
