#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean/std for CBCT .npy files (2D arrays).",
    )
    parser.add_argument(
        "--root",
        default="datasets_npy/HQ/training_set",
        help=(
            "Root directory containing CBCT .npy files "
            "(default: datasets_npy/HQ/training_set)"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subdirectories (default: on)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive scan",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    paths = sorted(root.rglob("*.npy") if args.recursive else root.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found under: {root}")

    sum_ = 0.0
    sumsq = 0.0
    total = 0

    for p in paths:
        arr = np.load(p)
        if arr.ndim != 2:
            raise SystemExit(f"Unsupported shape {arr.shape} in {p}")
        arr = arr.astype(np.float64)
        sum_ += arr.sum()
        sumsq += (arr * arr).sum()
        total += arr.size

    mean = sum_ / total
    var = sumsq / total - mean * mean
    std = var ** 0.5

    print(f"files: {len(paths)}")
    print(f"total: {total}")
    print(f"mean={mean:.8f}")
    print(f"std={std:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
