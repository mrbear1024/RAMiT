#!/usr/bin/env python3
"""Compute grayscale mean/std for .npy image arrays (normalized to [0,1]).

This matches the training path: np.load(...) -> /255 -> (optional) RGB->Gray.
"""
import argparse
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute grayscale mean/std for .npy images under a root directory.",
    )
    parser.add_argument(
        "--root",
        default="datasets/CBCT/HQ_HEAD",
        help="Root directory to scan for .npy files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories",
    )
    parser.add_argument(
        "--rgb-to-gray",
        action="store_true",
        default=True,
        help="Convert RGB to gray before statistics (default: on)",
    )
    parser.add_argument(
        "--no-rgb-to-gray",
        dest="rgb_to_gray",
        action="store_false",
        help="Disable RGB->gray conversion; use first channel instead",
    )
    return parser.parse_args()


def _to_hwc(arr: np.ndarray, path: Path) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Unsupported ndim={arr.ndim} for {path}")

    # Heuristic: treat CHW if the first dim looks like channels.
    if arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"Cannot infer channel axis for {path} with shape {arr.shape}")

    return arr


def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    # Match OpenCV COLOR_RGB2GRAY coefficients.
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def load_as_gray(path: Path, rgb_to_gray: bool) -> np.ndarray:
    arr = np.load(path)
    arr = _to_hwc(arr, path)

    if arr.ndim == 2:
        return arr

    # Now HWC
    if arr.shape[2] == 1:
        return arr[..., 0]

    if arr.shape[2] == 4:
        arr = arr[..., :3]  # drop alpha

    if rgb_to_gray:
        return _rgb_to_gray(arr)

    # Fall back to the first channel if RGB->gray is disabled.
    return arr[..., 0]


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    if args.recursive:
        paths = sorted(root.rglob("*.npy"))
    else:
        paths = sorted(root.glob("*.npy"))

    if not paths:
        raise SystemExit(f"No .npy files found under: {root}")

    total = 0
    sum_ = 0.0
    sumsq = 0.0

    for p in paths:
        arr = load_as_gray(p, args.rgb_to_gray).astype(np.float64) / 255.0
        sum_ += arr.sum()
        sumsq += (arr * arr).sum()
        total += arr.size

    mean = sum_ / total
    std = (sumsq / total - mean * mean) ** 0.5

    print(f"mean: {mean:.6f}")
    print(f"std: {std:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
