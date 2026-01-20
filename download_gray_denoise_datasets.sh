#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${ROOT_DIR}/_raw_datasets"
TESTSETS_DIR="${ROOT_DIR}/../testsets"

mkdir -p "$RAW_DIR" "$TESTSETS_DIR"

has_npy() {
  local dir="$1"
  shopt -s nullglob
  local files=("$dir"/*.npy)
  shopt -u nullglob
  [ ${#files[@]} -gt 0 ]
}

SET12_DST="${TESTSETS_DIR}/Set12/HQ"
CBSD68_DST="${TESTSETS_DIR}/CBSD68/HQ"
URBAN_DST="${TESTSETS_DIR}/urban100/HQ"

NEED_FFDNET=0
if ! has_npy "$SET12_DST" || ! has_npy "$CBSD68_DST"; then
  NEED_FFDNET=1
fi

NEED_URBAN=0
if ! has_npy "$URBAN_DST"; then
  NEED_URBAN=1
fi

if [ "$NEED_FFDNET" -eq 0 ] && [ "$NEED_URBAN" -eq 0 ]; then
  echo "All target testsets already exist under $TESTSETS_DIR. Skip download."
  exit 0
fi

# 1) FFDNet testsets: Set12, CBSD68
if [ "$NEED_FFDNET" -eq 1 ]; then
  if [ ! -d "$RAW_DIR/FFDNet/testsets" ]; then
    git clone --depth 1 https://github.com/cszn/FFDNet.git "$RAW_DIR/FFDNet"
  fi
fi

SET12_SRC="$RAW_DIR/FFDNet/testsets/Set12"
CBSD68_SRC="$RAW_DIR/FFDNet/testsets/CBSD68"

# 2) EDSR benchmark: Urban100
EDSR_TAR="$RAW_DIR/benchmark.tar"
URBAN_SRC=""
URBAN_HR=""
if [ "$NEED_URBAN" -eq 1 ]; then
  if [ ! -f "$EDSR_TAR" ]; then
    curl -L -o "$EDSR_TAR" http://cv.snu.ac.kr/research/EDSR/benchmark.tar
  fi
  URBAN_SRC="$(find "$RAW_DIR" -type d -name Urban100 | head -n 1)"
  if [ -z "$URBAN_SRC" ]; then
    tar -xf "$EDSR_TAR" -C "$RAW_DIR"
    URBAN_SRC="$(find "$RAW_DIR" -type d -name Urban100 | head -n 1)"
  fi
  if [ -z "$URBAN_SRC" ]; then
    echo "Urban100 not found after extracting benchmark.tar" >&2
    exit 1
  fi
  if [ -d "$URBAN_SRC/HR" ]; then
    URBAN_HR="$URBAN_SRC/HR"
  else
    URBAN_HR="$URBAN_SRC"
  fi
fi

export SET12_SRC CBSD68_SRC URBAN_HR TESTSETS_DIR

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ramit-env

python3 - <<'PY'
import os
import sys
import glob
import numpy as np

try:
    from PIL import Image
except Exception as exc:
    print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
    raise


def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(src_dir, ext)))
    paths = sorted(paths)
    if not paths:
        print(f"No images found in {src_dir}", file=sys.stderr)
        return 1
    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img).transpose(2, 0, 1)  # HWC -> CHW
        name = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(dst_dir, name + ".npy"), arr)
    print(f"Converted {len(paths)} images -> {dst_dir}")
    return 0

def has_npy(dst_dir):
    return bool(glob.glob(os.path.join(dst_dir, "*.npy")))

def maybe_convert(src_dir, dst_dir, label):
    if has_npy(dst_dir):
        print(f"Skip {label}: npy already exists -> {dst_dir}")
        return 0
    if not src_dir or not os.path.isdir(src_dir):
        print(f"Source dir missing for {label}: {src_dir}", file=sys.stderr)
        return 1
    return convert(src_dir, dst_dir)

set12 = os.environ["SET12_SRC"]
cbsd68 = os.environ["CBSD68_SRC"]
urban_hr = os.environ["URBAN_HR"]
testsets = os.environ["TESTSETS_DIR"]

err = 0
err |= maybe_convert(set12, os.path.join(testsets, "Set12", "HQ"), "Set12")
err |= maybe_convert(cbsd68, os.path.join(testsets, "CBSD68", "HQ"), "CBSD68")
err |= maybe_convert(urban_hr, os.path.join(testsets, "urban100", "HQ"), "urban100")

if err:
    sys.exit(1)
PY

echo "Done. testsets are in: $TESTSETS_DIR"
