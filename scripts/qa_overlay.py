#!/usr/bin/env python3
# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Render an image + mask as a QA overlay PNG (design.md Sec 7: "QA overlay PNGs").

A viewer-independent way to see exactly what a mask contains -- useful to confirm the model/server
produced a real mask when a client (3D Slicer) shows nothing, isolating a display problem from a
segmentation problem. Thin wrapper over PIL + numpy + SimpleITK, no new dependencies.

    uv run scripts/qa_overlay.py --image data/legus/<id>.nii.gz \
        --mask  data/legus/labels/original/<id>.nii.gz --out overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def _slice2d(path: str) -> np.ndarray:
    """Read a NIfTI/MHA and return one 2-D array (slice 0 of a singleton-z volume, or the plane)."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if arr.ndim == 3:
        # scripts/fetch_data.py writes (z=1, H, W); pick the first (usually only) slice.
        arr = arr[0]
    return np.asarray(arr)


def _to_uint8(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        return np.zeros(g.shape, dtype=np.uint8)
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def overlay(image_path: str, mask_path: str, out_path: str, alpha: float = 0.45) -> dict:
    """Write a red-mask-on-grayscale PNG; return a small dict of stats for the caller to print."""
    gray = _to_uint8(_slice2d(image_path))
    mask = _slice2d(mask_path) > 0
    if mask.shape != gray.shape:
        raise ValueError(f"image {gray.shape} and mask {mask.shape} are not on the same grid")

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float64)
    red = np.array([255.0, 40.0, 40.0])
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * red
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)

    return {
        "shape": gray.shape,
        "mask_pixels": int(mask.sum()),
        "mask_fraction": round(float(mask.mean()), 4),
        "out": out_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Render an image+mask QA overlay PNG.")
    ap.add_argument("--image", required=True, help="Source image (NIfTI/MHA).")
    ap.add_argument("--mask", required=True, help="Mask on the same grid as --image.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--alpha", type=float, default=0.45, help="Mask opacity 0..1 (default 0.45).")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats = overlay(args.image, args.mask, args.out, alpha=args.alpha)
    print(
        f"Wrote {stats['out']}  (frame {stats['shape']}, "
        f"mask {stats['mask_pixels']} px = {stats['mask_fraction'] * 100:.1f}% of frame)"
    )
    if stats["mask_pixels"] == 0:
        print("WARNING: the mask is empty -- nothing to see. The segmentation, not the display, is at fault.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
