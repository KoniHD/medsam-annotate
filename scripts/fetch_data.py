#!/usr/bin/env python
# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""M3: fetch public demo ultrasound data and lay it out as a MONAI Label datastore.

Usage (uv-runnable):

    uv run scripts/fetch_data.py --limit 4
    uv run scripts/fetch_data.py --limit 4 --force   # reconvert even if outputs exist

design.md Sec 7 is emphatic about the property this script exists to protect: ultrasound
depth/zoom varies per image, so a mask without its pixel spacing is not a measurement, just a
pixel count. Every image this script writes carries real physical spacing in its NIfTI affine,
and the sidecar JSON says in plain words whether that spacing is *known* (came from the source)
or *assumed* (we had nothing and refused to fake a calibration) -- see `_spacing_mm_from_coordinates`
and `SpacingResult` below, and `tests/test_datastore.py` for the round-trip proof.

--- A note on the data source (read before changing the default) --------------------------------

The task brief for this script named "wanglab/RVENet-MedSAM2" as the source, described as the
MedSAM2 authors' own echo set stored as .npz. Both details turned out to be wrong, verified by
actually fetching and inspecting the repo rather than trusting the description:

  * It is not npz. The only asset is `Annotations-MedSAM2.zip` (541 MB, matching the ~0.54GB
    estimate), which unzips to 247,169 PNG files -- mask frames only, one subdirectory per
    video, files named 0000.png, 0001.png, ...
  * It has no source images at all, and cannot get any automatically. The dataset card says so
    explicitly: "This dataset contains all the masks. Please follow the guideline on RVENet to
    access the raw images." Fetching https://rvenet.github.io/dataset/ confirms the raw videos
    require individually registering, filling out a Google Form, and accepting a Research Use
    Agreement that explicitly restricts redistribution. That is neither automatable nor
    something an idempotent public-data fetch script should route around.

So RVENet-MedSAM2 cannot populate the *images* side of an image+mask datastore, at all, no
matter how this script is written. This module's own project README (written before this file
existed) already names the fallback: "Demo data is public only (RVENet-MedSAM2, CAMUS)."

The default here is CAMUS (2D transthoracic echocardiography, the same modality MedSAM2's own
echo checkpoint targets), via the `zeahub/camus-sample` HDF5 conversion on Hugging Face:
ungated, CC BY-NC-SA 4.0 (explicitly permits redistribution, unlike RVENet's terms), and -- unlike
plain PNG exports -- each frame carries genuine per-pixel physical coordinates in metres, i.e.
real calibration, not an assumed one. See `_download` for how `--dataset` can point elsewhere.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import SimpleITK as sitk
from huggingface_hub import snapshot_download
from monailabel.interfaces.datastore import DefaultLabelTag

logger = logging.getLogger("fetch_data")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "zeahub/camus-sample"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_DATASTORE_DIR = REPO_ROOT / "data" / "legus"

# `wanglab/RVENet-MedSAM2` is intentionally *not* the default -- see the module docstring. It is
# still recognised here so pointing at it fails loudly and explains why, instead of silently
# writing a broken datastore.
KNOWN_MASKS_ONLY_DATASETS = {"wanglab/RVENet-MedSAM2"}

CAMUS_CITATION = (
    "S. Leclerc et al., \"Deep Learning for Segmentation Using an Open Large-Scale Dataset in "
    "2D Echocardiography.\" IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, "
    "2019. https://doi.org/10.1109/TMI.2019.2900516"
)
CAMUS_LICENSE = "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)"


# --- spacing: the property design.md Sec 7 cares about -------------------------------------


@dataclasses.dataclass
class SpacingResult:
    """Physical pixel spacing in millimetres, plus an honest account of where it came from.

    design.md Sec 7: "Raw PNG loses physical scale -> measurements become non-comparable... so
    calibration must be captured per image." A fake 1.0mm spacing that *looks* real is worse
    than a recorded unknown, so `known` is not decoration -- `tests/test_datastore.py` and
    downstream measurement code (apps/legus/lib/measure/statistics.py's `SpacingInfo.calibrated`)
    both key off it.
    """

    values_mm: tuple[float, float]  # (x/column spacing, y/row spacing), sitk order
    known: bool
    source: str


def _spacing_mm_from_coordinates(coordinates: np.ndarray | None) -> SpacingResult:
    """Derive (dx, dy) pixel spacing in mm from a zea-format per-pixel coordinate grid.

    `coordinates` is `(H, W, 3)` float32 metres, one Cartesian [x, y, z] position per pixel
    (see the zeahub/camus-sample README). Spacing is the physical distance between adjacent
    pixels along each array axis -- computed as a vector norm so it does not assume which of
    the 3 coordinate components varies along which array axis (robust to other zea-format
    sources with a different scan-plane convention).

    Falls back to an explicitly-*unknown* unit spacing if the coordinate grid is missing,
    degenerate, or too small to difference -- this is the "record it, don't fake it" path
    design.md Sec 7 asks for.
    """
    no_grid = coordinates is None or coordinates.ndim != 3
    too_small = coordinates is not None and coordinates.ndim == 3 and min(coordinates.shape[:2]) < 2
    if no_grid or too_small:
        return SpacingResult(
            values_mm=(1.0, 1.0), known=False, source="no coordinate grid in source; unit spacing assumed"
        )

    row_step = coordinates[1, 0] - coordinates[0, 0]
    col_step = coordinates[0, 1] - coordinates[0, 0]
    row_spacing_mm = float(np.linalg.norm(row_step)) * 1000.0
    col_spacing_mm = float(np.linalg.norm(col_step)) * 1000.0

    degenerate = row_spacing_mm <= 0 or col_spacing_mm <= 0
    degenerate = degenerate or not np.isfinite([row_spacing_mm, col_spacing_mm]).all()
    if degenerate:
        return SpacingResult(
            values_mm=(1.0, 1.0),
            known=False,
            source="degenerate coordinate grid (zero/non-finite step); unit spacing assumed",
        )

    return SpacingResult(
        values_mm=(col_spacing_mm, row_spacing_mm),  # sitk SetSpacing order: (x, y) = (col, row)
        known=True,
        source="per-pixel Cartesian coordinates (metres) from source dataset, converted to mm",
    )


# --- display windowing (the property M3 review finding 1 cares about) ----------------------


def to_display_uint8(image_2d: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Map raw source intensities to a displayable uint8 grayscale frame (0-255).

    CAMUS zea-format frames are stored as log-compressed dB values (observed range roughly
    [-60.0, 0.0]), not [0, 255] or [0, 1]. Writing that raw float range straight into the NIfTI
    breaks upstream `Sam2InferTask.run2d` (external/MONAILabel/monailabel/sam2/infer.py):
    `img_as_ubyte(slice_np)` raises `ValueError: Images of type float must be between -1 and 1.`
    on the no-slice branch, and `Image.fromarray(a).convert("RGB")` on the slice branch silently
    produces an all-zero (black) frame instead -- verified against the shipped demo data. Either
    way MedSAM2 never sees real pixels.

    Windowed per-image (this frame's own min/max -> [0, 255]) rather than to a fixed assumed dB
    range, because that only requires the data actually in the file, not an unverified
    assumption about every source's dB convention. The applied window is returned so callers can
    record it in the sidecar for auditability -- same spirit as `SpacingResult.source` for
    calibration: a transform this consequential must never be silent.
    """
    source = image_2d.astype(np.float64)
    lo = float(source.min())
    hi = float(source.max())
    if hi <= lo:
        # Degenerate (uniform) frame -- do not divide by zero; map every pixel to mid-gray.
        display = np.full(source.shape, 128, dtype=np.uint8)
    else:
        display = np.clip((source - lo) / (hi - lo) * 255.0, 0, 255)
        display = np.round(display).astype(np.uint8)

    window = {
        "method": "per_image_min_max_to_uint8",
        "source_min": lo,
        "source_max": hi,
        "mapped_range": [0, 255],
    }
    return display, window


# --- NIfTI + sidecar writers (the MONAI Label datastore convention) ------------------------


def write_datastore_pair(
    *,
    datastore_dir: Path,
    image_id: str,
    image_2d: np.ndarray,
    label_2d: np.ndarray,
    spacing: SpacingResult,
    sidecar: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Write one image + label pair into the MONAI Label `LocalDatastore` layout.

    Confirmed against `external/MONAILabel/monailabel/datastore/local.py`
    (`LocalDatastore.__init__`, default `images_dir="."`, `labels_dir="labels"`, and
    `DefaultLabelTag.ORIGINAL == "original"`): images live directly under the datastore root as
    `<id>.nii.gz`, and the CAMUS ground-truth label -- pre-existing, not annotator-produced --
    lives under `labels/original/<id>.nii.gz` (see `DefaultLabelTag`). Writing it under
    `labels/final/` would make `LocalDatastore.get_unlabeled_images()` return nothing (every
    image already "done") and `datalist()` (which trains on `DefaultLabelTag.FINAL`) would train
    on public ground truth instead of the annotator's own corrections -- verified against a real
    `LocalDatastore` instance. The default `extensions=("*.nii.gz", "*.nii")` filter also means
    the `.json` sidecar written next to each image is invisible to the datastore's own file scan
    -- no naming collision.

    `image_2d` must already be uint8 in [0, 255] (see `to_display_uint8`) -- this function does
    not window intensities, only lays out files. Both image and label are written as a 3-D
    volume with a singleton z-axis (array shape `(1, H, W)`, sitk size `(W, H, 1)`, 3-element
    spacing `(dx, dy, 1.0)`), never a genuinely 2-D NIfTI: `Sam2InferTask.run2d` indexes
    `image_tensor[:, :, slice_idx]`, which raises `IndexError: too many indices for tensor of
    dimension 2` on a true 2-D volume, and its slice-less fallback path can't produce a 3-channel
    RGB frame from a bare (H, W) array either. Verified end to end: with this layout
    `MedSAM2InferTask.segment(...)` returns a real mask (tens of thousands of nonzero pixels)
    instead of raising.
    """
    images_dir = datastore_dir
    labels_dir = datastore_dir / "labels" / DefaultLabelTag.ORIGINAL.value
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{image_id}.nii.gz"
    label_path = labels_dir / f"{image_id}.nii.gz"
    sidecar_path = images_dir / f"{image_id}.json"

    spacing_3d = (spacing.values_mm[0], spacing.values_mm[1], 1.0)

    # (1, H, W) into GetImageFromArray -> sitk size (W, H, 1): a real 3-D volume with a
    # singleton z-axis, not a 2-D image, so it round-trips through MONAI's `LoadImaged` and
    # upstream's slice-indexed `run2d` (see docstring above).
    image_uint8 = np.asarray(image_2d).astype(np.uint8)
    image = sitk.GetImageFromArray(image_uint8[np.newaxis, :, :])
    image.SetSpacing(spacing_3d)
    sitk.WriteImage(image, str(image_path), useCompression=True)

    label_uint8 = np.asarray(label_2d).astype(np.uint8)
    label = sitk.GetImageFromArray(label_uint8[np.newaxis, :, :])
    label.SetSpacing(spacing_3d)
    sitk.WriteImage(label, str(label_path), useCompression=True)

    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return image_path, label_path, sidecar_path


# --- Hugging Face fetch ----------------------------------------------------------------------


def _download(dataset: str, raw_dir: Path) -> Path:
    """Idempotently cache `dataset` under `raw_dir`.

    Wraps `huggingface_hub.snapshot_download`, which is itself content-addressed and skips
    files already present in `cache_dir` -- re-running this is a no-op network-wise once
    everything is cached, satisfying the "idempotent, no re-download" requirement without this
    script reimplementing any caching logic of its own.
    """
    if dataset in KNOWN_MASKS_ONLY_DATASETS:
        raise SystemExit(
            f"{dataset!r} contains MedSAM2-generated *masks only* (verified: 247,169 PNG mask "
            "frames, zero source images). The paired raw images require individually "
            "registering with RVENet and accepting a non-redistributable Research Use "
            "Agreement (see https://rvenet.github.io/dataset/) -- not something this script "
            "can or should automate. Use the default --dataset (zeahub/camus-sample) or another "
            "source that actually ships images. See the module docstring for the full story."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = raw_dir / ".hf_cache"
    logger.info(f"Fetching {dataset!r} (cache: {cache_dir})...")
    snapshot_path = snapshot_download(repo_id=dataset, repo_type="dataset", cache_dir=str(cache_dir))
    return Path(snapshot_path)


# --- CAMUS (zea HDF5) -> datastore conversion -------------------------------------------------


def _iter_camus_samples(snapshot_dir: Path):
    """Yield one dict per annotated frame across every `*.hdf5` file under `snapshot_dir`.

    Sorted by file path (deterministic across runs, so `--limit N` always selects the same
    first N samples). Only frames with a non-empty `metadata/annotations/label` (i.e. "ED" or
    "ES" -- the two phases CAMUS actually annotated within each half-cycle sequence) are
    yielded; the rest of the cine sequence has no ground-truth mask (README: "unannotated
    frames have only the first channel set").
    """
    hdf5_files = sorted(snapshot_dir.rglob("*.hdf5"))
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            if "metadata/subject/id" in f:
                patient_id = f["metadata/subject/id"][()].decode()
            else:
                patient_id = hdf5_path.stem
            phase_labels = [v.decode() for v in f["metadata/annotations/label"][:]]
            views = [v.decode() for v in f["metadata/annotations/view"][:]]
            image_quality = f["metadata/annotations/image_quality"][()].decode()
            n_frames = len(phase_labels)

            for track_name in sorted(f["tracks"].keys()):
                track = f["tracks"][track_name]
                seg_labels = [v.decode() for v in track["data/segmentation/labels"][:]]
                images = track["data/image/values"]
                coords = track["data/image/coordinates"][:]
                seg_values = track["data/segmentation/values"]

                for idx, phase in enumerate(phase_labels):
                    if not phase:  # unannotated frame in the cine sequence
                        continue
                    image_2d = images[idx]
                    seg_onehot = seg_values[idx]  # (H, W, n_labels) bool
                    label_2d = np.argmax(seg_onehot, axis=-1)  # 0 = background/unannotated

                    yield {
                        "hdf5_path": hdf5_path,
                        "patient_id": patient_id,
                        "view": views[idx],
                        "phase": phase,
                        "image_quality": image_quality,
                        "frame_count": n_frames,
                        "frame_index": idx,
                        "seg_labels": seg_labels,
                        "image_2d": image_2d,
                        "label_2d": label_2d,
                        "coordinates": coords,
                    }


def convert_camus(
    dataset: str, snapshot_dir: Path, datastore_dir: Path, limit: int | None, force: bool
) -> int:
    written = 0
    for sample in _iter_camus_samples(snapshot_dir):
        if limit is not None and written >= limit:
            break

        image_id = f"camus_{sample['patient_id']}_{sample['view']}_{sample['phase']}"
        image_path = datastore_dir / f"{image_id}.nii.gz"
        label_path = datastore_dir / "labels" / DefaultLabelTag.ORIGINAL.value / f"{image_id}.nii.gz"
        sidecar_path = datastore_dir / f"{image_id}.json"
        if not force and image_path.exists() and label_path.exists() and sidecar_path.exists():
            logger.info(f"skip (exists): {image_id}")
            written += 1
            continue

        spacing = _spacing_mm_from_coordinates(sample["coordinates"])
        display_image_2d, window = to_display_uint8(sample["image_2d"])
        sidecar = {
            "image_id": image_id,
            "source_dataset": dataset,
            "source_file": str(sample["hdf5_path"].relative_to(snapshot_dir)),
            "modality": "ultrasound (2D transthoracic echocardiography)",
            "patient_id": sample["patient_id"],
            "view": sample["view"],
            "cardiac_phase": sample["phase"],
            "image_quality": sample["image_quality"],
            "frame_count": sample["frame_count"],
            "extracted_frame_index": sample["frame_index"],
            "spacing_mm_xy": list(spacing.values_mm),
            "spacing_known": spacing.known,
            "spacing_source": spacing.source,
            "intensity_window": window,
            "segmentation_labels": sample["seg_labels"],
            "label_tag": DefaultLabelTag.ORIGINAL.value,
            "label_note": (
                "CAMUS ground-truth mask, written under the 'original' tag (not 'final') so the "
                "image reads as unlabeled to MONAI Label's active-learning sample selection and "
                "the annotator's own corrections -- not this pre-existing GT -- populate "
                "'final'/training data. Kept on disk to compute per-round Dice against a "
                "held-out split (design.md Sec 6 step 5)."
            ),
            "license": CAMUS_LICENSE,
            "citation": CAMUS_CITATION,
            "prompts": [],  # filled in by the app as the annotator box/point-prompts this image
        }

        write_datastore_pair(
            datastore_dir=datastore_dir,
            image_id=image_id,
            image_2d=display_image_2d,
            label_2d=sample["label_2d"],
            spacing=spacing,
            sidecar=sidecar,
        )
        logger.info(f"wrote {image_id} (spacing={spacing.values_mm} mm, known={spacing.known})")
        written += 1

    return written


# --- CLI ---------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Where the raw HF download is cached (idempotent)",
    )
    parser.add_argument(
        "--datastore-dir",
        type=Path,
        default=DEFAULT_DATASTORE_DIR,
        help="MONAI Label datastore output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of image+label pairs to convert (keeps the demo set small)",
    )
    parser.add_argument("--force", action="store_true", help="Reconvert even if outputs already exist")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("fetch_data").setLevel(logging.INFO)

    snapshot_dir = _download(args.dataset, args.raw_dir)

    if args.dataset == "zeahub/camus-sample" or args.dataset.startswith("zeahub/camus"):
        written = convert_camus(args.dataset, snapshot_dir, args.datastore_dir, args.limit, args.force)
    else:
        raise SystemExit(
            f"No converter registered for --dataset {args.dataset!r}. Only the CAMUS zea-HDF5 "
            "layout (zeahub/camus*) is currently supported -- add a converter analogous to "
            "convert_camus() for a new source, following the same spacing-honesty contract."
        )

    print(f"Wrote {written} image+label pair(s) to {args.datastore_dir}")
    print(f"  images:      {args.datastore_dir}/<id>.nii.gz")
    print(
        f"  labels:      {args.datastore_dir}/labels/{DefaultLabelTag.ORIGINAL.value}/<id>.nii.gz"
        "  (CAMUS ground truth, not 'final')"
    )
    print(f"  sidecar:     {args.datastore_dir}/<id>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
