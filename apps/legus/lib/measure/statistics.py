"""Per-structure measurement export: mask -> physically-calibrated CSV rows.

design.md §7: "the mask is an intermediate, [...] the measurement table is the
deliverable." This module is a thin wrapper over pyradiomics'
``RadiomicsFeatureExtractor`` that turns an (image, mask) pair into one row per
subject x structure, carrying real physical units derived from the image spacing.

We do not hand-roll any feature maths pyradiomics already computes (shape
descriptors, first-order intensity statistics, texture). The only work done here is:
picking which pyradiomics feature classes to enable (core geometry/intensity vs. the
optional full radiomics set), renaming the headline geometry features to
explicit-unit column names (``area_mm2`` / ``volume_mm3``), and flagging, per row,
whether the spacing behind those units is real calibration or an unverified default
-- see ``SpacingInfo``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

logger = logging.getLogger(__name__)

# File suffixes that reliably carry physical pixel/voxel spacing in their header. This
# is an allowlist, not a denylist: any suffix not in here (including an unrecognised
# extension or a DICOM series directory, whose suffix is "") is treated as uncalibrated
# rather than trusted by default -- see load_calibrated_image.
_CALIBRATED_SUFFIXES = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm"}
# Formats known to discard physical scale (design.md §7: "Raw PNG loses physical scale").
# Used only to pick a more specific warning message; the calibrated/uncalibrated
# decision itself is driven by the _CALIBRATED_SUFFIXES allowlist above.
_UNCALIBRATED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Core feature classes always computed: geometry (shape) + first-order intensity
# stats, which is where echo intensity (mean grayscale) lives -- design.md §7 flags
# it as a validated muscle-quality biomarker. Texture classes are the "optional full
# radiomics feature set" the design doc calls out separately.
_CORE_2D_FEATURE_CLASSES = ("shape2D", "firstorder")
_CORE_3D_FEATURE_CLASSES = ("shape", "firstorder")
_TEXTURE_FEATURE_CLASSES = ("glcm", "gldm", "glrlm", "glszm", "ngtdm")

# Rename pyradiomics' native feature names to columns whose units are explicit in the
# name itself, per design.md §7 ("an area_mm2 column, not a bare 'area'").
_RENAME = {
    "shape2D_PixelSurface": "area_mm2",
    "shape_VoxelVolume": "volume_mm3",
    "firstorder_Mean": "echo_intensity_mean",
}

# shape2D_/shape_ feature names that are pure ratios (dimensionless): they compare two
# lengths/areas/volumes of the *same* structure, so an unknown or wrong pixel spacing
# cancels out and does not corrupt them. Every other shape2D_/shape_ feature pyradiomics
# emits is a raw length/area/volume in mm/mm2/mm3 (see radiomics/shape2D.py,
# radiomics/shape.py docstrings) and must be nulled when spacing isn't real calibration.
_DIMENSIONLESS_SHAPE_SUFFIXES = {
    "Elongation",
    "Flatness",
    "Sphericity",
    "SphericalDisproportion",
    "Compactness1",
    "Compactness2",
}
_SHAPE_CLASS_PREFIXES = ("shape2D_", "shape_")
# firstorder_TotalEnergy is explicitly documented by pyradiomics as "Energy feature
# scaled by the volume of the voxel in cubic mm" -- also spacing-dependent even though
# it's a firstorder, not shape, feature.
_EXTRA_SPACING_DEPENDENT_FEATURES = {"firstorder_TotalEnergy"}


def _is_spacing_dependent(original_name: str) -> bool:
    """True if a pyradiomics feature (native name, pre-rename) carries mm/mm2/mm3 units."""
    for prefix in _SHAPE_CLASS_PREFIXES:
        if original_name.startswith(prefix):
            return original_name[len(prefix) :] not in _DIMENSIONLESS_SHAPE_SUFFIXES
    return original_name in _EXTRA_SPACING_DEPENDENT_FEATURES


@dataclass(frozen=True)
class SpacingInfo:
    """Where the spacing behind a row's physical-unit columns actually came from."""

    values: tuple[float, ...]
    # "image_header" | "cli_override" | "assumed_unit" | "unknown_format" |
    # "image_header_unit_default"
    source: str
    calibrated: bool


def _suffix(path: Path) -> str:
    return "".join(path.suffixes[-2:]) if path.suffixes[-1:] == [".gz"] else path.suffix


def _resolve_spacing_override(
    image: sitk.Image, path: Path, spacing_override: tuple[float, ...]
) -> tuple[float, ...]:
    """Match a ``--spacing-mm`` override to ``image``'s actual dimensionality.

    ``sitk.Image.SetSpacing`` requires exactly one value per axis and raises an opaque
    C++ exception otherwise. A 2D-shaped override is also accepted for a 3D image with
    a singleton trailing axis (e.g. NIfTI stored as (x, y, 1)) since ``_squeeze_z``
    drops that axis before any measurement runs, so its spacing is inert -- padded with
    the header's own value (or 1.0) rather than guessed.
    """
    dim = image.GetDimension()
    values = tuple(float(s) for s in spacing_override)
    if len(values) == dim:
        return values
    if len(values) == dim - 1 and image.GetSize()[dim - 1] == 1:
        pad = image.GetSpacing()[dim - 1] or 1.0
        return values + (pad,)
    raise ValueError(
        f"--spacing-mm got {len(values)} value(s) ({values}) but {path} is "
        f"{dim}-D with size {image.GetSize()}; expected {dim} values"
        + (f", or {dim - 1} for a 3D image with a singleton trailing axis" if dim == 3 else "")
        + "."
    )


def load_calibrated_image(
    path: str | Path, spacing_override: tuple[float, ...] | None = None
) -> tuple[sitk.Image, SpacingInfo]:
    """Read an image file and decide whether its spacing is real calibration.

    NIfTI/MHA/NRRD/DICOM headers *can* carry physical spacing; PNG/JPEG/BMP/TIFF never
    do (design.md §7: "Raw PNG loses physical scale"), and any other/unrecognised
    format (including a suffix-less DICOM series directory) is treated the same way:
    calibration is an allowlist (``_CALIBRATED_SUFFIXES``), not a denylist, so an
    unfamiliar format is uncalibrated by default rather than silently trusted.

    Even for an allowlisted format, a header that reports exactly unit spacing
    (1.0 on every axis) is treated as uncalibrated: that is SimpleITK's default when no
    spacing tag is present at all, which is indistinguishable from a genuine 1mm
    isotropic image -- and is exactly the failure mode of an ultrasound DICOM whose
    real scale lives in ``SequenceOfUltrasoundRegions`` rather than ``PixelSpacing``
    (design.md §11). An explicit ``spacing_override`` is the one path exempted from
    this check: it is the caller's deliberate assertion, not an inferred header value.

    Spacing must be supplied explicitly via ``spacing_override`` or the image is
    returned flagged as uncalibrated so callers don't silently mint fake mm values.
    """
    path = Path(path)
    image = sitk.ReadImage(str(path))
    suffix = _suffix(path)

    if spacing_override is not None:
        image.SetSpacing(_resolve_spacing_override(image, path, spacing_override))
        return image, SpacingInfo(image.GetSpacing(), "cli_override", True)

    if suffix not in _CALIBRATED_SUFFIXES:
        if suffix in _UNCALIBRATED_SUFFIXES:
            reason, source = "has no physical spacing metadata", "assumed_unit"
        else:
            reason, source = "is not a recognised calibrated format", "unknown_format"
        logger.warning(
            "%s %s (format=%r); measurements will be flagged as uncalibrated unless "
            "--spacing-mm is given.",
            path,
            reason,
            suffix,
        )
        return image, SpacingInfo(image.GetSpacing(), source, False)

    spacing = image.GetSpacing()
    if all(math.isclose(s, 1.0) for s in spacing):
        logger.warning(
            "%s reports exactly unit spacing %s -- SimpleITK's default when no "
            "calibration tag is present in the header (e.g. an ultrasound DICOM whose "
            "scale lives in SequenceOfUltrasoundRegions, not PixelSpacing). Treating as "
            "uncalibrated; pass --spacing-mm explicitly if %s mm is genuinely correct.",
            path,
            spacing,
            spacing,
        )
        return image, SpacingInfo(spacing, "image_header_unit_default", False)

    return image, SpacingInfo(spacing, "image_header", True)


def _is_2d(image: sitk.Image) -> bool:
    return image.GetDimension() == 2 or (image.GetDimension() == 3 and image.GetSize()[2] == 1)


def _squeeze_z(image: sitk.Image) -> sitk.Image:
    """Drop a singleton z axis from a 3D image, keeping in-plane (x, y) spacing."""
    if image.GetDimension() == 2:
        return image
    arr = sitk.GetArrayFromImage(image)[0]  # (z, y, x) -> (y, x)
    squeezed = sitk.GetImageFromArray(arr)
    squeezed.SetSpacing(image.GetSpacing()[:2])
    return squeezed


def _build_extractor(is_2d: bool, full_radiomics: bool) -> featureextractor.RadiomicsFeatureExtractor:
    settings = {"force2D": True, "force2Ddimension": 0} if is_2d else {}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in _CORE_2D_FEATURE_CLASSES if is_2d else _CORE_3D_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)
    if full_radiomics:
        for feature_class in _TEXTURE_FEATURE_CLASSES:
            extractor.enableFeatureClassByName(feature_class)
    return extractor


def _clean_feature_name(key: str) -> str:
    # pyradiomics keys look like "original_shape2D_PixelSurface".
    name = key.removeprefix("original_")
    return _RENAME.get(name, name)


def compute_structure_measurements(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    subject_id: str,
    spacing: SpacingInfo,
    structure_names: dict[int, str] | None = None,
    full_radiomics: bool = False,
) -> pd.DataFrame:
    """Compute one measurement row per non-zero label in ``mask``.

    ``image``/``mask`` must already carry the spacing to use (set via
    ``sitk.Image.SetSpacing``); ``spacing`` documents where that spacing came from so
    the output can flag uncalibrated rows instead of quietly presenting pixel counts
    as millimetres (design.md §7).
    """
    mask_arr = sitk.GetArrayFromImage(mask)
    labels = sorted(int(v) for v in np.unique(mask_arr) if v != 0)
    if not labels:
        raise ValueError("mask contains no non-zero (structure) labels")

    two_d = _is_2d(image)
    calc_image, calc_mask = (_squeeze_z(image), _squeeze_z(mask)) if two_d else (image, mask)

    rows = []
    for label in labels:
        extractor = _build_extractor(two_d, full_radiomics)
        result = extractor.execute(calc_image, calc_mask, label=label)

        row: dict[str, object] = {
            "subject_id": subject_id,
            "structure": (structure_names or {}).get(label, f"label_{label}"),
            "label": label,
            "dimensionality": "2D" if two_d else "3D",
            "spacing_mm": spacing.values,
            "spacing_source": spacing.source,
            "spacing_calibrated": spacing.calibrated,
        }

        # Raw voxel/pixel count is spacing-independent and always trustworthy; keep
        # it alongside the physical-unit columns so an uncalibrated row stays legible.
        row["pixel_count" if two_d else "voxel_count"] = int(np.count_nonzero(mask_arr == label))

        spacing_dependent_cols: list[str] = []
        for key, value in result.items():
            if key.startswith("diagnostics_"):
                continue
            original_name = key.removeprefix("original_")
            column = _clean_feature_name(key)
            row[column] = float(value) if np.isscalar(value) else value
            if _is_spacing_dependent(original_name):
                spacing_dependent_cols.append(column)

        if not spacing.calibrated:
            # An assumed/default unit spacing must never masquerade as a real
            # millimetre measurement. Null *every* spacing-dependent column pyradiomics
            # emitted for this row (area_mm2/volume_mm3 plus every other mm/mm2/mm3
            # shape feature -- e.g. shape2D_Perimeter, shape_MeshVolume,
            # shape_Maximum3DDiameter -- and firstorder_TotalEnergy), not just the two
            # renamed headline columns. Keep the raw pixel/voxel count and the
            # dimensionless shape ratios (Sphericity, Elongation, Flatness, ...).
            for col in spacing_dependent_cols:
                row[col] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def measure_subject(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    subject_id: str,
    spacing_mm: tuple[float, ...] | None = None,
    structure_names: dict[int, str] | None = None,
    full_radiomics: bool = False,
) -> pd.DataFrame:
    """Load an image + mask from disk and return their measurement rows.

    The image's spacing (header or ``spacing_mm`` override) is authoritative and is
    also applied to the mask, matching the datastore convention that mask and image
    share a grid (design.md §7).
    """
    image, spacing = load_calibrated_image(image_path, spacing_mm)
    mask, _ = load_calibrated_image(mask_path, spacing_mm)
    mask.SetSpacing(image.GetSpacing())
    return compute_structure_measurements(
        image,
        mask,
        subject_id=subject_id,
        spacing=spacing,
        structure_names=structure_names,
        full_radiomics=full_radiomics,
    )


def export_csv(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Export per-structure measurements (real physical units) from an image + mask to CSV."
    )
    parser.add_argument("--image", required=True, help="Path to the source image (NIfTI/MHA/DICOM/...).")
    parser.add_argument("--mask", required=True, help="Path to the label mask, same grid as --image.")
    parser.add_argument("--subject-id", required=True, help="Subject/study identifier for the output rows.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--spacing-mm",
        type=float,
        nargs="+",
        default=None,
        help="Explicit pixel/voxel spacing in mm, overriding header spacing, e.g. 0.2 0.2.",
    )
    parser.add_argument(
        "--structure-names",
        type=str,
        default=None,
        help='JSON mapping of mask label -> structure name, e.g. \'{"1": "muscle", "2": "fat"}\'.',
    )
    parser.add_argument(
        "--full-radiomics",
        action="store_true",
        help="Also compute the full pyradiomics texture feature set (design.md §7: optional).",
    )
    args = parser.parse_args(argv)

    structure_names = None
    if args.structure_names:
        structure_names = {int(k): v for k, v in json.loads(args.structure_names).items()}

    spacing_mm = tuple(args.spacing_mm) if args.spacing_mm else None
    try:
        df = measure_subject(
            args.image,
            args.mask,
            subject_id=args.subject_id,
            spacing_mm=spacing_mm,
            structure_names=structure_names,
            full_radiomics=args.full_radiomics,
        )
    except ValueError as exc:
        # Turn a shape/arity mismatch (e.g. --spacing-mm given the wrong number of
        # values for --image's dimensionality) into a clean CLI error instead of a raw
        # SimpleITK traceback -- see _resolve_spacing_override.
        parser.error(str(exc))
    export_csv(df, args.out)
    logger.info("Wrote %d row(s) to %s", len(df), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
