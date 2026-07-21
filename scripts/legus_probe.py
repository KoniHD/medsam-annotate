#!/usr/bin/env python
# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""M6: a real box-prompt REST round trip against a running server -- shared by
``scripts/start_server.sh`` (pre-warm) and ``scripts/check_demo.sh`` (preflight).

This is a thin wrapper, not a reimplementation: it delegates the actual request/response
handling to ``monailabel.client.MONAILabelClient`` -- the same client the 3D Slicer MONAI Label
plugin uses -- rather than hand-building the ``POST /infer/<model>`` request and hand-parsing
MONAI Label's multipart response. `MONAILabelClient.infer(model, image_id, params)` already does
exactly that (request construction, multipart parsing, writing the returned mask to a file) and
hands back `(mask_file_path, params)`.

Measured: `from monailabel.client import MONAILabelClient` completes in ~0.1s and pulls in no
torch -- so there is no dependency-weight cost to using it over a bespoke httpx-based client, and
using it means any change to MONAI Label's wire format is absorbed by the upstream client instead
of needing to be re-diagnosed here.

Subcommands:
    pick-box     Print "<image_id> <r1> <c1> <r2> <c2>" for the first (preferably
                  unlabeled) image under a studies directory. Used by the shell
                  scripts to build a request without hardcoding a dataset-specific
                  box.
    round-trip   POST a box prompt to a running server and report elapsed time +
                  whether the returned mask is non-empty. Writes the mask to
                  --out-mask if given (so check_demo.sh can feed it to the
                  measurement export next).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from monailabel.client import MONAILabelClient


def _images_under(studies: Path) -> list[Path]:
    exts = (".nii.gz", ".nii")
    return sorted(p for p in studies.iterdir() if p.is_file() and p.name.endswith(exts))


def _image_id(path: Path) -> str:
    name = path.name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else path.stem


def pick_demo_image(studies: Path) -> tuple[str, Path]:
    """Return (image_id, path) for the first unlabeled image under `studies`, or the
    first image at all if every image already has a `labels/final` entry."""
    images = _images_under(studies)
    if not images:
        raise SystemExit(f"no images found directly under {studies}")

    final_dir = studies / "labels" / "final"
    labeled_ids = set()
    if final_dir.is_dir():
        labeled_ids = {_image_id(p) for p in _images_under(final_dir)}

    for image in images:
        if _image_id(image) not in labeled_ids:
            return _image_id(image), image
    # Every image already labeled: still return one so a round-trip check can run.
    return _image_id(images[0]), images[0]


def central_box(image_path: Path, margin: float = 0.2) -> list[int]:
    """A generous central [r1, c1, r2, c2] box sized off the image's own
    dimensions, so it is valid regardless of dataset resolution -- no dataset-
    specific coordinates hardcoded anywhere in the demo tooling."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.ReadImageInformation()
    w, h = reader.GetSize()[0], reader.GetSize()[1]
    r1, c1 = int(h * margin), int(w * margin)
    r2, c2 = int(h * (1 - margin)), int(w * (1 - margin))
    return [r1, c1, r2, c2]


def round_trip(
    base_url: str,
    model: str,
    image_id: str,
    box: list[int],
    out_mask: Path | None = None,
    timeout: float = 120.0,
) -> tuple[float, int]:
    """POST a box prompt via the upstream MONAI Label client, return
    (elapsed_seconds, nonzero_voxel_count) and optionally write the mask NIfTI to `out_mask`."""
    params = {"roi": box, "foreground": [], "background": []}
    with tempfile.TemporaryDirectory(prefix="legus_probe_") as tmpdir:
        client = MONAILabelClient(base_url, tmpdir=tmpdir)
        t0 = time.monotonic()
        # infer() raises MONAILabelClientException on a non-200 response, sets the request
        # timeout on the underlying http.client connection isn't configurable here -- MONAI
        # Label's own client doesn't expose one -- so this relies on the server responding, which
        # is the same assumption the 3D Slicer plugin makes.
        mask_file, _result_params = client.infer(model, image_id, params)
        elapsed = time.monotonic() - t0

        if mask_file is None:
            raise RuntimeError(
                f"no mask file in infer response for model={model!r} image={image_id!r}; "
                "the response format may have changed upstream"
            )

        mask = sitk.ReadImage(mask_file)
        nonzero = int(np.count_nonzero(sitk.GetArrayFromImage(mask)))

        if out_mask is not None:
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mask_file, out_mask)

    _ = timeout  # kept for CLI-compatibility; MONAILabelClient has no per-call timeout knob
    return elapsed, nonzero


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pick = sub.add_parser("pick-box", help="print '<image_id> <r1> <c1> <r2> <c2>' for a demo image")
    p_pick.add_argument("--studies", required=True, type=Path)
    p_pick.add_argument("--margin", type=float, default=0.2)

    p_rt = sub.add_parser("round-trip", help="POST a box prompt and report the result")
    p_rt.add_argument("--base-url", default="http://localhost:8000")
    p_rt.add_argument("--model", default="medsam2_2d")
    p_rt.add_argument("--studies", required=True, type=Path)
    p_rt.add_argument("--margin", type=float, default=0.2)
    p_rt.add_argument("--out-mask", type=Path, default=None)
    p_rt.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args(argv)

    if args.command == "pick-box":
        image_id, image_path = pick_demo_image(args.studies)
        box = central_box(image_path, args.margin)
        print(f"{image_id} {box[0]} {box[1]} {box[2]} {box[3]}")
        return 0

    if args.command == "round-trip":
        image_id, image_path = pick_demo_image(args.studies)
        box = central_box(image_path, args.margin)
        try:
            elapsed, nonzero = round_trip(
                args.base_url, args.model, image_id, box, args.out_mask, args.timeout
            )
        except Exception as exc:  # noqa: BLE001 -- surface any failure as a clean CLI error
            target = f"{args.base_url}/infer/{args.model}"
            print(f"ERROR round-trip against {target} failed: {exc}", file=sys.stderr)
            return 1
        print(f"image={image_id} box={box} elapsed={elapsed:.2f}s nonzero_voxels={nonzero}")
        return 0 if nonzero > 0 else 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
