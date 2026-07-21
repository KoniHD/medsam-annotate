# Copyright (c) 2026 MedSAM annotation tool contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""MONAI Label app for the LEGUS pediatric lower-leg ultrasound annotation tool.

design.md Sec 5: the served model sits behind a thin, replaceable adapter -- this file is
deliberately ignorant of MedSAM2. It only knows the adapter class name
(`lib.infers.medsam2.MedSAM2InferTask`) and registers two instances of it (2D, 3D). Every
MedSAM2-specific detail (checkpoint path, hydra config, device selection/fallback) lives in
`lib/infers/medsam2.py`. Swapping in a MedSAM3 adapter later means changing the import + the
two `init_infers` lines below, nothing else in this file.

Two complementary serving modes are registered, both backed by the same MedSAM2 weights:

  * `medsam2_2d` / `medsam2_3d` (`InferType.DEEPGROW`) -- **interactive**: the annotator supplies a
    box/point prompt and gets a mask. This is SAM2's native mode and where day-one quality is good.
    Slicer/OHIF surface it under their interactive ("SmartEdit") affordance, which keys off the
    DEEPGROW type -- it is NOT a reference to MONAI Label's own DeepGrow model (design.md Sec 5's
    "DeepEdit note": we do not register DeepEdit/DeepGrow).
  * `medsam2` (`InferType.SEGMENTATION`) -- **automatic pre-labelling** (design.md Sec 6's
    "unprompted pre-labels"): no human prompt, one mask per image via a derived default box. SAM2
    cannot segment with no prompt at all, so "automatic" means a *synthetic* prompt; quality is
    rough on out-of-distribution data day one and climbs as the fine-tune loop teaches the target
    structure (design.md Sec 10 item 2 -- do not oversell day-one auto quality). Registered FIRST
    and advertising the same labels so the client's auto-run-on-next-sample resolves to this
    (valid segmentation) model rather than an interactive one.

design.md Sec 5: the served model sits behind a thin, replaceable adapter, so swapping in a MedSAM3
adapter later means changing the import + the `init_infers` lines below, nothing else in this file.
See `external/MONAILabel/sample-apps/radiology/main.py` (`sam_2d`/`sam_3d`) for the interactive
precedent this mirrors.
"""

import logging
import os

from lib.infers.medsam2 import MedSAM2AutoInferTask, MedSAM2InferTask
from lib.trainers.medsam2 import MedSAM2TrainTask
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.random import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="LEGUS -- Pediatric Lower-Leg Ultrasound Annotation",
            description="MedSAM2-assisted interactive segmentation (design.md)",
            version="0.1.0",
        )

    def init_infers(self) -> dict[str, InferTask]:
        # No DeepEdit/DeepGrow (design.md Sec 5). `medsam2` (auto/segmentation) is registered FIRST
        # so the Slicer client's auto-run-on-next-sample resolves the shared labels to it -- a valid
        # segmentation model -- instead of an interactive one (which is not in the auto-seg selector
        # and would POST /infer/ with an empty model name -> 404). Order matters here.
        return {
            "medsam2": MedSAM2AutoInferTask(model_dir=self.model_dir, dimension=2),
            "medsam2_2d": MedSAM2InferTask(model_dir=self.model_dir, type=InferType.DEEPGROW, dimension=2),
            "medsam2_3d": MedSAM2InferTask(model_dir=self.model_dir, type=InferType.DEEPGROW, dimension=3),
        }

    def init_strategies(self) -> dict[str, Strategy]:
        # A minimal, valid active-learning strategy (design.md Sec 12); sample-selection
        # sophistication is out of scope for M2.
        return {"random": Random()}

    def init_trainers(self) -> dict[str, TrainTask]:
        # design.md Sec 6 step 3 (fine-tune offline between rounds). Same replaceable-adapter
        # principle as init_infers: main.py only knows the class name -- every MedSAM2-specific
        # fact (warm-start checkpoint, FINAL-label datastore convention, device policy) lives in
        # lib/trainers/medsam2.py. This registers /train/medsam2 over REST.
        return {"medsam2": MedSAM2TrainTask(model_dir=self.model_dir)}
