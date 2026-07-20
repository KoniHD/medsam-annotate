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

design.md Sec 5 "DeepEdit note": MONAI Label ships DeepEdit/DeepGrow as its default interactive
models. This app does **not** register them -- MedSAM2 replaces that role. Our own
`medsam2_2d`/`medsam2_3d` tasks are tagged `InferType.DEEPGROW` only because that is the
interactive-task *type* MONAI Label's client UIs (Slicer/OHIF) key their "interactive
segmentation" affordance off of -- it is not a reference to MONAI Label's DeepGrow model. See
`external/MONAILabel/sample-apps/radiology/main.py` (`sam_2d`/`sam_3d`) for the precedent this
mirrors.
"""

import logging
import os

from lib.infers.medsam2 import MedSAM2InferTask
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.strategy import Strategy
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
        # Only the MedSAM2 adapter is registered -- no DeepEdit/DeepGrow (design.md Sec 5).
        return {
            "medsam2_2d": MedSAM2InferTask(model_dir=self.model_dir, type=InferType.DEEPGROW, dimension=2),
            "medsam2_3d": MedSAM2InferTask(model_dir=self.model_dir, type=InferType.DEEPGROW, dimension=3),
        }

    def init_strategies(self) -> dict[str, Strategy]:
        # A minimal, valid active-learning strategy (design.md Sec 12); sample-selection
        # sophistication is out of scope for M2.
        return {"random": Random()}
