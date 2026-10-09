from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.base_trainers import (
    FreGNetAblationBaseTrainer,
    FreGNetAuxAblationBaseTrainer,
)


class FreGNetBaselineTrainer(FreGNetAblationBaseTrainer):
    pass


class FreGNetWFETrainer(FreGNetAblationBaseTrainer):
    use_frequency = True


class FreGNetCMSAGTrainer(FreGNetAblationBaseTrainer):
    use_cmsag = True


class FreGNetAuxTrainer(FreGNetAuxAblationBaseTrainer):
    pass


class FreGNetWFEAuxTrainer(FreGNetAuxAblationBaseTrainer):
    use_frequency = True


class FreGNetCMSAGAuxTrainer(FreGNetAuxAblationBaseTrainer):
    use_cmsag = True


class FreGNetWFECMSAGTrainer(FreGNetAblationBaseTrainer):
    use_frequency = True
    use_cmsag = True
