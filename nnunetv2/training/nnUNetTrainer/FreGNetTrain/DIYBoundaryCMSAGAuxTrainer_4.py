from __future__ import annotations
import torch



from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryCMSAGAuxTrainer import (
    DIYBoundaryCMSAGAuxTrainer,
)


class DIYBoundaryCMSAGAuxTrainer_4(DIYBoundaryCMSAGAuxTrainer):
    boundary_radius = 4
    def __init__(
                self,
                plans: dict,
                configuration: str,
                fold: int,
                dataset_json: dict,
                device: torch.device = torch.device("cuda"),
            ):
                super().__init__(plans, configuration, fold, dataset_json, device)
                self.enable_deep_supervision = False
                self.num_epochs = 140
                self.initial_lr = 1e-3
