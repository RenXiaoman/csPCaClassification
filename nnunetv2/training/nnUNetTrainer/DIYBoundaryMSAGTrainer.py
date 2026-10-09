from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.jit import TracerWarning

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag import DIYBoundaryMSAG

warnings.filterwarnings("ignore", category=TracerWarning)


class DIYBoundaryMSAGTrainer(nnUNetTrainer):
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
        self.num_epochs = 250
        self.initial_lr = 1e-2

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return DIYBoundaryMSAG(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=3,
            feature_size=16,
            norm_name="instance",
            dropout_rate=0.0,
            depth=4,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _do_i_compile(self):
        return False
