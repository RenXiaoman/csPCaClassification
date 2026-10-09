from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryCMSAGAuxTrainer import DIYBoundaryCMSAGAuxTrainer
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag import DIYBoundaryCMSAG


class DIYBoundaryCMSAGAuxTrainer_sym2(DIYBoundaryCMSAGAuxTrainer):
    wavelet = "sym2"
    boundary_radius = 1
            
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
                self.num_epochs = 80
                self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels: int,
                                   num_output_channels: int, enable_deep_supervision: bool = True) -> nn.Module:
        return DIYBoundaryCMSAG(
            in_channels=num_input_channels, out_channels=num_output_channels, spatial_dims=3,
            feature_size=16, norm_name="instance", dropout_rate=0.0, depth=4,
            boundary_channels=max(1, num_output_channels - 1), wavelet="sym2", mode="zero",
        )
