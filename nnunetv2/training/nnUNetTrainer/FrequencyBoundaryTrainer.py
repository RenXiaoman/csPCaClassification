from __future__ import annotations

from torch import nn

from nnunetv2.training.nnUNetTrainer.BoundaryTrainer import BoundaryTrainer
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.ablation import BoundaryOnlyNet


class FrequencyBoundaryTrainer(BoundaryTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return BoundaryOnlyNet(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=3,
            feature_size=16,
            norm_name="instance",
            dropout_rate=0.0,
            depth=4,
            boundary_channels=1,
            use_frequency=True,
            use_msag=False,
        )
