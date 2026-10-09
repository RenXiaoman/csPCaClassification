from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.jit import TracerWarning

from nnunetv2.training.nnUNetTrainer.PairwiseAuxAblationTrainer import (
    PairwiseAuxAblationTrainer,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.FreGNetTrain.networks import (
    FreGNetAblationBackbone,
    FreGNetAuxAblationNet,
)

warnings.filterwarnings("ignore", category=TracerWarning)


class FreGNetAblationBaseTrainer(nnUNetTrainer):
    use_frequency = False
    use_cmsag = False

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
        self.num_epochs = 150
        self.initial_lr = 1e-3

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return FreGNetAblationBackbone(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=3,
            feature_size=16,
            norm_name="instance",
            dropout_rate=0.0,
            depth=4,
            use_frequency=cls.use_frequency,
            use_cmsag=cls.use_cmsag,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _do_i_compile(self):
        return False


class FreGNetAuxAblationBaseTrainer(PairwiseAuxAblationTrainer):
    use_frequency = False
    use_cmsag = False

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return FreGNetAuxAblationNet(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=3,
            feature_size=16,
            norm_name="instance",
            dropout_rate=0.0,
            depth=4,
            boundary_channels=max(1, num_output_channels - 1),
            use_frequency=cls.use_frequency,
            use_cmsag=cls.use_cmsag,
        )
