import torch
from torch import nn
from monai.networks.nets import UNet

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class UNetDetectionTrainer(nnUNetTrainer):
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
        self.num_epochs = 200

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
            channels=[32, 64, 128, 256, 512, 1024],
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
