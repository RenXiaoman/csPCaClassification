import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.custom_networks.as_unetr import CDSA_Net

class CDSATrainer(nnUNetTrainer):
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
        return CDSA_Net(
        in_channels=2,  # ADC and DWI modalities
        out_channels=2,  # Background and lesion
        img_size=(16, 256, 256),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=8,
        norm_name='instance'
    )
        
    def set_deep_supervision_enabled(self, enabled: bool):
        pass
