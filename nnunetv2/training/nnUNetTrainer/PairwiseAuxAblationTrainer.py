from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch.jit import TracerWarning

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

warnings.filterwarnings("ignore", category=TracerWarning)


class PairwiseAuxAblationTrainer(nnUNetTrainer):
    boundary_weight = 0.2
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
        self.num_epochs = 200
        self.initial_lr = 1e-3

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _do_i_compile(self):
        return False

    @staticmethod
    def _cross_kernel_3d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        kernel = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
        kernel[0, 0, 1, 1, 1] = 1
        kernel[0, 0, 0, 1, 1] = 1
        kernel[0, 0, 2, 1, 1] = 1
        kernel[0, 0, 1, 0, 1] = 1
        kernel[0, 0, 1, 2, 1] = 1
        kernel[0, 0, 1, 1, 0] = 1
        kernel[0, 0, 1, 1, 2] = 1
        return kernel

    def make_3d_boundary(self, target: torch.Tensor, boundary_channels: int) -> torch.Tensor:
        if boundary_channels == 1:
            return self.make_binary_3d_boundary(target)

        return self.make_multiclass_3d_boundary(target, boundary_channels)

    def make_binary_3d_boundary(self, target: torch.Tensor) -> torch.Tensor:
        mask = (target > 0).float()
        kernel = self._cross_kernel_3d(mask.device, mask.dtype)
        kernel_sum = float(kernel.sum().item())

        dilated = mask
        eroded = mask
        for _ in range(self.boundary_radius):
            dilated = (F.conv3d(dilated, kernel, padding=1) > 0).float()
            eroded = (F.conv3d(eroded, kernel, padding=1) >= kernel_sum).float()

        return (dilated * (1.0 - eroded)).float()

    def make_multiclass_3d_boundary(self, target: torch.Tensor, boundary_channels: int) -> torch.Tensor:
        boundaries = []
        for class_id in range(1, boundary_channels + 1):
            mask = (target == class_id).float()
            kernel = self._cross_kernel_3d(mask.device, mask.dtype)
            kernel_sum = float(kernel.sum().item())

            eroded = mask
            for _ in range(self.boundary_radius):
                eroded = (F.conv3d(eroded, kernel, padding=1) >= kernel_sum).float()

            boundaries.append((mask * (1.0 - eroded)).float())

        return torch.cat(boundaries, dim=1)

    @staticmethod
    def boundary_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        reduce_axes = tuple(range(2, probs.ndim))
        intersection = torch.sum(probs * target, dim=reduce_axes)
        denominator = torch.sum(probs, dim=reduce_axes) + torch.sum(target, dim=reduce_axes)
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        return 1.0 - dice.mean()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target_for_boundary = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target_for_boundary = target

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_logits, boundary_logits = self.network(data, training=True)
            boundary_target = self.make_3d_boundary(target_for_boundary, boundary_logits.shape[1])
            seg_loss = self.loss(seg_logits, target)
            boundary_loss = self.boundary_dice_loss(boundary_logits, boundary_target)
            boundary_loss = boundary_loss + F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
            total_loss = seg_loss + self.boundary_weight * boundary_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": total_loss.detach().cpu().numpy()}
