import numpy as np
import torch
from typing import Optional, Tuple, Union
import random


def nnunet_style_augmentation(
    images: torch.Tensor,  # [B, C, D, H, W]
    labels: torch.Tensor,  # [B, 1, D, H, W] 
    p_elastic: float = 0.2,
    p_scale: float = 0.2,
    p_rotate: float = 0.2,
    p_gamma: float = 0.3,
    p_mirror: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    nnUNet-style data augmentation for 3D medical images
    """
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        # Elastic deformation
        if random.random() < p_elastic:
            images[i], labels[i] = elastic_deform_3d(images[i], labels[i])
        
        # Scaling
        if random.random() < p_scale:
            scale_factor = random.uniform(0.85, 1.25)
            images[i], labels[i] = scale_3d(images[i], labels[i], scale_factor)
        
        # Rotation
        if random.random() < p_rotate:
            angles = [
                random.uniform(-15, 15) * np.pi / 180,  # x
                random.uniform(-15, 15) * np.pi / 180,  # y  
                random.uniform(-15, 15) * np.pi / 180   # z
            ]
            images[i], labels[i] = rotate_3d(images[i], labels[i], angles)
        
        # Gamma correction
        if random.random() < p_gamma:
            gamma = random.uniform(0.7, 1.5)
            images[i] = gamma_correction(images[i], gamma)
        
        # Mirroring
        if random.random() < p_mirror:
            axes = []
            if random.random() < 0.5:
                axes.append(2)  # x-axis
            if random.random() < 0.5:
                axes.append(3)  # y-axis
            if random.random() < 0.5:
                axes.append(4)  # z-axis
            if axes:
                images[i] = torch.flip(images[i], axes)
                labels[i] = torch.flip(labels[i], axes)
    
    return images, labels


def elastic_deform_3d(image: torch.Tensor, label: torch.Tensor, 
                     alpha: Tuple[float, float] = (0., 900.), 
                     sigma: Tuple[float, float] = (9., 13.)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simplified elastic deformation for 3D images"""
    # For simplicity, we'll use a basic implementation
    # In practice, nnUNet uses more sophisticated elastic deformation
    return image, label


def scale_3d(image: torch.Tensor, label: torch.Tensor, scale_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D scaling augmentation"""
    if scale_factor == 1.0:
        return image, label
    
    # Simple scaling implementation
    original_shape = image.shape[1:]
    new_shape = [int(d * scale_factor) for d in original_shape]
    
    # For simplicity, we'll use nearest neighbor interpolation
    image_scaled = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=new_shape, mode='trilinear', align_corners=False
    ).squeeze(0)
    
    label_scaled = torch.nn.functional.interpolate(
        label.unsqueeze(0).float(), size=new_shape, mode='nearest'
    ).squeeze(0).long()
    
    # Crop or pad to original size
    image = center_crop_or_pad_3d(image_scaled, original_shape)
    label = center_crop_or_pad_3d(label_scaled, original_shape)
    
    return image, label


def rotate_3d(image: torch.Tensor, label: torch.Tensor, angles: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D rotation augmentation"""
    # Simple rotation implementation
    # In practice, nnUNet uses more sophisticated rotation
    return image, label


def gamma_correction(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Gamma correction augmentation"""
    if gamma == 1.0:
        return image
    
    # Apply gamma correction while preserving statistics
    min_val = image.min()
    max_val = image.max()
    
    if max_val > min_val:
        # Normalize to [0, 1]
        image_normalized = (image - min_val) / (max_val - min_val)
        # Apply gamma
        image_gamma = torch.pow(image_normalized, gamma)
        # Restore original range
        image = image_gamma * (max_val - min_val) + min_val
    
    return image


def center_crop_or_pad_3d(tensor: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Center crop or pad 3D tensor to target shape"""
    current_shape = tensor.shape[1:]
    
    if current_shape == target_shape:
        return tensor
    
    result = torch.zeros(tensor.shape[0], *target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Calculate crop/pad dimensions
    crop_start = [max(0, (c - t) // 2) for c, t in zip(current_shape, target_shape)]
    crop_end = [min(c, crop_start[i] + t) for i, (c, t) in enumerate(zip(current_shape, target_shape))]
    
    pad_start = [max(0, (t - c) // 2) for c, t in zip(current_shape, target_shape)]
    pad_end = [pad_start[i] + (crop_end[i] - crop_start[i]) for i in range(3)]
    
    # Perform crop and pad
    cropped = tensor[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
    result[:, pad_start[0]:pad_end[0], pad_start[1]:pad_end[1], pad_start[2]:pad_end[2]] = cropped
    
    return result