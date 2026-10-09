#!/usr/bin/env bash
set -euo pipefail

export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"
clear

GPU_ID="${GPU_ID:-0}"

# Base.
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetBaselineTrainer

# Base + WFE.
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetWFETrainer

# Base + CMSAG.
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetCMSAGTrainer

# Base + Auxiliary task.
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetAuxTrainer

# Base + WFE + Auxiliary task.
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetWFEAuxTrainer
# Base + CMSAG + Auxiliary task.
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetCMSAGAuxTrainer

# Base + WFE + CMSAG.
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 141 3d_fullres 0 -tr FreGNetWFECMSAGTrainer

# Full model (Base + WFE + CMSAG + Auxiliary task).
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 141 3d_fullres 0 -tr DIYBoundaryCMSAGAuxTrainer


# 噪声实验
CUDA_VISIBLE_DEVICES=3  python paper_tools/noise_quality/noise_robustness_experiment.py \
--freGnet_checkpoint \
nnUNet/nnUNet_results/Dataset141_FullPICAI/DIYBoundaryCMSAGAuxTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth \
--baseline_checkpoint \
nnUNet/nnUNet_results/Dataset141_FullPICAI/FreGNetBaselineTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth \
--output_dir \
paper_tools/noise_quality \
--norm
