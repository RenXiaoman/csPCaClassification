export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"
clear

目前从事前列腺腺体分割，使用的数据集为PICAI(公开)和AHCDU(私有):
1.AHCDU数据集：130
2.PICAI数据集：131



模型架构
Baseline + 频域增强模块:                 nnunetv2/training/nnUNetTrainer/custom_networks/DIY/diy.py
Baseline + 频域增强模块 + 多尺度注意力门控: nnunetv2/training/nnUNetTrainer/custom_networks/DIY/DIY_boundary_msag.py
Baseline + 频域增强模块 + 多尺度注意力门控 + 分割边界辅助任务: nnunetv2/training/nnUNetTrainer/custom_networks/DIY/DIY_boundary_msag.py

Trainer说明:
Baseline + 频域增强模块 + 多尺度注意力门控: nnunetv2/training/nnUNetTrainer/DIYBoundaryMSAGTrainer.py
Baseline + 频域增强模块 + 多尺度注意力门控 + 分割边界辅助任务: nnunetv2/training/nnUNetTrainer/DIYBoundaryMSAGAuxTrainer.py

