项目说明
========

本项目基于 nnU-Net v2.7.0，主要用于医学图像分割，以及前列腺多模态 MRI 的分类实验。
项目同时包含标准 nnU-Net 流水线和若干自定义分类模型、训练脚本及推理脚本。


一、主要功能
============

1. nnU-Net 医学图像分割
   - 数据集格式检查与转换
   - 数据指纹分析、实验规划和预处理
   - 2D/3D 网络训练
   - 滑动窗口推理
   - 交叉验证、模型集成和后处理
   - Dice、IoU、HD95、ASSD、敏感度和精确率等指标计算

2. 前列腺 MRI 分类
   - 输入三种 MRI 模态：T2W、ADC、DWI
   - 支持四分类和二分类任务
   - 支持 3D ResNet-50、Swin Transformer、ViT、MONAI EfficientNet、DenseNet121、LMTTM-VMI、VLFATRollout、TomoGraphView、M3-Net 和 3D-MobiBrainNet
   - 支持 BiomedCLIP 临床文本融合
   - 支持 PMG 概率图引导的 PMG-CLIPNet
   - 支持类别均衡采样和训练曲线保存


二、目录结构
============

nnunetv2/                 nnU-Net 核心 Python 包
  dataset_conversion/     数据集转换
  experiment_planning/    指纹分析、规划和预处理
  preprocessing/          预处理模块
  training/               数据加载、损失、训练器和网络
  inference/              推理和预测导出
  evaluation/             结果评估和交叉验证
  postprocessing/         连通域等后处理
  model_sharing/          模型导入、导出和下载
  utilities/              通用工具

models/                   自定义分类模型
Options/                  分类训练参数定义
dataset_nnunet.py         三模态 MRI 数据集和增强
classification_sampling.py 类别均衡采样器

train_ResNet50_3D.py     3D ResNet-50 四分类训练
train_SwinT.py           3D Swin 分类训练
train_ViT.py             3D ViT 分类训练
train_stage1_binary.py   Stage-1 二分类训练
infer_stage1_efficientnet.py  EfficientNet Stage-1 推理
infer_stage1_densenet.py  DenseNet121 Stage-1 推理
infer_stage1_lmttm.py      LMTTM-VMI Stage-1 推理
infer_stage1_vlfat.py      VLFATRollout Stage-1 推理
infer_stage1_tomographview.py  TomoGraphView Stage-1 推理
infer_stage1_m3net.py       M3-Net Stage-1 推理
infer_stage1_mobibrainnet.py  3D-MobiBrainNet Stage-1 推理
infer_*.py               分类模型推理脚本
compute_seg_metrics.py   分割指标计算

nnUNet/nnUNet_raw/       原始数据集
nnUNet/nnUNet_preprocessed/  nnU-Net 预处理数据
nnUNet/nnUNet_results/   nnU-Net 分割模型结果
Checkpoint/              分类模型权重、日志和曲线
Prob/                    nnU-Net 输出的前列腺腺体概率图
BiomedCLIP/              本地 BiomedCLIP 模型文件

项目中的数据集编号约定如下：

    130 = AHCDU（腺体分割数据集）
    141 = PI-CAI（腺体分割数据集）

用于分类的完整数据集对应为：

    Dataset2301_FullAHCDU = AHCDU 分类数据集
    Dataset2302_FullPI-CAI = PI-CAI 分类数据集

因此，`Prob/130` 是 AHCDU 腺体概率图，`Prob/141` 是 PI-CAI 腺体概率图。
它们通常由 Dataset130/Dataset141 的 nnU-Net 腺体分割模型推理得到，不能当作分类标签或分割 GT 使用。


三、环境配置
============

建议使用 Python 3.10 或更高版本，并先按照当前 CUDA 环境安装匹配版本的 PyTorch。
然后安装项目依赖：

    pip install -e .

运行 nnU-Net 前必须设置以下环境变量：

    export nnUNet_raw=/root/autodl-tmp/nnUNetv2/nnUNet/nnUNet_raw
    export nnUNet_preprocessed=/root/autodl-tmp/nnUNetv2/nnUNet/nnUNet_preprocessed
    export nnUNet_results=/root/autodl-tmp/nnUNetv2/nnUNet/nnUNet_results

也可以直接执行：

    source Setting.sh

BiomedCLIP 实验还要求 `BiomedCLIP/` 下存在本地模型配置和权重文件。


四、nnU-Net 分割流程
====================

1. 准备数据

数据集目录应符合 nnU-Net 格式，例如：

    DatasetXXX_Name/
      imagesTr/
      labelsTr/
      imagesTs/
      labelsTs/
      dataset.json

多模态文件名应使用 `_0000`、`_0001`、`_0002` 等通道后缀。

2. 规划和预处理

    nnUNetv2_plan_and_preprocess -d 130 --verify_dataset_integrity

也可以使用数据集名称，例如：

    nnUNetv2_plan_and_preprocess -d Dataset130_ProstateAHCDU

3. 训练

    nnUNetv2_train 130 3d_fullres 0

使用自定义训练器时：

    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 130 3d_fullres 0 \
        -tr DIYBoundaryMSAGTrainer

常用参数包括：
    -c                 从已有 checkpoint 继续训练
    --val              只执行验证
    --val_best         使用最佳 checkpoint 验证
    --npz              保存验证概率
    -num_gpus N        使用多 GPU 训练

4. 推理

    nnUNetv2_predict \
        -i nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
        -o nnUNet/infer/Dataset130_DIYBoundaryMSAGTrainer/0 \
        -d 130 -c 3d_fullres -tr DIYBoundaryMSAGTrainer -f 0

5. 分割指标

    python compute_seg_metrics.py --help

具体参数以脚本中的命令行帮助为准。


五、分类实验
============

分类数据默认位于：

    nnUNet/nnUNet_raw/Dataset2301_FullAHCDU
    nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI

分类标签通常保存在 `labelsTr/A_labels.json` 或兼容的 `labels.json` 中。
标签值为 0 到 3。Stage-1 二分类会将原始类别 0 作为负类，将 1、2、3 合并为正类。

训练示例：

    python train_ResNet50_3D.py --help
    python train_SwinT.py --help
    python train_ViT.py --help
    python train_stage1_binary.py --help

`train_stage1_binary.py` 支持的模型包括：

    resnet50、efficientnet、densenet121、lmttm、vlfat、vit、swin、diy_clip、pmt_net

MONAI EfficientNet 使用 `efficientnet-b0` 的 3D 版本，输入为三模态 MRI，输出为 Stage-1 二分类。
训练示例：

    python train_stage1_binary.py --model efficientnet --dataset PICAI \
        --epochs 200 --batch-size 3 --task-name EfficientNet_PICAI --sample

推理示例：

    python infer_stage1_efficientnet.py --dataset PICAI \
        --task-name EfficientNet_PICAI --checkpoint best --batch-size 3 --gpu_id 0

MONAI DenseNet 使用 `DenseNet121` 的 3D 版本，训练和推理命令中的模型名为 `densenet121`。

LMTTM 使用独立的 3D LMTTM-VMI 风格分类器，输入三模态 MRI，训练时不读取
`Prob/130` 或 `Prob/141`，也不拼接 PMT 模块，作为独立对比基线。模型名为 `lmttm`：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_stage1_binary.py \
        --model lmttm --dataset PICAI --epochs 200 --batch-size 3 --lr 5e-6 \
        --task-name LMTTM_PICAI --sample --gpu_id 0 --num-workers 2

对应推理命令：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python infer_stage1_lmttm.py \
        --dataset PICAI --task-name LMTTM_PICAI --checkpoint best \
        --batch-size 3 --gpu_id 0

VLFATRollout 是独立的逐切片 Transformer 对比模型：每个深度切片使用
DeiT-B/16 提取空间特征，再使用时间 Transformer 聚合切片序列。它不读取
`Prob/130` 或 `Prob/141`，不使用 PMT 模块。模型名为 `vlfat`：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_stage1_binary.py \
        --model vlfat --dataset PICAI --epochs 200 --batch-size 1 --lr 6e-6 \
        --task-name VLFAT_PICAI --sample --gpu_id 0 --num-workers 4

推理命令：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python infer_stage1_vlfat.py \
        --dataset PICAI --task-name VLFAT_PICAI --checkpoint best \
        --batch-size 1 --gpu_id 0 --num-workers 4

TomoGraphView 是论文仓库中的全方向切片与球面图聚合方法。本项目提供一个不依赖
`torch_geometric`、可直接端到端训练的兼容基线：对三个轴生成 mean/max 六个视图，
共享 2D 编码器后使用完整图消息传递聚合。它不使用 PMT 概率图或临床文本，模型名为
`tomographview`：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_stage1_binary.py \
        --model tomographview --dataset PICAI --epochs 200 --batch-size 2 --lr 1e-4 \
        --task-name TomoGraphView_PICAI --sample --gpu_id 0 --num-workers 2

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python infer_stage1_tomographview.py \
        --dataset PICAI --task-name TomoGraphView_PICAI --checkpoint best \
        --batch-size 2 --gpu_id 0 --num-workers 2

M3-Net 是 Macro/Meso/Micro 多尺度 3D 分类模型。本项目将每个病例的三通道
T2W/ADC/DWI 体数据插值为 `32^3`、`64^3`、`96^3` 三个尺度，再进行跨尺度注意力融合。
由于原始 M3-Net 面向肺结节立方 patch，当前实现是前列腺 MRI 的兼容版本，不使用 PMT
概率图或临床信息。模型名为 `m3net`，建议 batch size 为 1：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_stage1_binary.py \
        --model m3net --dataset PICAI --epochs 200 --batch-size 1 --lr 1e-4 \
        --task-name M3Net_PICAI --sample --gpu_id 0 --num-workers 2

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python infer_stage1_m3net.py \
        --dataset PICAI --task-name M3Net_PICAI --checkpoint best \
        --batch-size 1 --gpu_id 0 --num-workers 2

3D-MobiBrainNet 原仓库是面向 ADNI 脑 MRI 的 AD/MCI/CN 三分类轻量 3D CNN，核心使用
depth-wise separable 3D convolution。本项目将首层改为三通道 T2W/ADC/DWI，并将分类头
改为 Stage-1 二分类；不使用 PMT 概率图和临床信息。模型名为 `mobibrainnet`：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_stage1_binary.py \
        --model mobibrainnet --dataset PICAI --epochs 200 --batch-size 2 --lr 1e-4 \
        --task-name MobiBrainNet_PICAI --sample --gpu_id 0 --num-workers 2

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python infer_stage1_mobibrainnet.py \
        --dataset PICAI --task-name MobiBrainNet_PICAI --checkpoint best \
        --batch-size 2 --gpu_id 0 --num-workers 2

训练结果会写入 `Checkpoint/<task-name>/`，包括：

    best.pt、latest.pt、metrics.jsonl
    loss.png、accuracy.png、precision.png、recall.png、auc.png

推理示例：

    python infer_ResNet50_3D_PICAI.py --help
    python infer_stage1_diy_clip.py --help
    python infer_stage1_pmt_net.py --help

PMTNet 使用腺体概率图引导分类：

    AHCDU 分类图像 Dataset2301_FullAHCDU + Prob/130/{case}.npz
    PI-CAI 分类图像 Dataset2302_FullPI-CAI + Prob/141/{case}.npz

`Prob` 文件是包含 `probabilities` 数组的 `.npz` 文件，数组通常为
`[2, D, H, W]`，PMTNet 读取第 1 个通道作为腺体前景概率，并将其输入 PMG 模块。
该概率图是分割模型的 softmax 预测结果，不是 GT；分类 GT 仍然来自分类数据集的标签文件。

推理脚本默认使用 `--split val`，因此会读取 `labelsTs` 对应的概率图。
如果找不到某个病例的概率图，脚本会使用全零概率图并记录缺失病例。


六、测试
========

运行单元测试：

    python -m unittest discover -s nnunetv2/tests -q

当前测试中，路径相关测试可以通过；外部 Trainer 查找相关测试在
`nnunetv2/utilities/find_class_by_name.py` 的导入清理逻辑处可能触发
`RuntimeError: dictionary changed size during iteration`，这是当前仓库已知问题。


七、注意事项
============

1. 训练和推理前确认三个 nnU-Net 环境变量已经生效。
2. GPU 编号通过 `CUDA_VISIBLE_DEVICES` 或分类脚本的 `--gpu_id` 指定。
3. 分类脚本依赖 MONAI、SimpleITK、scikit-learn、open_clip 和 transformers 等库。
4. 体数据尺寸、模态顺序和文件命名必须与数据集及模型设置一致。
5. `Checkpoint/`、`Prob/` 和本地数据目录可能占用大量磁盘空间，清理前请确认是否仍需使用。
