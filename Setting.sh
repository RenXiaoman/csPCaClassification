export nnUNet_preprocessed="/root/autodl-tmp/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/root/autodl-tmp/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/root/autodl-tmp/nnUNetv2/nnUNet/nnUNet_raw"
clear


nnUNetv2_generate_dataset_json -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/imagesTr \
-l nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr -o nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI \
--task_name Dataset2302_FullPI-CAI --modalities 0 1 2

ssh -p 44965 root@connect.westd.seetacloud.com

nnUNetv2_plan_and_preprocess -d 2302 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 130 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 131 --verify_dataset_integrity

# Baseline + 频域增强模块 + 多尺度注意力门控
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 130 3d_fullres 2 -tr DIYBoundaryMSAGTrainer  --c # AHCDU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 131 3d_fullres 1 -tr DIYBoundaryMSAGTrainer  --c # PICAI

# Baseline + 频域增强模块 + 多尺度注意力门控 + 分割边界辅助任务
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 130 3d_fullres 0 -tr DIYBoundaryMSAGAuxTrainer
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 0 -tr DIYBoundaryMSAGAuxTrainer



nnUNetv2_train 2302 3d_fullres 0 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 1 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 2 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 3 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 4 -tr CDSATrainer

cp -r /home/Space/clib/Projects/CAD/dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU  nnUNet/nnUNet_raw
cp -r /home/Space/clib/Projects/CAD/dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI nnUNet/nnUNet_raw


nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 130 -c 3d_fullres --save_probabilities


nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
  -o nnUNet/infer/Dataset130_DIYBoundaryMSAGTrainer/2 \
  -d 130 \
  -c 3d_fullres \
  -tr DIYBoundaryMSAGTrainer \
  -f 2 
  # -chk checkpoint_best.pth

nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI/imagesTs \
  -o nnUNet/infer/Dataset131_DIYBoundaryMSAGTrainer/1 \
  -d 131 \
  -c 3d_fullres \
  -tr DIYBoundaryMSAGTrainer \
  -f 1 
  # -chk checkpoint_best.pth