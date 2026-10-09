from pathlib import Path
from pprint import pprint
import random
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset, DataLoader
from typing import Union
from monai.visualize.utils import blend_images, matshow3d
from monai.transforms import ClipIntensityPercentiles, NormalizeIntensity, ScaleIntensity
import scipy.ndimage
# import pywt
# import dtcwt

# nnUNet data augmentation imports
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

def check_exists(*paths):
    """
    检查传入的所有路径是否存在，不存在就报错
    :param paths: 任意数量的 Path 或 str
    """
    for p in paths:
        p = Path(p)   # 保证类型安全
        if not p.exists():
            raise FileNotFoundError(f"❌ Path not found: {p}")

import numpy as np

# def process_nii_lowpass(data, nlevels=1):
#     """
#     对输入的 3D numpy 数据做 DTCWT 低频提取。

#     参数:
#         data: numpy.ndarray, shape = [D, H, W]
#         nlevels: DTCWT 分解层数

#     返回:
#         lowpass_data: numpy.ndarray, shape = [D, H_low, W_low]
#     """
#     if not isinstance(data, np.ndarray):
#         raise TypeError("data 必须是 numpy.ndarray")

#     if data.ndim != 3:
#         raise ValueError(f"data 必须是 3 维数组 [D, H, W]，当前 shape={data.shape}")

#     transform = dtcwt.Transform2d()
#     lowpass_slices = []

#     # 按 D 维遍历，每一张 slice 的 shape 是 [H, W]
#     for d in range(data.shape[0]):
#         slice_img = data[d, :, :]
#         transformed = transform.forward(slice_img, nlevels=nlevels)
#         LL = transformed.lowpass

#         lowpass_slices.append(LL)

#     # 堆叠回 [D, H_low, W_low]
#     lowpass_data = np.stack(lowpass_slices, axis=0)

#     return lowpass_data


# nnUNet default 3D augmentation parameters
default_3D_augmentation_params = {
    "do_elastic": False,
    # "elastic_deform_alpha": (0., 900.),
    # "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,
    # "do_scaling": True,
    # "scale_range": (0.7, 1.4),
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,
    # "do_rotation": True,
    # "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    # "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    # "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    # "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.8, 1.4),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1, 2),
    "border_mode_data": "constant",
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
}


class Lits_DataSet(Dataset):
    def __init__(self,
                 TASK: Path, 
                 imagesTr: Union[Path, str],
                 labelsTr: Union[Path, str],
                 size=(16, 256, 256),
                 augmentation_params=None,
                 enable_augmentation=True):
        self.TASK = TASK
        self.size = size
        self.enable_augmentation = enable_augmentation
        
        self.imagesTr = TASK / imagesTr
        self.labelsTr = TASK / labelsTr
        
        if (not self.labelsTr.exists()) or (not self.imagesTr.exists()):
            raise FileNotFoundError(f'one of labels and images dir not exist: {self.labelsTr}, {self.imagesTr}')
        
        self.labels_list = sorted(list(self.labelsTr.glob("*.nii.gz")))
        self.clip = ClipIntensityPercentiles(lower=0.5, upper=99.5, channel_wise=False)
        
        # Set augmentation parameters
        if augmentation_params is None:
            self.augmentation_params = default_3D_augmentation_params
        else:
            self.augmentation_params = augmentation_params
            
        # Create augmentation transforms
        self.transforms = self._create_augmentation_transforms() if enable_augmentation else None
    
    def __len__(self):
        return len(self.labels_list)
    
    def _create_augmentation_transforms(self):
        """Create nnUNet-style data augmentation transforms (only spatial transforms)"""
        tr_transforms = []
        
        # Spatial transforms only - intensity transforms are applied during preprocessing
        tr_transforms.append(SpatialTransform(
            patch_size=self.size,
            patch_center_dist_from_border=None,
            do_elastic_deform=self.augmentation_params.get("do_elastic"),
            alpha=self.augmentation_params.get("elastic_deform_alpha"),
            sigma=self.augmentation_params.get("elastic_deform_sigma"),
            do_rotation=self.augmentation_params.get("do_rotation"),
            angle_x=self.augmentation_params.get("rotation_x"),
            angle_y=self.augmentation_params.get("rotation_y"),
            angle_z=self.augmentation_params.get("rotation_z"),
            p_rot_per_axis=self.augmentation_params.get("rotation_p_per_axis"),
            do_scale=self.augmentation_params.get("do_scaling"),
            scale=self.augmentation_params.get("scale_range"),
            border_mode_data=self.augmentation_params.get("border_mode_data"),
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=-1,
            order_seg=1,
            random_crop=self.augmentation_params.get("random_crop"),
            p_el_per_sample=self.augmentation_params.get("p_eldef"),
            p_scale_per_sample=self.augmentation_params.get("p_scale"),
            p_rot_per_sample=self.augmentation_params.get("p_rot"),
            independent_scale_for_each_axis=self.augmentation_params.get("independent_scale_factor_for_each_axis")
        ))
        
        # Mirroring transform
        if self.augmentation_params.get("do_mirror") or self.augmentation_params.get("mirror"):
            tr_transforms.append(MirrorTransform(self.augmentation_params.get("mirror_axes")))
        
        return Compose(tr_transforms)
    
    def _apply_intensity_transforms(self, img):
        """Apply nnUNet-style intensity transforms to single image"""
        
        # Gaussian noise
        if random.random() < 0.1:
            noise = np.random.normal(0, 0.1, img.shape)
            img = img + noise
        
        # Gaussian blur
        if random.random() < 0.2:
            sigma = random.uniform(0.5, 1.0)
            img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
        
        # Brightness multiplicative
        if random.random() < 0.15:
            multiplier = random.uniform(0.75, 1.25)
            img = img * multiplier
        
        # Contrast augmentation
        if random.random() < 0.15:
            mean_val = np.mean(img)
            contrast_factor = random.uniform(0.75, 1.25)
            img = mean_val + contrast_factor * (img - mean_val)
        
        # Gamma correction
        if random.random() < 0.3:  # p_gamma from params
            gamma = random.uniform(0.7, 1.5)  # gamma_range from params
            # Preserve statistics
            min_val, max_val = np.min(img), np.max(img)
            if max_val > min_val:
                img_normalized = (img - min_val) / (max_val - min_val)
                img_gamma = np.power(img_normalized, gamma)
                img = img_gamma * (max_val - min_val) + min_val
        
        return img
    
    def __getitem__(self, idx):
        img_path = Path(self.labels_list[idx])
        patient_name = str(img_path.name).split('.')[0]  # '154.nii.gz'
        
        T2W_path = self.imagesTr / Path(str(patient_name) + '_0000.nii.gz')
        ADC_path = self.imagesTr / Path(str(patient_name) + '_0001.nii.gz')
        DWI_path = self.imagesTr / Path(str(patient_name) + '_0002.nii.gz')
        GT_path = img_path
        
        # 验证上面3个path是否存在
        check_exists(T2W_path, ADC_path, DWI_path, GT_path)
        
        T2W_raw = self.load(T2W_path).astype(np.float32)  # [D,H,W]
        ADC_raw = self.load(ADC_path).astype(np.float32)
        DWI_raw = self.load(DWI_path).astype(np.float32)
        gt      = self.load(GT_path).astype(np.float32)
        
        # 预处理: DTCWT低频提取
        # ADC_raw = process_nii_lowpass(ADC_raw)
        # DWI_raw = process_nii_lowpass(DWI_raw)
                
        
        # 清理标签值, 将所有大于0的值转为1(二分类问题)
        gt[gt > 0] = 1.0
        
        # 基础预处理: clip + z-score归一化
        T2W = self.z_score_normalization(self.clip(T2W_raw))  # -> [D,H,W]
        ADC = self.z_score_normalization(self.clip(ADC_raw))  # -> [D,H,W]
        DWI = self.z_score_normalization(self.clip(DWI_raw))
        
        
        # 应用强度变换 (nnUNet风格)
        if self.enable_augmentation:
            ADC = self._apply_intensity_transforms(ADC)
            DWI = self._apply_intensity_transforms(DWI)
            T2W = self._apply_intensity_transforms(T2W)
            
        
        
        # 添加通道维度
        T2W = T2W[np.newaxis, :]  # [1, D, H, W]
        ADC = ADC[np.newaxis, :]  # [1, D, H, W]
        DWI = DWI[np.newaxis, :]
        gt = gt[np.newaxis, :]    # [1, D, H, W]
        
        # 合并多模态数据 [3, D, H, W]
        image_data = np.concatenate([ADC, DWI, T2W], axis=0)
        
        # 应用数据增强
        if self.enable_augmentation and self.transforms is not None:
            # Prepare data in batchgenerators format
            data_dict = {
                'data': image_data[np.newaxis, ...],  # [1, 3, D, H, W]
                'seg': gt[np.newaxis, ...]           # [1, 1, D, H, W]
            }
            
            # Apply transforms
            data_dict = self.transforms(**data_dict)
            
            # Extract augmented data
            image_data = data_dict['data'][0]  # [3, D, H, W]
            gt = data_dict['seg'][0]           # [1, D, H, W]
            
            # Split back into individual modalities
            ADC = image_data[0:1, ...]  # [1, D, H, W]
            DWI = image_data[1:2, ...]  # [1, D, H, W]
            T2W = image_data[2:3, ...]  # [1, D, H, W]

        return ADC, DWI, T2W, gt, str(patient_name)
    
    
    def load(self, file: Union[Path, str]) -> np.ndarray:
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image
    
    def normalization(self, img, lmin=1, rmax=None, dividend=None, quantile=None):
        newimg = img.copy()
        newimg = newimg.astype(np.float32)
        if quantile is not None:
            maxval = round(np.percentile(newimg, 100 - quantile))
            minval = round(np.percentile(newimg, quantile))
            newimg[newimg >= maxval] = maxval
            newimg[newimg <= minval] = minval

        if lmin is not None:
            newimg[newimg < lmin] = lmin
            
        if rmax is not None:
            newimg[newimg > rmax] = rmax

        minval = np.min(newimg)
        if dividend is None:
            maxval = np.max(newimg)
            newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
        else:
            newimg = (np.asarray(newimg).astype(np.float32) - minval) / dividend
        return newimg
    
    def min_max_normalization(self, img):
        out = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
        return out
    
    def z_score_normalization(self, img):
        # 只计算非零区域的均值和标准差
        non_zero_mask = img > 0
        if np.any(non_zero_mask):
            mean_val = np.mean(img[non_zero_mask])
            std_val = np.std(img[non_zero_mask])
        else:
            mean_val = np.mean(img)
            std_val = np.std(img)
        out = (img - mean_val) / (std_val + 1e-8)
        return out
    

    
if __name__ == "__main__":
    TASK = Path('./dataset/Task205_picai_lesion/')
    
    imagesTr = 'prostate158_imagesTr'
    labelsTr = 'prostate158_labelsTr'
    
    if not TASK.exists():
        raise 'Dir not exist !!!'
    
    
    train_dataset = Lits_DataSet(TASK, imagesTr, labelsTr)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=1,
                                  num_workers=4,
                                  shuffle=True)
    print(f'length :{len(train_dataset)}')
    for id, (T2W, ADC, gt, patient_name) in enumerate(train_dataloader):  # [1, 5, 16, 128, 128]
        pass
        if id >0 :
            break
        print(f'T2W shape : {T2W.shape}, ADC shape : {ADC.shape}, patient_name : {patient_name}')
        ###########    [5, 1, 16, 128, 128] [5, 1, 16, 128, 128] [1, 5, 16, 128, 128]
        # b, c, l, w, e = T2W.shape[0], T2W.shape[1], T2W.shape[2], T2W.shape[3], T2W.shape[4]
        
        # blended = blend_images(T2W[:, 1], gt[:, 1], alpha=0.5, cmap="hsv")
        # fig = plt.figure()
        # matshow3d(blended, fig=fig, title=f"{patient_name} (T2W+GT blended)",
        #   channel_dim=0, frame_dim=1)
        # plt.show()
        # plt.savefig("blended_volume.png", dpi=300, bbox_inches="tight")
        
    

