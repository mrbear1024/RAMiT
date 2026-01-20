"""
================================================================================
dataset.py - 数据集类定义模块
================================================================================
本模块定义了 RAMiT 项目用于图像恢复任务的各种数据集类。

支持的任务类型:
    - 超分辨率 (Super-Resolution, SR): DIV2K, DF2K 数据集
    - 去噪 (Denoising, DN): DFBW 数据集 (DIV2K + Flickr2K + BSDS500 + WED)
    - 低光增强 (Low-Light Enhancement, LLE): LOL + VELOL 数据集
    - 雨天去除 (Deraining, DR): Rain13K 数据集
    - CBCT医学影像去噪: 灰度CBCT图像数据集

数据流:
    .npy文件 -> DataLoader -> 随机裁剪 -> 数据增强(翻转/旋转) -> 训练样本对(HQ, LQ)

关键设计:
    1. 所有数据集类继承自 DatasetRandomCrop 基类
    2. 支持动态裁剪位置预加载 (patch_load) 以实现可复现性
    3. 支持渐进式学习 (progressive learning) 的 patch size 变化
    4. LQ图像可由HQ图像动态添加噪声生成 (去噪任务)
================================================================================
"""

import os
import re
import cv2
import glob
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def make_some_noise(img, sigma, seed=None):
    """
    为图像添加高斯噪声 (用于去噪任务的LQ图像生成)

    Args:
        img (torch.Tensor): 输入图像张量，形状为 (C, H, W)，值域 [0, 1]
        sigma (tuple or list): 噪声强度范围
            - 若 len(sigma) == 1: 使用固定噪声强度 sigma[0]
            - 若 len(sigma) == 2: 从 [sigma[0], sigma[1]] 均匀采样噪声强度
        seed (int, optional): 随机种子，用于确保噪声生成的可复现性

    Returns:
        torch.Tensor: 添加噪声后的图像，形状与输入相同

    Note:
        噪声强度 sigma 以像素值 [0, 255] 为基准定义，
        因此实际添加的噪声为 randn * (s / 255)，
        其中 s 是采样的噪声强度值。
    """
    # 从 sigma 范围内均匀采样一个噪声强度值
    s = np.random.uniform(sigma[0], sigma[1]) if len(sigma)!=1 else sigma[0]

    # 设置随机种子以确保可复现性
    if seed is not None:
        torch.random.manual_seed(seed)

    # 生成高斯噪声并添加到图像
    # 噪声强度从 [0, 255] 范围转换到 [0, 1] 范围
    noise = torch.randn(*img.shape)*s/255
    img = (img + noise)
    return img


class DatasetRandomCrop(Dataset):
    """
    随机裁剪数据集基类

    所有任务特定的数据集类都继承此基类。该类实现了：
    1. 从 HQ/LQ 图像对中随机裁剪固定大小的 patch
    2. 数据增强：随机水平翻转和随机旋转(0°, 90°, 180°, 270°)
    3. 可选的裁剪位置预加载 (patch_load) 机制

    Attributes:
        crop_size (int): LQ 图像裁剪的 patch 大小
        model_time (str): 模型标识符/训练时间戳
        epoch_dict (dict or None): 预加载的裁剪参数字典
            - None: 每次动态随机生成裁剪参数
            - dict: 使用预先生成的裁剪参数，确保跨 epoch 的一致性
        gray (bool): 是否将图像转换为灰度图
        scale (int): 超分辨率缩放因子 (去噪等任务为 1)
        file_list_hq (list): HQ (高质量) 图像文件路径列表
        file_list_lq (list or None): LQ (低质量) 图像文件路径列表
            - None: LQ 图像由 HQ 图像动态生成 (如添加噪声)
    """

    def __init__(self, crop_size, model_time, patch_load, data_name=None, gray=False):
        """
        初始化数据集基类

        Args:
            crop_size (int): 裁剪的 patch 大小 (针对 LQ 图像)
            model_time (str): 模型标识符，用于日志和保存路径
            patch_load (bool): 是否使用预加载的裁剪参数
                - True: 使用 epoch_dict 存储裁剪参数
                - False: 每次动态随机生成裁剪参数
            data_name (str, optional): 数据集名称标识
            gray (bool): 是否转换为灰度图像 (默认 False)
        """
        super(DatasetRandomCrop, self).__init__()
        assert model_time is not None, "model_time should be assigned!"

        self.crop_size = crop_size
        self.model_time = model_time
        # patch_load=True 时初始化空字典，用于存储预生成的裁剪参数
        self.epoch_dict = dict() if patch_load else None
        self.gray = gray

    def __getitem__(self, idx):
        """
        获取数据集中的一个训练样本

        Args:
            idx (int): 样本索引 (可能大于实际文件数量，通过取模实现循环)

        Returns:
            tuple: (crop_hq, crop_lq)
                - crop_hq (torch.Tensor): 裁剪后的 HQ 图像 patch
                - crop_lq (torch.Tensor): 裁剪后的 LQ 图像 patch

        数据处理流程:
            1. 索引映射: idx 取模映射到实际文件索引
            2. 加载图像: 从 .npy 文件加载并归一化到 [0, 1]
            3. 灰度转换: 若 gray=True，将 RGB 转换为灰度
            4. 处理 alpha 通道: 4通道图像转换为3通道 RGB
            5. 添加噪声: 若定义了 sigma 属性，为 LQ 添加高斯噪声
            6. 随机裁剪: 从图像中裁剪固定大小的 patch
            7. 数据增强: 随机水平翻转 + 随机旋转
        """
        # 保存原始索引用于 epoch_dict 查找
        iidx = idx
        # 取模以循环使用文件列表
        idx%=len(self.file_list_hq)

        # 如果使用预加载裁剪参数
        if self.epoch_dict is not None:
            if len(self.epoch_dict[iidx])==1:
                # 单组参数: 直接使用
                crop_h, crop_w, random_flip, random_rotate = self.epoch_dict[iidx][0]
            else:
                # 双组参数: 根据索引奇偶性选择
                aa = 0 if iidx%2==0 else 1
                crop_h, crop_w, random_flip, random_rotate = self.epoch_dict[iidx][aa]
            # 对于 x3, x4 超分任务，需要调整裁剪坐标
            if self.scale not in [1,2]: # only for SR x3, x4
                crop_h = int(crop_h/self.scale*2)
                crop_w = int(crop_w/self.scale*2)

        # 获取 HQ 和 LQ 文件路径
        file_hq = self.file_list_hq[idx]
        file_lq = self.file_list_lq[idx] if self.file_list_lq is not None else None

        # 加载 HQ 图像
        if not self.gray:
            # RGB 模式: 直接加载并归一化
            img_hq = torch.from_numpy(np.load(file_hq))/255
        else:
            # 灰度模式: 加载 -> CHW转HWC -> RGB转灰度 -> 添加通道维度
            img_hq = np.transpose(np.load(file_hq), (1,2,0))
            img_hq = cv2.cvtColor(img_hq, cv2.COLOR_RGB2GRAY)
            img_hq = torch.from_numpy(img_hq).unsqueeze(0)/255

        # 加载 LQ 图像 (若无 LQ 文件则克隆 HQ)
        img_lq = torch.from_numpy(np.load(file_lq))/255 if file_lq is not None else torch.clone(img_hq)

        # 处理 4 通道 (RGBA) 图像，转换为 3 通道 RGB
        if img_hq.size(0) == 4:
            img_hq = TF.to_tensor(TF.to_pil_image(img_hq).convert('RGB'))
            img_lq = TF.to_tensor(TF.to_pil_image(img_hq).convert('RGB'))

        # 获取 LQ 图像尺寸
        _, lq_h, lq_w = img_lq.size()

        # 为 LQ 图像添加噪声 (仅当定义了 sigma 属性时，用于去噪任务)
        if hasattr(self, 'sigma'):
            img_lq = make_some_noise(img_lq, self.sigma)

        # 随机裁剪 patch
        if self.epoch_dict is None:
            # 动态生成随机裁剪位置
            crop_h = torch.randint(0, lq_h-self.crop_size, (1,)).item()
            crop_w = torch.randint(0, lq_w-self.crop_size, (1,)).item()

        # 裁剪 LQ patch (crop_size x crop_size)
        crop_lq = TF.crop(img_lq, crop_h, crop_w, self.crop_size, self.crop_size)
        # 裁剪 HQ patch (crop_size*scale x crop_size*scale)
        # 超分任务中 HQ 图像尺寸是 LQ 的 scale 倍
        crop_hq = TF.crop(img_hq, crop_h*self.scale, crop_w*self.scale, self.crop_size*self.scale, self.crop_size*self.scale)

        # 随机水平翻转和随机旋转 (数据增强)
        if self.epoch_dict is None:
            random_flip = torch.randint(0,2,(1,)).item()    # 0 或 1
            random_rotate = torch.randint(0,4,(1,)).item()  # 0, 1, 2, 3 对应 0°, 90°, 180°, 270°

        # 应用水平翻转
        crop_hq, crop_lq = (TF.hflip(crop_hq), TF.hflip(crop_lq)) if random_flip else (crop_hq, crop_lq)
        # 应用旋转
        crop_hq, crop_lq = TF.rotate(crop_hq, angle=90*random_rotate), TF.rotate(crop_lq, angle=90*random_rotate)

        return crop_hq, crop_lq


# =============================================================================
# 任务特定的数据集类
# =============================================================================

class DIV2KDatasetRandomCrop(DatasetRandomCrop):
    """
    DIV2K 超分辨率数据集

    用于图像超分辨率 (SR) 任务训练，支持 x2, x3, x4 倍放大。

    目录结构:
        ../DIV2K/
        ├── DIV2K_train_HR/           # 高分辨率原图 (800张)
        │   └── *.npy
        └── DIV2K_train_LR_bicubic/   # 双三次下采样的低分辨率图像
            ├── X2/                   # 2倍下采样
            ├── X3/                   # 3倍下采样
            └── X4/                   # 4倍下采样

    Attributes:
        scale (int): 从 LQ 路径自动解析的缩放因子 (2, 3, 或 4)
    """

    def __init__(self, root_hq, root_lq, crop_size, model_time=None, patch_load=False):
        """
        初始化 DIV2K 数据集

        Args:
            root_hq (str): HQ 图像目录路径
            root_lq (str): LQ 图像目录路径 (路径末尾应包含缩放因子，如 'X2')
            crop_size (int): LQ 图像的裁剪 patch 大小
            model_time (str, optional): 模型标识符
            patch_load (bool): 是否预加载裁剪参数
        """
        super(DIV2KDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DIV2K')
        # 从 LQ 路径中提取缩放因子 (如 '/X2' -> 2)
        self.scale = int(re.search('.+(\d)', root_lq).group(1))

        # 加载并排序文件列表
        self.file_list_hq = sorted(glob.glob(os.path.join(root_hq, '*.npy')))
        self.file_list_lq = sorted(glob.glob(os.path.join(root_lq, '*.npy')))

    def __len__(self):
        """
        返回数据集的有效长度

        Note:
            返回 800*80 = 64000，即每个原始图像被使用 80 次
            这种设计允许在一个 epoch 中从每张图像采样多个不同的 patch
        """
        return len(self.file_list_hq)*80


class DF2KDatasetRandomCrop(DatasetRandomCrop):
    """
    DF2K 超分辨率数据集 (DIV2K + Flickr2K 合并)

    用于图像超分辨率 (SR) 任务训练，合并了 DIV2K 和 Flickr2K 数据集。
    DF2K 是更大规模的训练集，包含约 3450 张图像。

    目录结构:
        ../DIV2K/
        ├── DIV2K_train_HR/               # DIV2K HQ (800张)
        └── DIV2K_train_LR_bicubic/X{2,3,4}/  # DIV2K LQ

        ../Flickr2K/
        ├── Flickr2K_HR/                  # Flickr2K HQ (2650张)
        └── Flickr2K_LR_bicubic/X{2,3,4}/ # Flickr2K LQ
    """

    def __init__(self, D_root_hq, D_root_lq, F_root_hq, F_root_lq, crop_size, model_time=None, patch_load=False):
        """
        初始化 DF2K 数据集

        Args:
            D_root_hq (str): DIV2K HQ 图像目录
            D_root_lq (str): DIV2K LQ 图像目录
            F_root_hq (str): Flickr2K HQ 图像目录
            F_root_lq (str): Flickr2K LQ 图像目录
            crop_size (int): LQ 图像的裁剪 patch 大小
            model_time (str, optional): 模型标识符
            patch_load (bool): 是否预加载裁剪参数
        """
        super(DF2KDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DF2K')
        self.scale = int(re.search('.+(\d)', D_root_lq).group(1))

        # 分别加载 DIV2K 和 Flickr2K 的文件列表
        D_file_list_hq = sorted(glob.glob(os.path.join(D_root_hq, '*.npy')))
        F_file_list_hq = sorted(glob.glob(os.path.join(F_root_hq, '*.npy')))
        D_file_list_lq = sorted(glob.glob(os.path.join(D_root_lq, '*.npy')))
        F_file_list_lq = sorted(glob.glob(os.path.join(F_root_lq, '*.npy')))

        # 合并两个数据集
        self.file_list_hq = D_file_list_hq + F_file_list_hq
        self.file_list_lq = D_file_list_lq + F_file_list_lq

    def __len__(self):
        """
        返回数据集的有效长度

        Note:
            乘数 18.551 是根据 DF2K 图像的平均尺寸计算得出的，
            确保每个 epoch 的样本数量合理
        """
        return int(len(self.file_list_hq)*18.551)


class DFBWDatasetRandomCrop(DatasetRandomCrop):
    """
    DFBW 去噪数据集 (DIV2K + Flickr2K + BSDS500 + WED)

    用于图像去噪 (DN) 任务训练。该数据集只有 HQ 图像，
    LQ 图像通过动态添加高斯噪声生成。

    特点:
        - 无预设的 LQ 图像，噪声在训练时动态添加
        - 支持可变噪声强度范围 (sigma)
        - 可选灰度模式

    目录结构:
        ../DIV2K/DIV2K_train_HR/    # DIV2K HQ
        ../Flickr2K/Flickr2K_HR/    # Flickr2K HQ
        ../BSDS500/HQ/              # BSDS500 HQ
        ../WED/HQ/                  # WED HQ
    """

    def __init__(self, D_root_hq, F_root_hq, B_root_hq, W_root_hq, crop_size, sigma, gray=False, model_time=None, patch_load=False):
        """
        初始化 DFBW 去噪数据集

        Args:
            D_root_hq (str): DIV2K HQ 图像目录
            F_root_hq (str): Flickr2K HQ 图像目录
            B_root_hq (str): BSDS500 HQ 图像目录
            W_root_hq (str): WED HQ 图像目录
            crop_size (int): 裁剪 patch 大小
            sigma (tuple): 噪声强度范围 [sigma_min, sigma_max]
            gray (bool): 是否转换为灰度图 (默认 False)
            model_time (str, optional): 模型标识符
            patch_load (bool): 是否预加载裁剪参数
        """
        super(DFBWDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DFBW', gray)
        # 去噪任务的 scale 为 1 (HQ 和 LQ 尺寸相同)
        self.scale = 1

        # 加载四个数据集的 HQ 文件列表
        D_file_list_hq = sorted(glob.glob(os.path.join(D_root_hq, '*.npy')))
        F_file_list_hq = sorted(glob.glob(os.path.join(F_root_hq, '*.npy')))
        B_file_list_hq = sorted(glob.glob(os.path.join(B_root_hq, '*.npy')))
        W_file_list_hq = sorted(glob.glob(os.path.join(W_root_hq, '*.npy')))

        # 合并所有 HQ 文件
        self.file_list_hq = D_file_list_hq + F_file_list_hq + B_file_list_hq + W_file_list_hq
        # LQ 设为 None，表示通过添加噪声动态生成
        self.file_list_lq = None

        # 存储噪声强度范围，在 __getitem__ 中使用
        self.sigma = sigma

    def __len__(self):
        """返回数据集有效长度 (每张图像使用 3 次)"""
        return len(self.file_list_hq)*3


class LLEDatasetRandomCrop(DatasetRandomCrop):
    """
    低光增强 (LLE) 数据集 (LOL + VELOL)

    用于低光图像增强任务训练，合并了 LOL 和 VELOL 两个数据集。

    目录结构:
        ../LOL/
        ├── HQ/     # 正常曝光图像
        └── LQ/     # 低光图像

        ../VELOL/
        ├── HQ/     # 正常曝光图像
        └── LQ/     # 低光图像
    """

    def __init__(self, L_root_hq, V_root_hq, L_root_lq, V_root_lq, crop_size, model_time=None, patch_load=False):
        """
        初始化 LLE 数据集

        Args:
            L_root_hq (str): LOL HQ 图像目录
            V_root_hq (str): VELOL HQ 图像目录
            L_root_lq (str): LOL LQ 图像目录
            V_root_lq (str): VELOL LQ 图像目录
            crop_size (int): 裁剪 patch 大小
            model_time (str, optional): 模型标识符
            patch_load (bool): 是否预加载裁剪参数
        """
        super(LLEDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'LLE')
        self.scale = 1  # LLE 任务 HQ/LQ 尺寸相同

        # 加载 LOL 和 VELOL 的文件列表
        L_file_list_hq = sorted(glob.glob(os.path.join(L_root_hq, '*.npy')))
        V_file_list_hq = sorted(glob.glob(os.path.join(V_root_hq, '*.npy')))
        L_file_list_lq = sorted(glob.glob(os.path.join(L_root_lq, '*.npy')))
        V_file_list_lq = sorted(glob.glob(os.path.join(V_root_lq, '*.npy')))

        # 合并数据集
        self.file_list_hq = L_file_list_hq + V_file_list_hq
        self.file_list_lq = L_file_list_lq + V_file_list_lq

    def __len__(self):
        """返回数据集有效长度"""
        return int(len(self.file_list_lq)*14.006)


class DRDatasetRandomCrop(DatasetRandomCrop):
    """
    雨天去除 (DR) 数据集 (Rain13K)

    用于图像去雨任务训练。

    目录结构:
        ../Rain13K/
        ├── HQ/     # 无雨清晰图像
        └── LQ/     # 带雨图像
    """

    def __init__(self, root_hq, root_lq, crop_size, model_time=None, patch_load=False):
        """
        初始化 DR 去雨数据集

        Args:
            root_hq (str): HQ (无雨) 图像目录
            root_lq (str): LQ (带雨) 图像目录
            crop_size (int): 裁剪 patch 大小
            model_time (str, optional): 模型标识符
            patch_load (bool): 是否预加载裁剪参数
        """
        super(DRDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DR')
        self.scale = 1  # 去雨任务 HQ/LQ 尺寸相同

        self.file_list_hq = sorted(glob.glob(os.path.join(root_hq, '*.npy')))
        self.file_list_lq = sorted(glob.glob(os.path.join(root_lq, '*.npy')))

    def __len__(self):
        """返回数据集有效长度"""
        return int(len(self.file_list_lq)*1.8234)


class CBCTGrayDenoiseDatasetRandomCrop(DatasetRandomCrop):
    """
    CBCT 医学影像灰度去噪数据集

    专门用于 CBCT (Cone Beam Computed Tomography) 医学影像的去噪任务。

    特点:
        1. 处理灰度图像 (单通道)
        2. 支持递归扫描子目录
        3. HQ/LQ 文件必须一一对应 (相同的相对路径)
        4. 自动处理多种数组维度格式 (2D, CHW, HWC)

    目录结构示例:
        datasets_npy/
        ├── HQ/
        │   └── training_set/
        │       ├── case_001/
        │       │   ├── slice_001.npy
        │       │   └── slice_002.npy
        │       └── case_002/
        │           └── ...
        └── LQ/
            └── training_set/
                ├── case_001/
                │   ├── slice_001.npy  # 必须与 HQ 中的文件一一对应
                │   └── slice_002.npy
                └── case_002/
                    └── ...
    """

    def __init__(self, root_hq, root_lq, crop_size, model_time=None, patch_load=False, recursive=True):
        """
        初始化 CBCT 灰度去噪数据集

        Args:
            root_hq (str): HQ 图像根目录
            root_lq (str): LQ 图像根目录
            crop_size (int): 裁剪 patch 大小
            model_time (str, optional): 模型标识符
            patch_load (bool): 是否预加载裁剪参数
            recursive (bool): 是否递归扫描子目录 (默认 True)

        Raises:
            ValueError: 当 HQ/LQ 文件不匹配或找不到配对文件时
        """
        # 初始化基类，设置 gray=True 表示灰度模式
        super(CBCTGrayDenoiseDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'CBCT', True)
        self.scale = 1  # 去噪任务无缩放
        self.root_hq = Path(root_hq)
        self.root_lq = Path(root_lq)
        self.recursive = recursive

        # 扫描 HQ 和 LQ 目录中的所有 .npy 文件
        hq_paths = self._list_npy(self.root_hq)
        lq_paths = self._list_npy(self.root_lq)

        # 构建相对路径到绝对路径的映射
        # 这确保了 HQ 和 LQ 文件通过相对路径配对
        hq_map = {p.relative_to(self.root_hq): p for p in hq_paths}
        lq_map = {p.relative_to(self.root_lq): p for p in lq_paths}

        # 找出共同的相对路径 (有效的配对)
        common = sorted(set(hq_map.keys()) & set(lq_map.keys()))
        # 检查是否有不匹配的文件
        missing_hq = sorted(set(lq_map.keys()) - set(hq_map.keys()))
        missing_lq = sorted(set(hq_map.keys()) - set(lq_map.keys()))

        # 如果存在不匹配，报错
        if missing_hq or missing_lq:
            raise ValueError(
                "HQ/LQ mismatch. "
                f"missing_hq={missing_hq[:3]} missing_lq={missing_lq[:3]}"
            )
        if not common:
            raise ValueError(f"No paired .npy files found under: {root_hq} and {root_lq}")

        # 构建配对的文件列表
        self.file_list_hq = [str(hq_map[p]) for p in common]
        self.file_list_lq = [str(lq_map[p]) for p in common]

    def _list_npy(self, root: Path):
        """
        列出目录下所有 .npy 文件

        Args:
            root (Path): 根目录路径

        Returns:
            list[Path]: 排序后的 .npy 文件路径列表

        Raises:
            ValueError: 目录不存在时
        """
        if not root.exists():
            raise ValueError(f"Path not found: {root}")
        if self.recursive:
            # 递归扫描所有子目录
            return sorted(root.rglob("*.npy"))
        # 仅扫描当前目录
        return sorted(root.glob("*.npy"))

    def _load_gray_npy(self, path: str) -> torch.Tensor:
        """
        加载 .npy 文件并转换为灰度张量

        支持多种输入格式:
            - 2D 数组 (H, W): 直接作为灰度图
            - 3D 数组 (C, H, W) 或 (H, W, C): 自动检测并转换

        Args:
            path (str): .npy 文件路径

        Returns:
            torch.Tensor: 形状为 (1, H, W) 的灰度图张量 (未归一化)

        Raises:
            ValueError: 不支持的数组形状时

        Note:
            返回的张量未进行 /255 归一化，这在 __getitem__ 中不使用此方法时需要注意
        """
        arr = np.load(path)

        if arr.ndim == 2:
            # 已经是 2D 灰度图
            gray = arr
        elif arr.ndim == 3:
            # 3D 数组，需要判断是 CHW 还是 HWC 格式
            if arr.shape[0] in (1, 3, 4):
                # 假设是 CHW 格式，转换为 HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                # 单通道，直接取出
                gray = arr[..., 0]
            else:
                # 多通道 (RGB/RGBA)，使用 OpenCV 转换为灰度
                gray = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape} in {path}")

        # 添加通道维度，返回 (1, H, W) 形状的张量
        return torch.from_numpy(gray).unsqueeze(0)

    def __getitem__(self, idx):
        """
        获取 CBCT 数据集中的一个训练样本

        重写基类方法以使用 _load_gray_npy 加载灰度图像。

        Args:
            idx (int): 样本索引

        Returns:
            tuple: (crop_hq, crop_lq) - 裁剪并增强后的 HQ/LQ 图像对
        """
        iidx = idx
        idx %= len(self.file_list_hq)

        # 处理预加载的裁剪参数
        if self.epoch_dict is not None:
            if len(self.epoch_dict[iidx]) == 1:
                crop_h, crop_w, random_flip, random_rotate = self.epoch_dict[iidx][0]
            else:
                aa = 0 if iidx % 2 == 0 else 1
                crop_h, crop_w, random_flip, random_rotate = self.epoch_dict[iidx][aa]

        file_hq = self.file_list_hq[idx]
        file_lq = self.file_list_lq[idx] if self.file_list_lq is not None else None

        # 使用专门的灰度加载方法 (注意: 这里未除以 255，与基类不同)
        img_hq = self._load_gray_npy(file_hq)
        img_lq = self._load_gray_npy(file_lq) if file_lq is not None else torch.clone(img_hq)

        _, lq_h, lq_w = img_lq.size()

        # 可选的噪声添加
        if hasattr(self, 'sigma'):
            img_lq = make_some_noise(img_lq, self.sigma)

        # 随机裁剪
        if self.epoch_dict is None:
            crop_h = torch.randint(0, lq_h - self.crop_size, (1,)).item()
            crop_w = torch.randint(0, lq_w - self.crop_size, (1,)).item()

        crop_lq = TF.crop(img_lq, crop_h, crop_w, self.crop_size, self.crop_size)
        crop_hq = TF.crop(img_hq, crop_h * self.scale, crop_w * self.scale,
                          self.crop_size * self.scale, self.crop_size * self.scale)

        # 数据增强
        if self.epoch_dict is None:
            random_flip = torch.randint(0, 2, (1,)).item()
            random_rotate = torch.randint(0, 4, (1,)).item()

        crop_hq, crop_lq = (TF.hflip(crop_hq), TF.hflip(crop_lq)) if random_flip else (crop_hq, crop_lq)
        crop_hq, crop_lq = TF.rotate(crop_hq, angle=90 * random_rotate), TF.rotate(crop_lq, angle=90 * random_rotate)

        return crop_hq, crop_lq

    def __len__(self):
        """
        返回数据集长度

        Note:
            与其他数据集不同，CBCT 数据集不使用乘数扩展长度，
            返回实际的配对文件数量
        """
        return len(self.file_list_hq)
