"""
数字岩心数据集
支持连续灰度数据和离散标签数据
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Sequence
import os

from am_loader import read_am_file
from load_mrccm import load_mat_volume


class DigitalRockDataset(Dataset):
    """
    数字岩心数据集
    将体素数据转换为 (坐标, 密度值) 对
    支持：
    - 连续灰度数据 (0-255)
    - 离散标签数据 (如分割后的多相数据)
    """
    def __init__(
        self,
        voxel_data: np.ndarray,
        n_samples_per_epoch: int = 1000000,
        normalize: bool = True,
        is_label_data: bool = False,
        sample_weights: Optional[np.ndarray] = None,
        classification: bool = False,
    ):
        """
        Args:
            voxel_data: (D, H, W) 体素数据
            n_samples_per_epoch: 每个epoch采样的点数
            normalize: 是否归一化密度值到 [0, 1]
            is_label_data: 是否为标签数据（自动检测：如果唯一值<20则认为是标签）
            sample_weights: (D, H, W) 每个体素的采样/损失权重
        """
        self.voxel_data = voxel_data.astype(np.int64 if classification else np.float32)
        self.n_samples = n_samples_per_epoch
        self.shape = voxel_data.shape

        # 自动检测是否为标签数据
        unique_values = np.unique(voxel_data)
        self.classification = classification
        self.is_label_data = (is_label_data or len(unique_values) < 20) and not classification
        self.unique_labels = unique_values

        if self.classification:
            print(f"分类标签数据，唯一值: {unique_values}")
        elif self.is_label_data:
            # 标签数据：归一化到 [0, 1] 基于最大标签值
            max_label = self.voxel_data.max()
            if max_label > 0:
                self.voxel_data = self.voxel_data / max_label
            print(f"检测到标签数据，唯一值: {unique_values}")
        elif normalize and self.voxel_data.max() > 1.0:
            # 灰度数据：归一化到 [0, 1]
            self.voxel_data = self.voxel_data / 255.0

        self.sample_weights = None
        if sample_weights is not None:
            if sample_weights.shape != self.shape:
                raise ValueError(
                    f"sample_weights shape {sample_weights.shape} does not match voxel_data shape {self.shape}"
                )
            weights = sample_weights.astype(np.float32)
            weights = np.clip(weights, 1e-6, None)
            self.sample_weights = weights
            print(
                "结构权重统计: "
                f"min={self.sample_weights.min():.3f}, "
                f"max={self.sample_weights.max():.3f}, "
                f"mean={self.sample_weights.mean():.3f}"
            )

        print(f"数据集形状: {self.shape}")
        print(f"归一化后值范围: [{self.voxel_data.min():.3f}, {self.voxel_data.max():.3f}]")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 随机采样坐标
        z = np.random.randint(0, self.shape[0])
        y = np.random.randint(0, self.shape[1])
        x = np.random.randint(0, self.shape[2])

        # 归一化坐标到 [0, 1]
        coords = np.array([
            x / (self.shape[2] - 1),
            y / (self.shape[1] - 1),
            z / (self.shape[0] - 1),
        ], dtype=np.float32)

        # 获取密度值
        density = self.voxel_data[z, y, x]

        if self.sample_weights is None:
            weight = 1.0
        else:
            weight = self.sample_weights[z, y, x]

        return (
            torch.from_numpy(coords),
            torch.tensor([density], dtype=torch.long if self.classification else torch.float32),
            torch.tensor([weight], dtype=torch.float32),
        )


class DigitalRockDatasetFull(Dataset):
    """
    全体素数据集（用于验证）
    """
    def __init__(self, voxel_data: np.ndarray, normalize: bool = True):
        self.voxel_data = voxel_data.astype(np.float32)
        self.shape = voxel_data.shape

        if normalize and self.voxel_data.max() > 1.0:
            self.voxel_data = self.voxel_data / 255.0

        # 预计算所有坐标
        D, H, W = self.shape
        z = np.linspace(0, 1, D)
        y = np.linspace(0, 1, H)
        x = np.linspace(0, 1, W)

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        self.coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
        self.densities = self.voxel_data.flatten()

    def __len__(self):
        return len(self.densities)

    def __getitem__(self, idx):
        return torch.from_numpy(self.coords[idx]), torch.tensor([self.densities[idx]], dtype=torch.float32)



def _normalize_crop_origin(
    shape: Sequence[int], crop_size: Optional[int], crop_origin: Optional[Sequence[int]]
) -> Optional[tuple[int, int, int]]:
    if crop_size is None:
        if crop_origin is not None:
            raise ValueError("crop_origin requires crop_size")
        return None
    if crop_size <= 0 or crop_size > min(shape):
        raise ValueError(f"crop_size={crop_size} is invalid for shape {tuple(shape)}")
    if crop_origin is None:
        return tuple((int(dim) - crop_size) // 2 for dim in shape)
    if len(crop_origin) != 3:
        raise ValueError("crop_origin must contain D H W")
    origin = tuple(int(value) for value in crop_origin)
    if any(start < 0 or start + crop_size > int(dim) for start, dim in zip(origin, shape)):
        raise ValueError(
            f"crop_origin={origin}, crop_size={crop_size} exceeds shape {tuple(shape)}"
        )
    return origin


def _crop_slices(origin: Sequence[int], crop_size: int) -> tuple[slice, slice, slice]:
    return tuple(slice(start, start + crop_size) for start in origin)


def _infer_raw_shape(path: str, raw_shape: Optional[Sequence[int]]) -> tuple[int, int, int]:
    if raw_shape is not None:
        if len(raw_shape) != 3 or any(int(dim) <= 0 for dim in raw_shape):
            raise ValueError("raw_shape must contain three positive dimensions")
        shape = tuple(int(dim) for dim in raw_shape)
    else:
        voxel_count = os.path.getsize(path)
        side = round(voxel_count ** (1.0 / 3.0))
        if side ** 3 != voxel_count:
            raise ValueError("Cannot infer RAW shape; pass --raw_shape D H W")
        shape = (side, side, side)
    expected_bytes = int(np.prod(shape))
    if expected_bytes != os.path.getsize(path):
        raise ValueError(
            f"RAW shape {shape} expects {expected_bytes} uint8 bytes, "
            f"but file has {os.path.getsize(path)} bytes"
        )
    return shape


def load_digital_rock(
    path: str,
    downsample: int = 1,
    crop_size: int = None,
    crop_origin: Optional[Sequence[int]] = None,
    raw_shape: Optional[Sequence[int]] = None,
    nc_variable: str = "data",
) -> np.ndarray:
    """
    加载数字岩心数据

    Args:
        path: AM文件路径或npy文件路径
        downsample: 下采样因子
        crop_size: 立方体裁剪尺寸
        crop_origin: 裁剪起点，顺序为 D H W；不指定时使用中心裁剪
        raw_shape: RAW 文件尺寸，顺序为 D H W；立方体 uint8 文件可自动推断
        nc_variable: NetCDF/HDF5 中的体数据变量名

    Returns:
        voxel_data: (D, H, W) 体素数据
    """
    ext = os.path.splitext(path)[1].lower()

    crop_applied = False
    if ext == '.npy':
        voxel_data = np.load(path)
    elif ext == '.am':
        voxel_data = read_am_file(path)
    elif ext == '.mat':
        voxel_data = load_mat_volume(
            path,
            downsample=1,
            crop_size=crop_size,
            crop_origin=crop_origin,
        )
        crop_applied = crop_size is not None
        if crop_applied:
            origin = tuple(crop_origin) if crop_origin is not None else None
    elif ext == '.raw':
        shape = _infer_raw_shape(path, raw_shape)
        origin = _normalize_crop_origin(shape, crop_size, crop_origin)
        raw_data = np.memmap(path, dtype=np.uint8, mode='r', shape=shape, order='C')
        voxel_data = np.asarray(raw_data[_crop_slices(origin, crop_size)] if origin else raw_data)
        crop_applied = origin is not None
    elif ext == '.nc':
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Loading NetCDF4 files requires h5py") from exc
        with h5py.File(path, 'r') as handle:
            if nc_variable not in handle:
                raise KeyError(
                    f"NetCDF variable '{nc_variable}' not found; available: {list(handle.keys())}"
                )
            dataset = handle[nc_variable]
            if dataset.ndim != 3:
                raise ValueError(f"NetCDF variable must be 3D, got shape {dataset.shape}")
            origin = _normalize_crop_origin(dataset.shape, crop_size, crop_origin)
            voxel_data = dataset[_crop_slices(origin, crop_size)] if origin else dataset[...]
            crop_applied = origin is not None
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    print(f"原始数据形状: {voxel_data.shape}")
    print(f"原始数据大小: {voxel_data.nbytes / 1024 / 1024:.2f} MB")

    # 裁剪（用于测试或减少计算量）
    if crop_size is not None and not crop_applied:
        D, H, W = voxel_data.shape
        origin = _normalize_crop_origin((D, H, W), crop_size, crop_origin)
        voxel_data = voxel_data[_crop_slices(origin, crop_size)]
        print(f"从 {origin} 裁剪到 {crop_size}^3 后形状: {voxel_data.shape}")
    elif crop_applied:
        print(f"从 {origin} 裁剪到 {crop_size}^3 后形状: {voxel_data.shape}")

    if downsample > 1:
        voxel_data = voxel_data[::downsample, ::downsample, ::downsample]
        print(f"下采样 {downsample}x 后形状: {voxel_data.shape}")

    # 打印数据统计
    unique_values = np.unique(voxel_data)
    print(f"唯一值数量: {len(unique_values)}")
    if len(unique_values) < 20:
        print(f"唯一值: {unique_values}")

    return voxel_data



def create_low_resolution(voxel_data: np.ndarray, factor: int) -> np.ndarray:
    """
    创建低分辨率版本（用于超分辨率实验）

    Args:
        voxel_data: 原始体素数据
        factor: 下采样因子

    Returns:
        low_res: 低分辨率体素数据
    """
    return voxel_data[::factor, ::factor, ::factor]
