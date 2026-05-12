"""
MRCCM 数据集加载工具
用于加载 Multi-Resolution Complex Carbonates Micro-CT Dataset
"""

import os
from typing import Optional

import numpy as np


def _first_3d_dataset(h5_file):
    import h5py

    for key in h5_file.keys():
        if key.startswith('#'):
            continue
        obj = h5_file[key]
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
            return key, obj
    raise ValueError('No 3D dataset found in MAT v7.3 file')


def load_mat_volume(filepath: str, downsample: int = 1, crop_size: Optional[int] = None) -> np.ndarray:
    """
    加载 MRCCM MATLAB v7.3/HDF5 体数据。

    如果提供 crop_size，则直接从 HDF5 数据集中读取中心裁剪块，避免加载完整 HR 体数据。
    """
    import h5py

    with h5py.File(filepath, 'r') as f:
        key, dset = _first_3d_dataset(f)
        print(f"MAT v7.3 key: {key}, shape={dset.shape}, dtype={dset.dtype}")

        if crop_size is not None and crop_size < min(dset.shape):
            D, H, W = dset.shape
            d_start = (D - crop_size) // 2
            h_start = (H - crop_size) // 2
            w_start = (W - crop_size) // 2
            data = dset[
                d_start:d_start + crop_size,
                h_start:h_start + crop_size,
                w_start:w_start + crop_size,
            ]
            print(f"从 MAT 文件中心裁剪到 {crop_size}³ 后形状: {data.shape}")
        else:
            data = dset[:]

    data = np.asarray(data)

    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
        print(f"下采样 {downsample}x 后形状: {data.shape}")

    return data


def load_mat_file(filepath):
    """
    加载 .mat 文件，优先支持 MATLAB v7.3/HDF5。
    """
    try:
        data = load_mat_volume(filepath)
        return data
    except Exception as e:
        print(f"h5py 加载失败: {e}")
        try:
            from scipy.io import loadmat
            mat = loadmat(filepath)
            print(f"文件 keys: {[k for k in mat.keys() if not k.startswith('__')]}")
            for key, val in mat.items():
                if not key.startswith('__'):
                    print(f"  变量 '{key}': shape={val.shape}, dtype={val.dtype}")
                    return val
        except Exception as e2:
            print(f"scipy 加载也失败: {e2}")
            return None


def check_mrccm_data(data_dir='MRCCM'):
    """
    检查 MRCCM 数据集的所有文件
    """
    print("=" * 60)
    print("MRCCM 数据集检查")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return

    files = os.listdir(data_dir)
    print(f"\n找到 {len(files)} 个文件:")
    for f in files:
        filepath = os.path.join(data_dir, f)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {f}: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("逐个加载检查")
    print("=" * 60)

    for f in files:
        if f.endswith('.mat'):
            print(f"\n>>> 加载 {f}")
            filepath = os.path.join(data_dir, f)
            data = load_mat_file(filepath)
            if data is not None:
                print(f"    最终 shape: {data.shape}")
                print(f"    值范围: [{data.min()}, {data.max()}]")
                print(f"    唯一值数量: {len(np.unique(data))}")
                if len(np.unique(data)) <= 10:
                    print(f"    唯一值: {np.unique(data)}")


def load_ils_data(data_dir='MRCCM'):
    """
    加载 Indiana Limestone 数据
    返回: (lr_volume, hr_segmented)
    """
    lr_file = None
    hr_seg_file = None

    for f in os.listdir(data_dir):
        f_lower = f.lower()
        if 'ils' in f_lower or 'indiana' in f_lower:
            if 'lr' in f_lower:
                lr_file = os.path.join(data_dir, f)
            elif 'seg' in f_lower or 'hr' in f_lower:
                hr_seg_file = os.path.join(data_dir, f)

    lr_data = load_mat_file(lr_file) if lr_file else None
    hr_data = load_mat_file(hr_seg_file) if hr_seg_file else None

    return lr_data, hr_data


def load_mec_data(data_dir='MRCCM'):
    """
    加载 Middle Eastern Carbonate 数据
    返回: (lr_volume, hr_segmented)
    """
    lr_file = None
    hr_seg_file = None

    for f in os.listdir(data_dir):
        f_lower = f.lower()
        if 'mec' in f_lower or 'middle' in f_lower:
            if 'lr' in f_lower:
                lr_file = os.path.join(data_dir, f)
            elif 'seg' in f_lower or 'hr' in f_lower:
                hr_seg_file = os.path.join(data_dir, f)

    lr_data = load_mat_file(lr_file) if lr_file else None
    hr_data = load_mat_file(hr_seg_file) if hr_seg_file else None

    return lr_data, hr_data


def convert_to_npy(data_dir='MRCCM', output_dir='data'):
    """
    将 MRCCM 数据转换为 npy 格式，方便后续使用
    """
    os.makedirs(output_dir, exist_ok=True)

    print("转换 Indiana Limestone (ILS)...")
    ils_lr, ils_hr = load_ils_data(data_dir)
    if ils_lr is not None:
        np.save(os.path.join(output_dir, 'ils_lr.npy'), ils_lr)
        print(f"  保存 ils_lr.npy: {ils_lr.shape}")
    if ils_hr is not None:
        np.save(os.path.join(output_dir, 'ils_hr_seg.npy'), ils_hr)
        print(f"  保存 ils_hr_seg.npy: {ils_hr.shape}")

    print("\n转换 Middle Eastern Carbonate (MEC)...")
    mec_lr, mec_hr = load_mec_data(data_dir)
    if mec_lr is not None:
        np.save(os.path.join(output_dir, 'mec_lr.npy'), mec_lr)
        print(f"  保存 mec_lr.npy: {mec_lr.shape}")
    if mec_hr is not None:
        np.save(os.path.join(output_dir, 'mec_hr_seg.npy'), mec_hr)
        print(f"  保存 mec_hr_seg.npy: {mec_hr.shape}")

    print("\n转换完成！")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        data_dir = sys.argv[2] if len(sys.argv) > 2 else 'MRCCM'
        convert_to_npy(data_dir)
    else:
        data_dir = sys.argv[1] if len(sys.argv) > 1 else 'MRCCM'
        check_mrccm_data(data_dir)
