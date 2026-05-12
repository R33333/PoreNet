"""
评估指标
支持连续灰度数据和离散标签数据
"""

import numpy as np
from typing import Dict, Optional, List
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """计算 PSNR"""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """计算 SSIM (3D)"""
    ssim_values = []
    for i in range(pred.shape[0]):
        s = ssim(pred[i], gt[i], data_range=data_range)
        ssim_values.append(s)
    return np.mean(ssim_values)


def compute_porosity(volume: np.ndarray, threshold: float = 0.5,
                     pore_labels: Optional[List[int]] = None,
                     is_normalized: bool = True,
                     max_label: float = 7.0,
                     dataset_type: str = 'auto') -> float:
    """
    计算孔隙度
    """
    if pore_labels is None:
        if dataset_type == 'mrccm':
            pore_labels = [1, 3]
        elif dataset_type == 'berea':
            pore_labels = [0]
        elif dataset_type == 'auto':
            if is_normalized:
                test_labels = np.unique(np.round(volume * max_label).astype(int))
            else:
                test_labels = np.unique(volume.astype(int))

            if set(test_labels) == {1, 2, 3} or set(test_labels) <= {1, 2, 3}:
                pore_labels = [1, 3]
            elif 0 in test_labels:
                pore_labels = [0]
            else:
                pore_labels = [int(test_labels.min())]

    if pore_labels is not None:
        if is_normalized:
            volume_labels = np.round(volume * max_label).astype(int)
        else:
            volume_labels = volume.astype(int)

        pore_voxels = sum((volume_labels == label).sum() for label in pore_labels)
    else:
        pore_voxels = (volume < threshold).sum()

    total_voxels = volume.size
    porosity = pore_voxels / total_voxels

    return porosity


def detect_dataset_type(volume: np.ndarray) -> str:
    """
    自动检测数据集类型
    """
    unique_vals = np.unique(volume)

    if set(unique_vals) == {1, 2, 3} or set(unique_vals) <= {1, 2, 3}:
        return 'mrccm'

    if 0 in unique_vals and len(unique_vals) <= 10:
        return 'berea'

    if len(unique_vals) > 10:
        return 'grayscale'

    return 'auto'


def compute_label_accuracy(pred: np.ndarray, gt: np.ndarray,
                           max_label: float = 7.0) -> Dict[str, float]:
    """
    计算标签分类准确率（用于分割数据）
    """
    pred_labels = np.round(pred * max_label).astype(int)
    gt_labels = np.round(gt * max_label).astype(int)

    accuracy = (pred_labels == gt_labels).mean()

    unique_labels = np.unique(gt_labels)
    per_class_acc = {}
    for label in unique_labels:
        mask = gt_labels == label
        if mask.sum() > 0:
            per_class_acc[f'acc_label_{label}'] = (pred_labels[mask] == label).mean()

    return {
        'label_accuracy': accuracy,
        **per_class_acc
    }


def _binary_pore_mask(volume: np.ndarray,
                      pore_labels: Optional[List[int]],
                      is_normalized: bool,
                      max_label: float,
                      threshold: float = 0.5) -> np.ndarray:
    if pore_labels is not None:
        if is_normalized:
            labels = np.round(volume * max_label).astype(int)
        else:
            labels = volume.astype(int)
        mask = np.isin(labels, pore_labels)
    else:
        mask = volume < threshold
    return mask.astype(np.uint8)


def compute_connectivity_metrics(pred: np.ndarray,
                                 gt: np.ndarray,
                                 pore_labels: Optional[List[int]] = None,
                                 is_normalized: bool = True,
                                 max_label: float = 7.0) -> Dict[str, float]:
    """
    计算孔隙连通性相关指标。
    当前实现使用 26 邻域连通域统计作为 topology-aware 的轻量代理指标。
    """
    try:
        from scipy import ndimage
    except ImportError:
        return {'connectivity_error': float('nan')}

    structure = np.ones((3, 3, 3), dtype=np.uint8)

    pred_mask = _binary_pore_mask(pred, pore_labels, is_normalized, max_label)
    gt_mask = _binary_pore_mask(gt, pore_labels, is_normalized, max_label)

    pred_labeled, pred_components = ndimage.label(pred_mask, structure=structure)
    gt_labeled, gt_components = ndimage.label(gt_mask, structure=structure)

    pred_sizes = np.bincount(pred_labeled.ravel())[1:]
    gt_sizes = np.bincount(gt_labeled.ravel())[1:]

    pred_largest = float(pred_sizes.max()) if pred_sizes.size > 0 else 0.0
    gt_largest = float(gt_sizes.max()) if gt_sizes.size > 0 else 0.0

    pred_pore_voxels = float(pred_mask.sum())
    gt_pore_voxels = float(gt_mask.sum())

    pred_largest_ratio = pred_largest / pred_pore_voxels if pred_pore_voxels > 0 else 0.0
    gt_largest_ratio = gt_largest / gt_pore_voxels if gt_pore_voxels > 0 else 0.0

    connectivity_error = abs(pred_largest_ratio - gt_largest_ratio)
    component_count_error = abs(float(pred_components) - float(gt_components))

    return {
        'connected_components_pred': float(pred_components),
        'connected_components_gt': float(gt_components),
        'largest_component_ratio_pred': pred_largest_ratio,
        'largest_component_ratio_gt': gt_largest_ratio,
        'connectivity_error': connectivity_error,
        'component_count_error': component_count_error,
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray,
                    is_label_data: bool = False,
                    max_label: float = 7.0,
                    pore_labels: Optional[List[int]] = None) -> Dict[str, float]:
    """
    计算所有指标
    """
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)

    metrics = {
        'psnr': compute_psnr(pred, gt),
        'ssim': compute_ssim(pred, gt),
        'mse': float(np.mean((pred - gt) ** 2)),
        'mae': float(np.mean(np.abs(pred - gt))),
    }

    if is_label_data:
        label_metrics = compute_label_accuracy(pred, gt, max_label)
        metrics.update(label_metrics)
        connectivity_metrics = compute_connectivity_metrics(
            pred,
            gt,
            pore_labels=pore_labels,
            is_normalized=True,
            max_label=max_label,
        )
        metrics.update(connectivity_metrics)

    return metrics


def compute_pore_size_distribution(volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    计算孔隙尺寸分布（简化版本）
    """
    try:
        from scipy import ndimage

        binary = (volume < threshold).astype(np.uint8)
        labeled, num_features = ndimage.label(binary)
        pore_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

        return np.array(pore_sizes)
    except ImportError:
        print("警告: scipy 未安装，无法计算孔隙尺寸分布")
        return np.array([])


def compare_pore_size_distribution(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    比较孔隙尺寸分布
    """
    pred_psd = compute_pore_size_distribution(pred, threshold)
    gt_psd = compute_pore_size_distribution(gt, threshold)

    if len(pred_psd) == 0 or len(gt_psd) == 0:
        return {'psd_error': float('nan')}

    max_size = max(pred_psd.max(), gt_psd.max())
    bins = np.linspace(0, max_size, 50)

    pred_hist, _ = np.histogram(pred_psd, bins=bins, density=True)
    gt_hist, _ = np.histogram(gt_psd, bins=bins, density=True)

    psd_error = np.sum(np.abs(pred_hist - gt_hist)) * (bins[1] - bins[0])

    return {
        'psd_error': psd_error,
        'num_pores_pred': len(pred_psd),
        'num_pores_gt': len(gt_psd),
    }
