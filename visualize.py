"""
可视化工具 - 论文绘图
支持生成 Fuel / Energy & Fuels 期刊质量的图表
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
import os
import json

# 设置论文级别的图表样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_slices(
    volume: np.ndarray,
    title: str = '',
    slice_indices: Optional[Tuple[int, int, int]] = None,
    save_path: Optional[str] = None,
    cmap: str = 'gray',
):
    """
    绘制三个正交切片
    
    Args:
        volume: (D, H, W) 体素数据
        title: 标题
        slice_indices: (z, y, x) 切片位置，None则取中间
        save_path: 保存路径
        cmap: colormap
    """
    D, H, W = volume.shape
    
    if slice_indices is None:
        slice_indices = (D // 2, H // 2, W // 2)
    
    z_idx, y_idx, x_idx = slice_indices
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY 平面 (Z slice)
    axes[0].imshow(volume[z_idx, :, :], cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(f'XY plane (z={z_idx})')
    axes[0].axis('off')
    
    # XZ 平面 (Y slice)
    axes[1].imshow(volume[:, y_idx, :], cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f'XZ plane (y={y_idx})')
    axes[1].axis('off')
    
    # YZ 平面 (X slice)
    axes[2].imshow(volume[:, :, x_idx], cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f'YZ plane (x={x_idx})')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    save_path: Optional[str] = None,
):
    """
    对比可视化
    """
    if slice_idx is None:
        slice_idx = gt.shape[axis] // 2
    
    if axis == 0:
        gt_slice = gt[slice_idx]
        pred_slice = pred[slice_idx]
    elif axis == 1:
        gt_slice = gt[:, slice_idx]
        pred_slice = pred[:, slice_idx]
    else:
        gt_slice = gt[:, :, slice_idx]
        pred_slice = pred[:, :, slice_idx]
    
    diff = np.abs(gt_slice - pred_slice)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    # 差异直方图
    axes[3].hist(diff.flatten(), bins=50, color='blue', alpha=0.7)
    axes[3].set_title('Error Distribution')
    axes[3].set_xlabel('Absolute Error')
    axes[3].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_training_curves(
    train_losses: list,
    val_metrics: list,
    save_path: Optional[str] = None,
):
    """
    绘制训练曲线
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 训练损失
    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    
    # PSNR
    if val_metrics:
        psnrs = [m['psnr'] for m in val_metrics]
        epochs = list(range(10, 10 * len(psnrs) + 1, 10))  # 假设每10个epoch验证一次
        axes[1].plot(epochs, psnrs, 'o-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Validation PSNR')
    
    # SSIM
    if val_metrics:
        ssims = [m['ssim'] for m in val_metrics]
        axes[2].plot(epochs, ssims, 'o-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('SSIM')
        axes[2].set_title('Validation SSIM')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_paper_figure(
    gt: np.ndarray,
    pred: np.ndarray,
    bicubic: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """
    创建论文质量的对比图
    """
    if slice_idx is None:
        slice_idx = gt.shape[0] // 2
    
    gt_slice = gt[slice_idx]
    pred_slice = pred[slice_idx]
    
    n_cols = 3 if bicubic is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    if bicubic is not None:
        bicubic_slice = bicubic[slice_idx]
        axes[1].imshow(bicubic_slice, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Bicubic', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('INR (Ours)', fontsize=12)
        axes[2].axis('off')
        
        # 差异图
        diff = np.abs(gt_slice - pred_slice)
        axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[3].set_title('Error Map', fontsize=12)
        axes[3].axis('off')
    else:
        axes[1].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('INR (Ours)', fontsize=12)
        axes[1].axis('off')
        
        diff = np.abs(gt_slice - pred_slice)
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[2].set_title('Error Map', fontsize=12)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_compression_comparison(
    results: Dict,
    save_path: Optional[str] = None,
):
    """
    绘制压缩实验结果对比图
    
    Args:
        results: 压缩实验结果字典 {'small': {...}, 'medium': {...}, 'large': {...}}
    """
    configs = list(results.keys())
    psnrs = [results[c]['psnr'] for c in configs]
    ssims = [results[c]['ssim'] for c in configs]
    compression_ratios = [results[c]['compression_ratio'] for c in configs]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # PSNR
    bars1 = axes[0].bar(configs, psnrs, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('(a) Reconstruction Quality (PSNR)')
    for bar, val in zip(bars1, psnrs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=11)
    
    # SSIM
    bars2 = axes[1].bar(configs, ssims, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Structural Similarity (SSIM)')
    axes[1].set_ylim(0, 1.05)
    for bar, val in zip(bars2, ssims):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Compression Ratio
    bars3 = axes[2].bar(configs, compression_ratios, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Compression Ratio')
    axes[2].set_title('(c) Compression Ratio')
    for bar, val in zip(bars3, compression_ratios):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{val:.1f}x', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_sr_comparison(
    results: Dict,
    save_path: Optional[str] = None,
):
    """
    绘制超分辨率实验结果对比图
    
    Args:
        results: SR结果字典 {'2x': {'inr': {...}, 'bicubic': {...}}, '4x': {...}}
    """
    factors = list(results.keys())
    
    inr_psnrs = [results[f]['inr']['psnr'] for f in factors]
    bicubic_psnrs = [results[f]['bicubic']['psnr'] for f in factors]
    inr_ssims = [results[f]['inr']['ssim'] for f in factors]
    bicubic_ssims = [results[f]['bicubic']['ssim'] for f in factors]
    
    x = np.arange(len(factors))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # PSNR comparison
    bars1 = axes[0].bar(x - width/2, bicubic_psnrs, width, label='Bicubic', 
                        color='#95a5a6', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, inr_psnrs, width, label='INR (Ours)', 
                        color='#e74c3c', edgecolor='black')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('(a) Super-Resolution PSNR')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(factors)
    axes[0].set_xlabel('Upscaling Factor')
    axes[0].legend()
    
    # SSIM comparison
    bars3 = axes[1].bar(x - width/2, bicubic_ssims, width, label='Bicubic', 
                        color='#95a5a6', edgecolor='black')
    bars4 = axes[1].bar(x + width/2, inr_ssims, width, label='INR (Ours)', 
                        color='#e74c3c', edgecolor='black')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Super-Resolution SSIM')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(factors)
    axes[1].set_xlabel('Upscaling Factor')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_porosity_comparison(
    gt_porosities: List[float],
    pred_porosities: List[float],
    labels: List[str],
    save_path: Optional[str] = None,
):
    """
    绘制孔隙度对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 柱状图
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0].bar(x - width/2, gt_porosities, width, label='Ground Truth', 
                color='#3498db', edgecolor='black')
    axes[0].bar(x + width/2, pred_porosities, width, label='Predicted', 
                color='#e74c3c', edgecolor='black')
    axes[0].set_ylabel('Porosity')
    axes[0].set_title('(a) Porosity Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    
    # 散点图（相关性）
    max_val = max(max(gt_porosities), max(pred_porosities)) * 1.1
    axes[1].scatter(gt_porosities, pred_porosities, s=100, c='#e74c3c', 
                    edgecolor='black', linewidth=1.5, zorder=3)
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('Ground Truth Porosity')
    axes[1].set_ylabel('Predicted Porosity')
    axes[1].set_title('(b) Porosity Correlation')
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_pore_structure(
    gt: np.ndarray,
    pred: np.ndarray,
    slice_idx: Optional[int] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    绘制孔隙结构对比图（二值化）
    """
    if slice_idx is None:
        slice_idx = gt.shape[0] // 2
    
    gt_slice = gt[slice_idx]
    pred_slice = pred[slice_idx]
    
    # 二值化
    gt_binary = (gt_slice < threshold).astype(float)
    pred_binary = (pred_slice < threshold).astype(float)
    
    # 差异
    diff = np.abs(gt_binary - pred_binary)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(gt_binary, cmap='binary', vmin=0, vmax=1)
    axes[0].set_title('(a) Ground Truth Pores')
    axes[0].axis('off')
    
    axes[1].imshow(pred_binary, cmap='binary', vmin=0, vmax=1)
    axes[1].set_title('(b) Predicted Pores')
    axes[1].axis('off')
    
    # Overlay: 绿色=正确, 红色=错误
    overlay = np.zeros((*gt_slice.shape, 3))
    overlay[gt_binary == pred_binary, 1] = gt_binary[gt_binary == pred_binary]  # 绿色
    overlay[gt_binary != pred_binary, 0] = 1  # 红色
    axes[2].imshow(overlay)
    axes[2].set_title('(c) Overlay (Green=Match, Red=Error)')
    axes[2].axis('off')
    
    axes[3].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title('(d) Difference Map')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_compression_vs_quality(
    compression_ratios: List[float],
    psnrs: List[float],
    ssims: List[float],
    labels: List[str],
    save_path: Optional[str] = None,
):
    """
    绘制压缩率 vs 质量曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # PSNR vs Compression Ratio
    axes[0].plot(compression_ratios, psnrs, 'o-', color='#e74c3c', 
                 markersize=10, linewidth=2, markeredgecolor='black')
    for i, label in enumerate(labels):
        axes[0].annotate(label, (compression_ratios[i], psnrs[i]), 
                        textcoords="offset points", xytext=(10, 5), fontsize=10)
    axes[0].set_xlabel('Compression Ratio')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('(a) PSNR vs Compression Ratio')
    axes[0].grid(True, alpha=0.3)
    
    # SSIM vs Compression Ratio
    axes[1].plot(compression_ratios, ssims, 'o-', color='#3498db', 
                 markersize=10, linewidth=2, markeredgecolor='black')
    for i, label in enumerate(labels):
        axes[1].annotate(label, (compression_ratios[i], ssims[i]), 
                        textcoords="offset points", xytext=(10, 5), fontsize=10)
    axes[1].set_xlabel('Compression Ratio')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) SSIM vs Compression Ratio')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_sr_visual_comparison(
    gt: np.ndarray,
    bicubic: np.ndarray,
    inr: np.ndarray,
    slice_idx: Optional[int] = None,
    zoom_region: Optional[Tuple[int, int, int, int]] = None,  # (y1, y2, x1, x2)
    save_path: Optional[str] = None,
):
    """
    创建超分辨率视觉对比图（带放大区域）
    """
    if slice_idx is None:
        slice_idx = gt.shape[0] // 2
    
    gt_slice = gt[slice_idx]
    bicubic_slice = bicubic[slice_idx]
    inr_slice = inr[slice_idx]
    
    # 自动选择放大区域（中心区域）
    if zoom_region is None:
        h, w = gt_slice.shape
        size = min(h, w) // 4
        y1, x1 = h // 2 - size // 2, w // 2 - size // 2
        zoom_region = (y1, y1 + size, x1, x1 + size)
    
    y1, y2, x1, x2 = zoom_region
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 上排：完整图像
    axes[0, 0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color='red', linewidth=2))
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bicubic_slice, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Bicubic')
    axes[0, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color='red', linewidth=2))
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inr_slice, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('INR (Ours)')
    axes[0, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color='red', linewidth=2))
    axes[0, 2].axis('off')
    
    # 下排：放大区域
    axes[1, 0].imshow(gt_slice[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('GT (Zoomed)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bicubic_slice[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Bicubic (Zoomed)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(inr_slice[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('INR (Zoomed)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def generate_latex_table(
    results: Dict,
    table_type: str = 'compression',
) -> str:
    """
    生成 LaTeX 表格
    
    Args:
        results: 实验结果
        table_type: 'compression' or 'sr'
    
    Returns:
        LaTeX 表格代码
    """
    if table_type == 'compression':
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Compression experiment results on digital rock data.}
\label{tab:compression}
\begin{tabular}{lcccccc}
\toprule
Config & Size (MB) & Ratio & PSNR (dB) & SSIM & Porosity Error \\
\midrule
"""
        for config, metrics in results.items():
            latex += f"{config.capitalize()} & {metrics.get('model_size_mb', 0):.2f} & "
            latex += f"{metrics.get('compression_ratio', 0):.1f}x & "
            latex += f"{metrics.get('psnr', 0):.2f} & "
            latex += f"{metrics.get('ssim', 0):.4f} & "
            latex += f"{metrics.get('porosity_error', 0):.4f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    else:  # sr
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Super-resolution experiment results.}
\label{tab:sr}
\begin{tabular}{lccccc}
\toprule
Factor & Method & PSNR (dB) & SSIM & Porosity Error \\
\midrule
"""
        for factor, data in results.items():
            latex += f"{factor} & Bicubic & {data['bicubic']['psnr']:.2f} & "
            latex += f"{data['bicubic']['ssim']:.4f} & - \\\\\n"
            latex += f" & INR (Ours) & {data['inr']['psnr']:.2f} & "
            latex += f"{data['inr']['ssim']:.4f} & "
            latex += f"{data['inr'].get('porosity_error', 0):.4f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


if __name__ == '__main__':
    # 测试
    print("Visualize module loaded successfully.")
    print("Available functions:")
    print("  - plot_slices()")
    print("  - plot_comparison()")
    print("  - plot_training_curves()")
    print("  - create_paper_figure()")
    print("  - plot_compression_comparison()")
    print("  - plot_sr_comparison()")
    print("  - plot_porosity_comparison()")
    print("  - plot_pore_structure()")
    print("  - plot_compression_vs_quality()")
    print("  - create_sr_visual_comparison()")
    print("  - generate_latex_table()")
