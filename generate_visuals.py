"""
从已完成的实验生成可视化结果
用于已训练完成但没有可视化图像的实验
"""

import argparse
import os
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DigitalRockINR
from metrics import compute_metrics, compute_porosity, detect_dataset_type


def generate_visuals(
    data_path: str,
    model_path: str,
    output_dir: str,
    n_levels: int = 16,
    log2_hashmap_size: int = 20,
    hidden_dim: int = 64,
):
    """
    从已保存的模型生成可视化结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading data: {data_path}")
    voxel_data = np.load(data_path)
    D, H, W = voxel_data.shape
    print(f"Volume shape: {D} x {H} x {W}")
    
    # 检测数据类型
    unique_values = np.unique(voxel_data)
    is_label_data = len(unique_values) < 20
    max_label = float(voxel_data.max()) if voxel_data.max() > 0 else 1.0
    dataset_type = detect_dataset_type(voxel_data)
    print(f"Dataset type: {dataset_type}, Is label data: {is_label_data}")
    
    # 归一化
    voxel_data = voxel_data.astype(np.float32)
    data_min = float(voxel_data.min())
    data_max = float(voxel_data.max())
    
    if is_label_data:
        voxel_norm = voxel_data / max_label
        print(f"Label data normalized by max_label={max_label}")
    elif data_max > 1.0:
        voxel_norm = (voxel_data - data_min) / (data_max - data_min)
        print(f"Grayscale data normalized: [{data_min:.1f}, {data_max:.1f}] -> [0, 1]")
    else:
        voxel_norm = voxel_data
    
    # 创建模型
    model = DigitalRockINR(
        encoding_type='hash',
        n_levels=n_levels,
        log2_hashmap_size=log2_hashmap_size,
        hidden_dim=hidden_dim,
    ).to(device)
    
    # 加载权重
    print(f"Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 计算模型大小和压缩率
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    original_size = voxel_data.nbytes
    compression_ratio = original_size / model_size
    print(f"Model size: {model_size/1024/1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成重建
    print("\nGenerating reconstruction...")
    torch.cuda.empty_cache()
    with torch.no_grad():
        chunk_size = 32 if D > 500 else 64
        pred_volume = model.query_volume((D, H, W), device, chunk_size=chunk_size)
        pred_np = pred_volume.cpu().numpy()
        del pred_volume
        torch.cuda.empty_cache()
    
    gt_np = voxel_norm
    
    # 计算指标
    metrics = compute_metrics(pred_np, gt_np, is_label_data, max_label)
    print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    if 'label_accuracy' in metrics:
        print(f"Label Accuracy: {metrics['label_accuracy']:.4f}")
    
    # 孔隙度
    porosity_gt = compute_porosity(gt_np, pore_labels=None, max_label=max_label, dataset_type=dataset_type)
    porosity_pred = compute_porosity(pred_np, pore_labels=None, max_label=max_label, dataset_type=dataset_type)
    print(f"Porosity - GT: {porosity_gt:.4f}, Pred: {porosity_pred:.4f}")
    
    # 保存重建体积
    np.save(os.path.join(output_dir, 'reconstruction.npy'), pred_np)
    print(f"Saved: reconstruction.npy")
    
    # ========== 生成可视化 ==========
    print("\nGenerating visualizations...")
    
    # 1. 重建对比图
    slice_idx = D // 2
    gt_slice = gt_np[slice_idx]
    pred_slice = pred_np[slice_idx]
    diff_slice = np.abs(gt_slice - pred_slice)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Reconstruction', fontsize=12)
    axes[1].axis('off')
    
    im2 = axes[2].imshow(diff_slice, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title('Error Map', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    axes[3].hist(diff_slice.flatten(), bins=50, color='steelblue', alpha=0.8, edgecolor='black')
    axes[3].set_xlabel('Absolute Error')
    axes[3].set_ylabel('Count')
    axes[3].set_title('Error Distribution')
    axes[3].set_xlim(0, 0.5)
    
    plt.suptitle(f'Compression Results (PSNR: {metrics["psnr"]:.2f} dB, Ratio: {compression_ratio:.1f}x)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: comparison.png")
    
    # 2. 三正交切片视图
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(gt_np[D//2, :, :], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'GT - XY plane (z={D//2})')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gt_np[:, H//2, :], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'GT - XZ plane (y={H//2})')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_np[:, :, W//2], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'GT - YZ plane (x={W//2})')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(pred_np[D//2, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Pred - XY plane (z={D//2})')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(pred_np[:, H//2, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Pred - XZ plane (y={H//2})')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(pred_np[:, :, W//2], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Pred - YZ plane (x={W//2})')
    axes[1, 2].axis('off')
    
    plt.suptitle('Orthogonal Slices Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slices.png'), dpi=150)
    plt.close()
    print(f"Saved: slices.png")
    
    # 3. 如果是标签数据，生成标签对比图
    if is_label_data:
        gt_labels = np.round(gt_np[slice_idx] * max_label).astype(int)
        pred_labels = np.round(pred_np[slice_idx] * max_label).astype(int)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(gt_labels, cmap='tab10', vmin=0, vmax=int(max_label))
        axes[0].set_title('GT Labels')
        axes[0].axis('off')
        
        axes[1].imshow(pred_labels, cmap='tab10', vmin=0, vmax=int(max_label))
        axes[1].set_title('Pred Labels')
        axes[1].axis('off')
        
        label_diff = (gt_labels != pred_labels).astype(float)
        axes[2].imshow(label_diff, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Label Errors ({label_diff.mean()*100:.1f}%)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_comparison.png'), dpi=150)
        plt.close()
        print(f"Saved: label_comparison.png")
    
    # 4. 多切片展示（论文用）
    n_slices = 5
    slice_positions = np.linspace(D//4, 3*D//4, n_slices).astype(int)
    
    fig, axes = plt.subplots(3, n_slices, figsize=(3*n_slices, 9))
    
    for i, s in enumerate(slice_positions):
        axes[0, i].imshow(gt_np[s], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'z={s}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(pred_np[s], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        diff = np.abs(gt_np[s] - pred_np[s])
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Ground Truth', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=12)
    axes[2, 0].set_ylabel('Error', fontsize=12)
    
    plt.suptitle('Multi-Slice Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_slice.png'), dpi=150)
    plt.close()
    print(f"Saved: multi_slice.png")
    
    # 保存指标
    results = {
        'data_shape': [D, H, W],
        'original_size_mb': float(original_size / 1024 / 1024),
        'model_size_mb': float(model_size / 1024 / 1024),
        'compression_ratio': float(compression_ratio),
        'psnr': float(metrics['psnr']),
        'ssim': float(metrics['ssim']),
        'porosity_gt': float(porosity_gt),
        'porosity_pred': float(porosity_pred),
        'porosity_error': float(abs(porosity_gt - porosity_pred)),
    }
    if 'label_accuracy' in metrics:
        results['label_accuracy'] = float(metrics['label_accuracy'])
    
    with open(os.path.join(output_dir, 'visual_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: visual_metrics.json")
    
    print(f"\n=== Summary ===")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Porosity error: {abs(porosity_gt - porosity_pred):.4f}")
    print(f"\nAll visualizations saved to: {output_dir}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to original data (.npy)')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--n_levels', type=int, default=16, help='Hash encoding levels')
    parser.add_argument('--log2_hashmap_size', type=int, default=20, help='Log2 hashmap size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    
    args = parser.parse_args()
    
    generate_visuals(
        args.data,
        args.model,
        args.output,
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        hidden_dim=args.hidden_dim,
    )


if __name__ == '__main__':
    main()
