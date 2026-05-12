"""
超分辨率实验
支持连续灰度数据和离散标签数据
"""

import torch
import numpy as np
import os
import argparse
import json
from tqdm import tqdm

from model import DigitalRockINR, DigitalRockINRTinyCuda
from dataset import load_digital_rock, create_low_resolution, DigitalRockDataset
from metrics import compute_metrics, compute_porosity
from torch.utils.data import DataLoader
import torch.optim as optim


def super_resolution_experiment(
    data_path: str,
    output_dir: str,
    sr_factors: list = [2, 4],
    epochs: int = 100,
    use_tcnn: bool = False,
    downsample: int = 1,
    crop_size: int = None,
):
    """
    超分辨率实验
    
    流程:
    1. 加载高分辨率数据 (GT)
    2. 创建低分辨率版本
    3. 在低分辨率数据上训练 INR
    4. 用 INR 查询高分辨率
    5. 与 GT 比较
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载高分辨率数据
    print(f"Loading high-resolution data: {data_path}")
    hr_data = load_digital_rock(data_path, downsample=downsample, crop_size=crop_size)
    print(f"HR shape: {hr_data.shape}")
    
    # 检测是否为标签数据
    unique_values = np.unique(hr_data)
    is_label_data = len(unique_values) < 20
    max_label = float(hr_data.max()) if hr_data.max() > 0 else 1.0
    print(f"Is label data: {is_label_data}, Max label: {max_label}")
    
    # 归一化
    hr_data = hr_data.astype(np.float32)
    if is_label_data:
        hr_data = hr_data / max_label
    elif hr_data.max() > 1.0:
        hr_data = hr_data / 255.0
    
    results = {}
    
    for factor in sr_factors:
        print(f"\n{'='*50}")
        print(f"Super-resolution factor: {factor}x")
        print(f"{'='*50}")
        
        # 创建低分辨率版本
        lr_data = create_low_resolution(hr_data, factor)
        print(f"LR shape: {lr_data.shape}")
        
        # 创建数据集（在低分辨率上训练）
        n_samples = min(1000000, lr_data.size)
        dataset = DigitalRockDataset(lr_data, n_samples_per_epoch=n_samples)
        train_loader = DataLoader(dataset, batch_size=65536, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        
        # 创建模型
        if use_tcnn:
            model = DigitalRockINRTinyCuda(
                finest_resolution=max(hr_data.shape),  # 目标是HR分辨率
            ).to(device)
        else:
            model = DigitalRockINR(
                encoding_type='hash',
                finest_resolution=max(hr_data.shape),
                n_levels=16,
                log2_hashmap_size=19,
            ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            for coords, densities in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                coords = coords.to(device)
                densities = densities.to(device)
                
                pred = model(coords)
                loss = torch.nn.functional.mse_loss(pred, densities)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
        
        # 超分辨率重建
        print("Generating super-resolved volume...")
        model.eval()
        sr_data = model.query_volume(hr_data.shape, device).cpu().numpy()
        
        # 计算指标
        metrics = compute_metrics(sr_data, hr_data, is_label_data, max_label)
        
        # 孔隙度
        pore_labels = [0] if is_label_data else None
        metrics['porosity_hr'] = compute_porosity(hr_data, pore_labels=pore_labels, max_label=max_label)
        metrics['porosity_sr'] = compute_porosity(sr_data, pore_labels=pore_labels, max_label=max_label)
        metrics['porosity_error'] = abs(metrics['porosity_hr'] - metrics['porosity_sr'])
        
        # Bicubic 基线
        from scipy.ndimage import zoom
        bicubic_sr = zoom(lr_data, factor, order=3)
        # 确保形状匹配
        bicubic_sr = bicubic_sr[:hr_data.shape[0], :hr_data.shape[1], :hr_data.shape[2]]
        bicubic_sr = np.clip(bicubic_sr, 0, 1)
        bicubic_metrics = compute_metrics(bicubic_sr, hr_data, is_label_data, max_label)
        bicubic_metrics['porosity'] = compute_porosity(bicubic_sr, pore_labels=pore_labels, max_label=max_label)
        
        print(f"\nResults for {factor}x SR:")
        print(f"  INR     - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        print(f"  Bicubic - PSNR: {bicubic_metrics['psnr']:.2f} dB, SSIM: {bicubic_metrics['ssim']:.4f}")
        if is_label_data and 'label_accuracy' in metrics:
            print(f"  INR Label Accuracy: {metrics['label_accuracy']:.4f}")
        print(f"  Porosity error: {metrics['porosity_error']:.4f}")
        
        results[f'{factor}x'] = {
            'inr': metrics,
            'bicubic': bicubic_metrics,
        }
        
        # 保存结果 - 只保存切片（完整数据太大）
        np.save(os.path.join(output_dir, f'sr_{factor}x_slice.npy'), sr_data[sr_data.shape[0]//2])
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_{factor}x.pth'))
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
    
    # 保存所有结果
    with open(os.path.join(output_dir, 'sr_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='output/super_resolution')
    parser.add_argument('--factors', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--use_tcnn', action='store_true')
    
    args = parser.parse_args()
    
    super_resolution_experiment(
        args.data,
        args.output,
        sr_factors=args.factors,
        epochs=args.epochs,
        use_tcnn=args.use_tcnn,
        downsample=args.downsample,
        crop_size=args.crop_size,
    )
