"""
一键运行所有实验
针对数字岩心数据的 INR 压缩和超分辨率实验
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime


def run_compression_experiment(data_path: str, output_dir: str, epochs: int = 100,
                               downsample: int = 1, crop_size: int = None):
    """压缩实验"""
    from dataset import load_digital_rock
    from model import DigitalRockINR
    from metrics import compute_metrics, compute_porosity
    from torch.utils.data import DataLoader
    from dataset import DigitalRockDataset
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"\n{'='*60}")
    print("COMPRESSION EXPERIMENT")
    print(f"{'='*60}")
    
    voxel_data = load_digital_rock(data_path, downsample=downsample, crop_size=crop_size)
    original_size = voxel_data.nbytes
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    
    # 检测是否为标签数据
    unique_values = np.unique(voxel_data)
    is_label_data = len(unique_values) < 20
    max_label = float(voxel_data.max()) if voxel_data.max() > 0 else 1.0
    print(f"Is label data: {is_label_data}, Max label: {max_label}")
    
    # 归一化
    voxel_data = voxel_data.astype(np.float32)
    if is_label_data:
        voxel_data = voxel_data / max_label
    elif voxel_data.max() > 1.0:
        voxel_data = voxel_data / 255.0
    
    # 不同压缩率的实验配置
    # 针对 1200³ 数据优化的参数
    configs = [
        {'n_levels': 8, 'log2_hashmap_size': 17, 'hidden_dim': 64, 'name': 'small'},
        {'n_levels': 12, 'log2_hashmap_size': 19, 'hidden_dim': 64, 'name': 'medium'},
        {'n_levels': 16, 'log2_hashmap_size': 20, 'hidden_dim': 64, 'name': 'large'},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n--- Config: {config['name']} ---")
        
        # 创建模型
        model = DigitalRockINR(
            encoding_type='hash',
            n_levels=config['n_levels'],
            log2_hashmap_size=config['log2_hashmap_size'],
            hidden_dim=config['hidden_dim'],
            finest_resolution=max(voxel_data.shape),
        ).to(device)
        
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compression_ratio = original_size / model_size
        print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        
        # 训练 - 针对大数据优化
        n_samples = min(2000000, voxel_data.size // 2)  # 每 epoch 采样点数
        dataset = DigitalRockDataset(voxel_data, n_samples_per_epoch=n_samples)
        train_loader = DataLoader(dataset, batch_size=65536, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for coords, densities in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                coords, densities = coords.to(device), densities.to(device)
                pred = model(coords)
                loss = torch.nn.functional.mse_loss(pred, densities)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.6f}")
        
        # 评估
        model.eval()
        print("Generating reconstruction...")
        pred_volume = model.query_volume(voxel_data.shape, device).cpu().numpy()
        
        # 计算指标
        metrics = compute_metrics(pred_volume, voxel_data, is_label_data, max_label)
        metrics['compression_ratio'] = compression_ratio
        metrics['model_size_mb'] = model_size / 1024 / 1024
        metrics['original_size_mb'] = original_size / 1024 / 1024
        
        # 孔隙度
        pore_labels = [0] if is_label_data else None
        metrics['porosity_gt'] = compute_porosity(voxel_data, pore_labels=pore_labels,
                                                  max_label=max_label)
        metrics['porosity_pred'] = compute_porosity(pred_volume, pore_labels=pore_labels,
                                                    max_label=max_label)
        metrics['porosity_error'] = abs(metrics['porosity_gt'] - metrics['porosity_pred'])
        
        print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        if is_label_data and 'label_accuracy' in metrics:
            print(f"Label Accuracy: {metrics['label_accuracy']:.4f}")
        print(f"Porosity Error: {metrics['porosity_error']:.4f}")
        
        results[config['name']] = metrics
        
        # 保存
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_{config['name']}.pth"))
        # 不保存完整重建（太大），只保存一个切片用于可视化
        np.save(os.path.join(output_dir, f"recon_slice_{config['name']}.npy"), 
                pred_volume[pred_volume.shape[0]//2])
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
    
    # 保存结果
    with open(os.path.join(output_dir, 'compression_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    return results


def run_sr_experiment(data_path: str, output_dir: str, epochs: int = 100,
                      downsample: int = 1, crop_size: int = None):
    """超分辨率实验"""
    from super_resolution import super_resolution_experiment
    
    print(f"\n{'='*60}")
    print("SUPER-RESOLUTION EXPERIMENT")
    print(f"{'='*60}")
    
    return super_resolution_experiment(
        data_path,
        output_dir,
        sr_factors=[2, 4],
        epochs=epochs,
        downsample=downsample,
        crop_size=crop_size,
    )


def run_all_experiments(data_path: str, base_output_dir: str, epochs: int = 100,
                        downsample: int = 1, crop_size: int = None):
    """运行所有实验"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Downsample: {downsample}")
    print(f"Crop size: {crop_size}")
    
    # 1. 压缩实验
    compression_results = run_compression_experiment(
        data_path,
        os.path.join(output_dir, 'compression'),
        epochs=epochs,
        downsample=downsample,
        crop_size=crop_size,
    )
    
    # 2. 超分辨率实验
    sr_results = run_sr_experiment(
        data_path,
        os.path.join(output_dir, 'super_resolution'),
        epochs=epochs,
        downsample=downsample,
        crop_size=crop_size,
    )
    
    # 汇总
    all_results = {
        'compression': compression_results,
        'super_resolution': sr_results,
        'config': {
            'data_path': data_path,
            'epochs': epochs,
            'downsample': downsample,
            'crop_size': crop_size,
        }
    }
    
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to AM or npy file')
    parser.add_argument('--output', type=str, default='output', help='Base output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor')
    parser.add_argument('--crop_size', type=int, default=None, help='Crop size for testing')
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['all', 'compression', 'sr'],
                        help='Which experiment to run')
    
    args = parser.parse_args()
    
    if args.experiment == 'all':
        run_all_experiments(args.data, args.output, args.epochs, 
                           args.downsample, args.crop_size)
    elif args.experiment == 'compression':
        run_compression_experiment(args.data, os.path.join(args.output, 'compression'), 
                                   args.epochs, args.downsample, args.crop_size)
    elif args.experiment == 'sr':
        run_sr_experiment(args.data, os.path.join(args.output, 'super_resolution'), 
                          args.epochs, args.downsample, args.crop_size)
