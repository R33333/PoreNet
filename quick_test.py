"""
快速测试脚本 - 验证代码框架是否正常工作
"""

import torch
import numpy as np
import os
import sys

def test_am_loader():
    """测试 AM 文件读取器"""
    print("\n" + "="*50)
    print("测试 1: AM 文件读取器")
    print("="*50)
    
    from am_loader import create_synthetic_rock, write_am_file, read_am_file
    
    # 创建合成数据
    volume = create_synthetic_rock(shape=(64, 64, 64), porosity=0.2)
    print(f"✓ 创建合成岩心: shape={volume.shape}")
    
    # 保存为 AM
    write_am_file('test_data/test_rock.am', volume, compress=True)
    print("✓ 保存为 AM 文件")
    
    # 读回
    volume_loaded = read_am_file('test_data/test_rock.am')
    print(f"✓ 读取 AM 文件: shape={volume_loaded.shape}")
    
    # 验证
    if np.array_equal(volume, volume_loaded):
        print("✓ 数据验证通过")
    else:
        print("✗ 数据不一致!")
        return False
    
    return True


def test_model():
    """测试 INR 模型"""
    print("\n" + "="*50)
    print("测试 2: INR 模型")
    print("="*50)
    
    from model import DigitalRockINR, HashEncoding, SinusoidalEncoding
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 测试 Hash Encoding
    encoder = HashEncoding(n_levels=8, n_features_per_level=2).to(device)
    test_coords = torch.rand(1000, 3, device=device)
    features = encoder(test_coords)
    print(f"✓ Hash Encoding: input={test_coords.shape}, output={features.shape}")
    
    # 测试完整模型
    model = DigitalRockINR(
        encoding_type='hash',
        n_levels=8,
        log2_hashmap_size=16,
        hidden_dim=32,
        n_hidden_layers=2,
    ).to(device)
    
    output = model(test_coords)
    print(f"✓ INR 模型: input={test_coords.shape}, output={output.shape}")
    
    # 测试体积查询
    volume = model.query_volume((16, 16, 16), device)
    print(f"✓ 体积查询: output={volume.shape}")
    
    # 计算模型大小
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"✓ 模型大小: {model_size / 1024:.2f} KB")
    
    return True


def test_dataset():
    """测试数据集"""
    print("\n" + "="*50)
    print("测试 3: 数据集加载")
    print("="*50)
    
    from dataset import DigitalRockDataset, load_digital_rock
    from torch.utils.data import DataLoader
    
    # 创建测试数据
    test_volume = np.random.randint(0, 256, (64, 64, 64), dtype=np.uint8)
    np.save('test_data/test_volume.npy', test_volume)
    
    # 加载数据
    volume = load_digital_rock('test_data/test_volume.npy')
    print(f"✓ 加载数据: shape={volume.shape}")
    
    # 创建数据集
    dataset = DigitalRockDataset(volume, n_samples_per_epoch=10000)
    print(f"✓ 数据集大小: {len(dataset)}")
    
    # 测试 DataLoader
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    coords, densities = next(iter(loader))
    print(f"✓ DataLoader: coords={coords.shape}, densities={densities.shape}")
    
    return True


def test_metrics():
    """测试评估指标"""
    print("\n" + "="*50)
    print("测试 4: 评估指标")
    print("="*50)
    
    from metrics import compute_psnr, compute_ssim, compute_porosity, compute_metrics
    
    # 创建测试数据
    gt = np.random.rand(32, 32, 32).astype(np.float32)
    pred = gt + np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    pred = np.clip(pred, 0, 1)
    
    psnr = compute_psnr(pred, gt)
    print(f"✓ PSNR: {psnr:.2f} dB")
    
    ssim = compute_ssim(pred, gt)
    print(f"✓ SSIM: {ssim:.4f}")
    
    porosity = compute_porosity(gt)
    print(f"✓ 孔隙度: {porosity:.4f}")
    
    metrics = compute_metrics(pred, gt)
    print(f"✓ 所有指标: {metrics}")
    
    return True


def test_training():
    """测试简单训练流程"""
    print("\n" + "="*50)
    print("测试 5: 训练流程 (mini)")
    print("="*50)
    
    from model import DigitalRockINR
    from dataset import DigitalRockDataset
    from metrics import compute_metrics
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建小数据
    gt_volume = np.random.rand(32, 32, 32).astype(np.float32)
    
    # 数据集
    dataset = DigitalRockDataset(gt_volume, n_samples_per_epoch=5000)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # 模型
    model = DigitalRockINR(
        encoding_type='hash',
        n_levels=4,
        log2_hashmap_size=14,
        hidden_dim=32,
        n_hidden_layers=2,
        finest_resolution=32,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练 5 个 epoch
    for epoch in range(5):
        total_loss = 0
        for coords, densities in loader:
            coords, densities = coords.to(device), densities.to(device)
            
            pred = model(coords)
            loss = torch.nn.functional.mse_loss(pred, densities)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.6f}")
    
    # 验证
    model.eval()
    with torch.no_grad():
        pred_volume = model.query_volume(gt_volume.shape, device).cpu().numpy()
    
    metrics = compute_metrics(pred_volume, gt_volume)
    print(f"✓ 训练完成 - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    
    return True


def main():
    print("="*60)
    print("INR Digital Rock - 快速测试")
    print("="*60)
    
    # 创建测试目录
    os.makedirs('test_data', exist_ok=True)
    
    tests = [
        ("AM 文件读取器", test_am_loader),
        ("INR 模型", test_model),
        ("数据集加载", test_dataset),
        ("评估指标", test_metrics),
        ("训练流程", test_training),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("所有测试通过！可以开始实验。" if all_passed else "部分测试失败，请检查。"))
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
