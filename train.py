"""
训练 INR 模型
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DigitalRockDataset, load_digital_rock
from metrics import compute_metrics, compute_porosity, detect_dataset_type
from model import DigitalRockINR, DigitalRockINRTinyCuda


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def infer_pore_labels(voxel_data: np.ndarray) -> list[int]:
    dataset_type = detect_dataset_type(voxel_data)
    if dataset_type == 'mrccm':
        return [1, 3]
    if dataset_type == 'berea':
        return [0]
    unique_vals = np.unique(voxel_data)
    return [int(unique_vals.min())]


def build_structure_weight_map(
    voxel_data: np.ndarray,
    pore_labels: list[int],
    boundary_weight: float = 3.0,
    throat_weight: float = 2.0,
    throat_radius_voxels: float = 2.5,
) -> np.ndarray:
    """
    为标签体数据构造结构感知权重图。

    - 边界区域：提高权重，减少相界面模糊
    - 狭窄孔隙/细喉：进一步提高权重，优先保留连通通道
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError('Topology-aware training requires scipy.') from exc

    pore_mask = np.isin(voxel_data, pore_labels)
    structure = np.ones((3, 3, 3), dtype=bool)

    # 相边界：邻域内出现不同标签
    boundary_mask = ndimage.generic_gradient_magnitude(
        voxel_data.astype(np.float32),
        derivative=lambda x, axis, output, mode, cval: ndimage.sobel(x, axis=axis, output=output, mode=mode, cval=cval),
    ) > 0

    # 孔隙内部到固体边界的距离，小距离代表细喉或狭窄通道
    pore_distance = ndimage.distance_transform_edt(pore_mask)
    throat_mask = pore_mask & (pore_distance <= throat_radius_voxels)

    # 对孔隙做一次轻微腐蚀，避免将所有孔隙都视为细喉
    eroded_pore = ndimage.binary_erosion(pore_mask, structure=structure, iterations=1, border_value=0)
    throat_mask = throat_mask & ~eroded_pore

    weights = np.ones(voxel_data.shape, dtype=np.float32)
    weights[boundary_mask] += boundary_weight
    weights[throat_mask] += throat_weight

    return weights


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    use_structure_weights: bool = False,
    loss_type: str = 'mse',
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for coords, densities, weights in pbar:
        coords = coords.to(device)
        densities = densities.to(device)
        weights = weights.to(device)

        pred = model(coords)

        if loss_type == 'cross_entropy':
            target = densities.long().view(-1)
            ce = nn.functional.cross_entropy(pred, target, reduction='none')
            loss = (ce * weights.view(-1)).mean() if use_structure_weights else ce.mean()
        elif use_structure_weights:
            loss_map = (pred - densities) ** 2
            loss = (loss_map * weights).mean()
        else:
            loss = nn.functional.mse_loss(pred, densities)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    gt_volume: np.ndarray,
    device: str,
    pore_labels: list[int],
    classification: bool = False,
    label_values: np.ndarray | None = None,
):
    """验证模型"""
    model.eval()

    resolution = gt_volume.shape
    if device == 'cuda':
        torch.cuda.synchronize()
    reconstruction_start = time.perf_counter()
    with torch.no_grad():
        pred_volume = model.query_volume(resolution, device)
        if classification:
            pred_volume = torch.argmax(pred_volume, dim=-1).float()
            if label_values is not None:
                pred_volume = torch.as_tensor(label_values, device=pred_volume.device)[pred_volume.long()]
                pred_volume = pred_volume / float(np.max(label_values))
        pred_volume = pred_volume.cpu().numpy()
    if device == 'cuda':
        torch.cuda.synchronize()
    reconstruction_time = time.perf_counter() - reconstruction_start

    gt_normalized = gt_volume.astype(np.float32)

    unique_values = np.unique(gt_volume)
    is_label_data = len(unique_values) < 20
    max_label = float(gt_volume.max()) if gt_volume.max() > 0 else 1.0

    if is_label_data:
        gt_normalized = gt_normalized / max_label
    elif gt_normalized.max() > 1.0:
        gt_normalized = gt_normalized / 255.0

    metrics = compute_metrics(
        pred_volume,
        gt_normalized,
        is_label_data=is_label_data,
        max_label=max_label,
        pore_labels=pore_labels,
    )

    gt_porosity = compute_porosity(
        gt_normalized,
        pore_labels=pore_labels if is_label_data else None,
        is_normalized=True,
        max_label=max_label,
    )
    pred_porosity = compute_porosity(
        pred_volume,
        pore_labels=pore_labels if is_label_data else None,
        is_normalized=True,
        max_label=max_label,
    )
    metrics['porosity_gt'] = gt_porosity
    metrics['porosity_pred'] = pred_porosity
    metrics['porosity_error'] = abs(gt_porosity - pred_porosity)
    metrics['reconstruction_time_seconds'] = reconstruction_time

    return metrics, pred_volume


def main():
    parser = argparse.ArgumentParser(description='Train INR for Digital Rock')
    parser.add_argument('--data', type=str, required=True, help='Path to AM, MAT, NPY, RAW, or NC file')
    parser.add_argument('--output', type=str, default='output/inr_experiment', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--samples_per_epoch', type=int, default=2000000, help='Samples per epoch')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor')
    parser.add_argument('--crop_size', type=int, default=None, help='Crop size (for testing)')
    parser.add_argument('--crop_origin', type=int, nargs=3, metavar=('D', 'H', 'W'), help='Explicit crop origin in D H W order')
    parser.add_argument('--raw_shape', type=int, nargs=3, metavar=('D', 'H', 'W'), help='RAW uint8 shape in D H W order')
    parser.add_argument('--nc_variable', type=str, default='data', help='Volume variable in NetCDF4/HDF5 files')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--use_tcnn', action='store_true', help='Use tiny-cuda-nn')
    parser.add_argument('--encoding_type', type=str, default='hash', choices=['hash', 'hybrid', 'sinusoidal'], help='Encoding type for the PyTorch INR model')
    parser.add_argument('--use_siren', action='store_true', help='Use SIREN (sine activation) instead of ReLU')
    parser.add_argument('--siren_omega_0', type=float, default=30.0, help='Omega_0 parameter for SIREN')
    parser.add_argument('--validate_every', type=int, default=10, help='Validate every N epochs')
    parser.add_argument('--n_levels', type=int, default=16, help='Hash encoding levels')
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help='Log2 hashmap size')
    parser.add_argument('--plane_n_levels', type=int, default=8, help='Plane feature encoding levels for hybrid grid-plane encoder')
    parser.add_argument('--plane_log2_hashmap_size', type=int, default=15, help='Log2 hashmap size for each plane-feature table')
    parser.add_argument('--plane_n_features_per_level', type=int, default=2, help='Features per level for plane-feature encoding')
    parser.add_argument('--plane_base_resolution', type=int, default=16, help='Base resolution for plane-feature encoding')
    parser.add_argument('--plane_axes', nargs='+', choices=['xy', 'xz', 'yz'], default=['xy', 'xz', 'yz'], help='Active feature planes')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_hidden_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--pore_label', type=int, default=None, help='Optional single pore label override')
    parser.add_argument('--use_structure_loss', action='store_true', help='Enable PoreGuard-INR structure-weighted loss')
    parser.add_argument('--boundary_weight', type=float, default=1.5, help='Extra weight for phase boundaries in PoreGuard-INR')
    parser.add_argument('--throat_weight', type=float, default=1.0, help='Extra weight for narrow pore/throat voxels in PoreGuard-INR')
    parser.add_argument('--throat_radius', type=float, default=2.5, help='Distance threshold in voxels for throat emphasis')
    parser.add_argument('--loss_type', choices=['mse', 'cross_entropy'], default='mse', help='Label loss: scalar MSE or categorical cross-entropy')

    args = parser.parse_args()
    if args.use_tcnn and args.loss_type == 'cross_entropy':
        raise ValueError('Categorical experiment uses the PyTorch encoder for both arms.')

    os.makedirs(args.output, exist_ok=True)
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading data from: {args.data}")
    voxel_data = load_digital_rock(
        args.data,
        downsample=args.downsample,
        crop_size=args.crop_size,
        crop_origin=args.crop_origin,
        raw_shape=args.raw_shape,
        nc_variable=args.nc_variable,
    )
    print(f"Volume shape: {voxel_data.shape}")

    original_voxel_data = voxel_data.copy()
    unique_values = np.unique(voxel_data)
    is_label_data = len(unique_values) < 20
    max_label = float(voxel_data.max()) if voxel_data.max() > 0 else 1.0
    label_values = np.sort(unique_values.astype(np.int64)) if is_label_data else None
    label_to_index = {int(v): i for i, v in enumerate(label_values)} if label_values is not None else None
    if args.loss_type == 'cross_entropy' and not is_label_data:
        raise ValueError('--loss_type cross_entropy requires segmented label data')
    if args.loss_type == 'cross_entropy':
        voxel_data = np.vectorize(label_to_index.get)(voxel_data).astype(np.int64)
    print(f"Is label data: {is_label_data}, Max label: {max_label}")

    pore_labels = [args.pore_label] if args.pore_label is not None else infer_pore_labels(original_voxel_data)
    print(f"Pore labels used for topology/porosity evaluation: {pore_labels}")

    original_size = original_voxel_data.nbytes
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")

    sample_weights = None
    if args.use_siren:
        method_name = 'COIN (SIREN)'
    elif args.encoding_type == 'hybrid':
        method_name = 'PoreNet' if args.use_structure_loss else 'HGP-INR baseline'
    else:
        method_name = 'PoreGuard-INR' if args.use_structure_loss else 'Hash-INR baseline'
    print(f"Method: {method_name}")
    print(f"Encoding type: {args.encoding_type}")
    if args.use_siren:
        print(f"Using SIREN with omega_0={args.siren_omega_0}")
    if args.use_structure_loss:
        if not is_label_data:
            raise ValueError('PoreGuard-INR structure-weighted loss currently supports segmented/label data only.')
        print(
            'Building PoreGuard-INR structure weight map '
            f'(boundary_weight={args.boundary_weight}, '
            f'throat_weight={args.throat_weight}, '
            f'throat_radius={args.throat_radius})...'
        )
        sample_weights = build_structure_weight_map(
            original_voxel_data,
            pore_labels=pore_labels,
            boundary_weight=args.boundary_weight,
            throat_weight=args.throat_weight,
            throat_radius_voxels=args.throat_radius,
        )

    dataset = DigitalRockDataset(
        voxel_data,
        n_samples_per_epoch=args.samples_per_epoch,
        is_label_data=is_label_data and args.loss_type == 'mse',
        sample_weights=sample_weights,
        classification=args.loss_type == 'cross_entropy',
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    if args.use_tcnn:
        if args.encoding_type != 'hash':
            raise ValueError('--use_tcnn currently supports only --encoding_type hash.')
        if args.use_siren:
            raise ValueError('--use_tcnn does not support --use_siren.')
        model = DigitalRockINRTinyCuda(
            n_levels=args.n_levels,
            log2_hashmap_size=args.log2_hashmap_size,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            finest_resolution=max(voxel_data.shape),
        ).to(device)
    else:
        model = DigitalRockINR(
            encoding_type=args.encoding_type,
            n_levels=args.n_levels,
            log2_hashmap_size=args.log2_hashmap_size,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            finest_resolution=max(voxel_data.shape),
            plane_n_levels=args.plane_n_levels,
            plane_n_features_per_level=args.plane_n_features_per_level,
            plane_log2_hashmap_size=args.plane_log2_hashmap_size,
            plane_base_resolution=args.plane_base_resolution,
            plane_finest_resolution=max(voxel_data.shape),
            plane_axes=args.plane_axes,
            use_siren=args.use_siren,
            siren_omega_0=args.siren_omega_0,
            output_dim=len(label_values) if args.loss_type == 'cross_entropy' else 1,
        ).to(device)

    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {original_size / model_size:.2f}x")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0.0
    train_losses = []
    val_metrics_history = []

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        loss = train(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            use_structure_weights=args.use_structure_loss,
            loss_type=args.loss_type,
        )
        train_losses.append(loss)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % args.validate_every == 0 or epoch == args.epochs:
            metrics, pred_volume = validate(
                model, original_voxel_data if args.loss_type == 'cross_entropy' else voxel_data, device, pore_labels,
                classification=args.loss_type == 'cross_entropy',
                label_values=label_values,
            )
            val_metrics_history.append(metrics)

            print(f"  Validation - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
            if 'label_accuracy' in metrics:
                print(f"  Label accuracy: {metrics['label_accuracy']:.4f}")
            print(
                f"  Porosity - GT: {metrics['porosity_gt']:.4f}, "
                f"Pred: {metrics['porosity_pred']:.4f}, "
                f"Error: {metrics['porosity_error']:.4f}"
            )
            if 'connectivity_error' in metrics:
                print(
                    f"  Connectivity - Largest component ratio error: {metrics['connectivity_error']:.4f}, "
                    f"Component count error: {metrics['component_count_error']:.1f}"
                )

            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
                np.save(os.path.join(args.output, 'best_reconstruction.npy'), pred_volume)
                print(f"  Saved best model (PSNR: {best_psnr:.2f} dB)")

    if device == 'cuda':
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start_time
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device == 'cuda' else 0.0
    )
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    torch.save(model.state_dict(), os.path.join(args.output, 'final_model.pth'))
    np.save(os.path.join(args.output, 'final_reconstruction.npy'), pred_volume)

    results = {
        'method': method_name,
        'label_values': label_values.tolist() if label_values is not None else None,
        'args': vars(args),
        'original_size_mb': original_size / 1024 / 1024,
        'model_size_mb': model_size / 1024 / 1024,
        'compression_ratio': original_size / model_size,
        'best_psnr': best_psnr,
        'training_time_seconds': training_time,
        'peak_gpu_memory_mb': peak_gpu_memory_mb,
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
    }

    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {args.output}")
    print(f"Compression ratio: {original_size / model_size:.2f}x")
    print(f"Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
