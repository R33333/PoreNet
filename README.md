# PoreNet: Structure-Preserving Neural Compression of Digital Rock Images via Pore-Connectivity Encoding

## Overview

PoreNet is a structure-preserving neural compression framework for segmented digital rock micro-CT images. It addresses the key limitation of generic implicit neural representations (INR): failure to preserve petrophysically critical properties such as porosity and pore connectivity during compression.

**Key results** (ILS sandstone-like carbonate, 8× compression):
- Porosity error reduced by **96%** compared to Instant-NGP baseline
- Connectivity error reduced by **76%**
- PSNR improved by +0.5%, label accuracy improved by +0.14%

### Two core innovations

1. **Pore-Connectivity Encoder (PCE)**: Combines 3D hash grids with orthogonal plane features to capture both fine-grained pore boundaries and cross-slice connectivity patterns.
2. **Geometry-Weighted Loss (GWL)**: Assigns higher reconstruction weights to phase boundaries and narrow pore throats based on geometric analysis.

## Project Structure

```
Hash-INR-Rock/
├── model.py               # PoreNet model (Hash Encoding + Plane Features + MLP)
├── train.py               # Main training script
├── dataset.py             # Dataset loading (AM, MAT, NPY formats)
├── metrics.py             # Evaluation metrics (PSNR, SSIM, porosity, connectivity)
├── am_loader.py           # Avizo/Amira AM file reader (HxByteRLE)
├── load_mrccm.py          # MRCCM dataset loader (HDF5 .mat)
├── preprocess_data.py     # Data inspection and preprocessing
├── check_am_header.py     # Lightweight AM file header checker
├── run_experiments.py     # Batch experiment runner
├── super_resolution.py    # Super-resolution experiment (additional capability)
├── visualize.py           # Publication-quality figure generation
├── generate_visuals.py    # Generate visualizations from trained models
├── summarize_results.py   # Aggregate experiment results
├── summarize_all_results.py  # Full results comparison table
├── quick_test.py          # Quick validation test
├── huaTu.py               # Academic plotting style configuration
├── run_cloud.sh           # Shell script for batch experiments
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU required)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Optional** (for faster training):
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Quick Start

### 1. Data Preparation

Supported formats:
- **MATLAB HDF5** (`.mat`) — MRCCM dataset (ILS, MEC samples)
- **Avizo/Amira** (`.am`) — segmented volumes with HxByteRLE compression
- **NumPy** (`.npy`) — raw voxel arrays

The MRCCM dataset can be downloaded from the [Digital Rocks Portal](https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-362).

### 2. Run Compression (PoreNet)

```bash
python train.py \
    --data MRCCM/ILS_seg_hr.mat \
    --crop_size 256 \
    --epochs 60 \
    --n_levels 16 \
    --log2_hashmap_size 15 \
    --use_plane_features \
    --plane_levels 4 \
    --plane_log2_hashmap_size 13 \
    --boundary_weight 1.5 \
    --throat_weight 1.0 \
    --output output/ils_crop256_porenet
```

### 3. Run Instant-NGP Baseline

```bash
python train.py \
    --data MRCCM/ILS_seg_hr.mat \
    --crop_size 256 \
    --epochs 60 \
    --n_levels 16 \
    --log2_hashmap_size 15 \
    --output output/ils_crop256_baseline
```

### 4. Batch Experiments

```bash
bash run_cloud.sh quick        # Quick test (256³ crop, 20 epochs)
bash run_cloud.sh compression  # Full compression experiment
bash run_cloud.sh all          # All experiments
```

### 5. Summarize Results

```bash
python summarize_all_results.py
```

## Key Parameters

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_levels` | 16 | Number of hash encoding levels |
| `--log2_hashmap_size` | 15 | Hash table size (2^15 entries per level) |
| `--hidden_dim` | 64 | MLP hidden layer width |
| `--use_plane_features` | False | Enable PCE orthogonal plane branch |
| `--plane_levels` | 4 | Number of plane encoding levels |
| `--plane_log2_hashmap_size` | 13 | Plane hash table size |

### GWL (Geometry-Weighted Loss)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--boundary_weight` | 1.5 | Weight for phase boundary voxels |
| `--throat_weight` | 1.0 | Weight for narrow pore throat voxels |
| `--throat_radius` | 2.5 | Throat detection radius (voxels) |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 60 | Training epochs |
| `--batch_size` | 2^15 | Samples per iteration |
| `--lr` | 1e-3 | Learning rate (Adam + cosine annealing) |

## Datasets

Experiments use the **Multi-Resolution Complex Carbonates Micro-CT Dataset (MRCCM)**:

| Sample | Type | Micro-porosity | Characteristics |
|--------|------|----------------|-----------------|
| ILS | Sandstone-like carbonate | ~5% | Relatively homogeneous |
| MEC | Complex carbonate | ~14% | Fine-grained, high complexity |

Both samples contain three segmented phases: pore (1), solid (0), and micro-porosity (3).

## Hardware

Tested on:
- **GPU**: NVIDIA Tesla T4 (16 GB)
- **CPU**: Intel Xeon Gold 5218R (2×20 cores)
- **RAM**: 256 GB

Training time: ~13–15 minutes per 256³ configuration (60 epochs).

## Citation

```bibtex
@article{ren2026porenet,
  title={PoreNet: Structure-Preserving Neural Compression of Digital Rock Images via Pore-Connectivity Encoding},
  author={Ren, Yong and Dou, Liangbin and Liu, Ye and Wang, Bin and Zhang, Min},
  journal={Computers \& Geosciences},
  year={2026}
}
```

## Acknowledgments

This research was supported by the National Science and Technology Major Project (Grant No. 2025ZD1404300). We thank the [Digital Rocks Portal](https://www.digitalrocksportal.org/) for providing data infrastructure and the contributors of the MRCCM dataset.

## License

MIT License
