```markdown
# PoreNet: Structure-Preserving Neural Compression of Digital Rock Images via Pore-Connectivity Encoding

## Overview

PoreNet is a structure-aware neural compression framework for segmented digital-rock micro-CT images. It combines a Pore-Connectivity Encoder (PCE), which integrates local 3D hash-grid features with orthogonal-plane features, and a Geometry-Weighted Loss (GWL), which emphasizes phase interfaces and boundary-adjacent pore regions during training.

PoreNet is designed as a task-oriented lossy representation for remote visualization, data transfer, and structure-based screening. It is not intended to replace lossless archival storage or calibrated permeability and multiphase-flow simulations.

## Main Results

The detailed baseline, ablation, capacity, parameter-matched, and spatial-crop analyses focus on the ILS and MEC carbonate samples. A separate cohort experiment evaluates eight samples consisting of four sandstone samples and four carbonate samples.

On the ILS crop at approximately 6.7x compression, PoreNet:

- Reduces porosity error by 96% relative to the Hash-INR baseline.
- Reduces largest-connected-component error by 76%.
- Improves selected reconstruction metrics.
- Produces a larger representation and a lower nominal compression ratio than the Hash-INR baseline.
- Does not uniformly improve component-count or fine-scale topological accuracy.

The eight-sample cohort includes four sandstone samples and four carbonate samples. Relative to the Hash-INR baseline, PoreNet reduces the cohort mean porosity error by 75.4%, largest-connected-component error by 87.0%, and specific-surface-area error by 52.3% under the tested protocol.

## Scope and Limitations

The primary operating regime demonstrated in the study is the tested 256^3 crop range with the evaluated log15 configuration.

The 512^3 scale-stress experiment shows that PoreNet can improve porosity agreement at a larger crop size, but connectivity and component-count errors increase. Larger-volume deployment therefore requires validation of the target structural descriptors and may require increased model capacity, tiled processing, hierarchical reconstruction, or spatially adaptive encoding.

The repository also includes experiments for:

- Random-seed repeatability.
- Hash-table capacity sensitivity.
- Parameter-matched 3D hash baselines.
- Spatial-crop robustness.
- Scalar label-permutation sensitivity.
- Relative transport-proxy evaluation.
- Optional small-component filtering.

The transport-proxy experiment is a dimensionless voxel-graph test. It does not establish absolute Darcy permeability, multiphase relative permeability, or general validity for calibrated flow simulations.

## Project Structure

```text
PoreNet/
├── model.py
├── train.py
├── dataset.py
├── metrics.py
├── am_loader.py
├── load_mrccm.py
├── preprocess_data.py
├── check_am_header.py
├── run_experiments.py
├── generate_visuals.py
├── visualize.py
├── summarize_results.py
├── summarize_all_results.py
├── quick_test.py
├── transport_proxy.py
├── label_permutation_experiment.py
├── postprocess_filter.py
├── requirements.txt
├── configs/
│   ├── crop_coordinates.json
│   ├── random_seeds.json
│   ├── ils_crop256_porenet.json
│   ├── ils_crop256_hash_inr.json
│   ├── mec_crop256_porenet.json
│   └── ils_crop512_porenet.json
├── test_data/
│   ├── test_sandstone_64.npy
│   └── test_carbonate_64.npy
└── README.md
```

## Installation

### Requirements

- Python 3.8 or later.
- PyTorch 2.7.1.
- CUDA 11.8.
- NumPy.
- SciPy.
- scikit-image.
- h5py.
- Matplotlib.
- tqdm.
- Pillow.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Optional accelerated hash encoding:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Quick Test

Two small 64^3 example volumes are included in `test_data/`:

```bash
python train.py \
    --data test_data/test_sandstone_64.npy \
    --epochs 5 \
    --output output/test_sandstone
```

```bash
python train.py \
    --data test_data/test_carbonate_64.npy \
    --epochs 5 \
    --output output/test_carbonate
```

These files are demonstration inputs and are not the original Digital Rocks Portal samples.

## Main PoreNet Experiment

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
    --batch_size 32768 \
    --lr 1e-3 \
    --seed 2026 \
    --output output/ils_crop256_porenet
```

## Hash-INR Baseline

```bash
python train.py \
    --data MRCCM/ILS_seg_hr.mat \
    --crop_size 256 \
    --epochs 60 \
    --n_levels 16 \
    --log2_hashmap_size 15 \
    --batch_size 32768 \
    --lr 1e-3 \
    --seed 2026 \
    --output output/ils_crop256_hash_inr
```

## Preprocessing

Supported input formats include:

- MATLAB HDF5 `.mat` files for MRCCM data.
- Avizo/Amira `.am` segmented volumes.
- NumPy `.npy` voxel arrays.

The preprocessing workflow does not use interpolation, intensity rescaling, augmentation, or random spatial cropping. Dataset-specific labels are retained as integers during loading. Normalized scalar codes are assigned only for model training.

Example:

```bash
python preprocess_data.py \
    --data path/to/input_volume \
    --all \
    --output preprocessed
```

## Dataset Sources

The experiments use eight segmented samples:

| Samples | Rock type | Source |
|---|---|---|
| Berea, Bentheimer, Bandera Gray, Parker | Sandstone | Digital Rocks Portal project DRP-317 |
| ILS, MEC | Carbonate | MRCCM dataset |
| P01, SW01 | Carbonate | Digital Rocks Portal project DRP-503 |

The original Digital Rocks Portal datasets are not redistributed in this repository. Users should obtain the source data from the corresponding project pages and comply with their terms of use.

ILS and MEC contain three segmented phases: pore, solid, and micro-porosity. The remaining six cohort samples are binary pore-solid volumes.

## Main Parameters

### Hash-grid and decoder

| Parameter | Value |
|---|---:|
| Hash levels | 16 |
| 3D hash-table capacity | 2^15 entries per level |
| 3D features per level | 2 |
| Base resolution | 16 |
| Plane levels | 4 |
| Plane hash-table capacity | 2^13 entries per level |
| Decoder hidden layers | 3 |
| Decoder width | 64 neurons per layer |
| Activation | ReLU |
| Output activation | Sigmoid |

### GWL

| Parameter | Value |
|---|---:|
| Boundary weight | 1.5 |
| Boundary-adjacent pore weight | 1.0 |
| Near-region threshold | 2.5 voxels |

### Training

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| Scheduler | Cosine annealing |
| Epochs | 60 |
| Batch size | 32768 |
| Samples per epoch | 1048576 |
| Default seed | 2026 |
| Repeatability seeds | 2026, 2027, 2028 |

## Reproducibility Records

The `configs/` directory records:

- Crop size.
- Crop origin.
- Random seed.
- Hash-grid parameters.
- Plane-feature parameters.
- Decoder configuration.
- Optimizer and scheduler.
- Batch size.
- Samples per epoch.
- Number of epochs.
- GWL weights.
- Output file locations.

The primary experiments use 256^3 crops. The spatial-crop experiment uses the following offsets along the first dimension:

```text
ILS:
(0, 632, 632)
(1648, 632, 632)
(3296, 632, 632)

MEC:
(0, 632, 632)
(1922, 632, 632)
(3844, 632, 632)
```

The repeatability experiment uses seeds 2026, 2027, and 2028.

## Output Files

Training produces:

```text
best_model.pth
final_model.pth
best_reconstruction.npy
results.json
```

The trained representations are stored as PyTorch state-dictionary checkpoints. Reconstructed volumes are stored as `.npy` files, and metric summaries are stored as `.json` files.

## Additional Analyses

### Label-permutation sensitivity

```bash
python label_permutation_experiment.py
```

The experiment evaluates the six possible three-phase label encodings:

```text
123, 132, 213, 231, 312, 321
```

### Relative transport proxy

```bash
python transport_proxy.py \
    --ground-truth path/to/ground_truth.npy \
    --reconstruction path/to/reconstruction.npy \
    --output output/transport_proxy.json
```

The proxy downsamples pore masks to 64^3 using 4^3 block majority voting and evaluates a unit-conductance six-neighbour voxel graph. It reports conductance normalized by the corresponding ground-truth value.

### Result summaries

```bash
python summarize_results.py
python summarize_all_results.py
```

## Hardware

The reported experiments were conducted using:

- NVIDIA Tesla T4 GPU with 16 GB memory.
- PyTorch 2.7.1.
- CUDA 11.8.

A 256^3 crop takes approximately 13–15 minutes per configuration for 60 epochs on one Tesla T4 GPU. The reported time depends on the hardware, configuration, and input volume.

