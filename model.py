import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class SinusoidalEncoding(nn.Module):
    def __init__(self, n_frequencies: int = 10):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.register_buffer('freqs', 2.0 ** torch.arange(n_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) * self.freqs * np.pi
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return encoded.view(x.shape[0], -1)

    @property
    def output_dim(self):
        return 3 * 2 * self.n_frequencies


class HashEncoding(nn.Module):
    def __init__(self, n_levels: int = 16, n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19, base_resolution: int = 16,
                 finest_resolution: int = 512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / max(n_levels - 1, 1))
        self.resolutions = [int(np.ceil(base_resolution * (b ** i))) for i in range(n_levels)]
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features_per_level) * 0.01)
            for _ in range(n_levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0 - 1e-6)
        all_features = []
        for level, resolution in enumerate(self.resolutions):
            scaled = x * resolution
            floor_coords = scaled.floor().long()
            weights = scaled - floor_coords.float()
            features = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        corner = floor_coords + torch.tensor([i, j, k], device=x.device)
                        hash_idx = self._hash_3d(corner)
                        w = (weights[:, 0] if i else (1 - weights[:, 0])) * \
                            (weights[:, 1] if j else (1 - weights[:, 1])) * \
                            (weights[:, 2] if k else (1 - weights[:, 2]))
                        features += w.unsqueeze(-1) * self.hash_tables[level][hash_idx]
            all_features.append(features)
        return torch.cat(all_features, dim=-1)

    def _hash_3d(self, coords: torch.Tensor) -> torch.Tensor:
        primes = [1, 2654435761, 805459861]
        result = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for i in range(3):
            result ^= coords[:, i] * primes[i]
        return result % self.hashmap_size

    @property
    def output_dim(self):
        return self.n_levels * self.n_features_per_level


class PlaneHashEncoding(nn.Module):
    def __init__(self, n_levels: int = 8, n_features_per_level: int = 2,
                 log2_hashmap_size: int = 15, base_resolution: int = 16,
                 finest_resolution: int = 512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.n_planes = 3
        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / max(n_levels - 1, 1))
        self.resolutions = [int(np.ceil(base_resolution * (b ** i))) for i in range(n_levels)]
        self.plane_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features_per_level) * 0.01)
            for _ in range(self.n_planes * n_levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0 - 1e-6)
        plane_coords = (x[:, [0, 1]], x[:, [0, 2]], x[:, [1, 2]])
        all_features = []
        for plane_id, coords_2d in enumerate(plane_coords):
            for level, resolution in enumerate(self.resolutions):
                scaled = coords_2d * resolution
                floor_coords = scaled.floor().long()
                weights = scaled - floor_coords.float()
                features = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)
                table = self.plane_tables[plane_id * self.n_levels + level]
                for i in range(2):
                    for j in range(2):
                        corner = floor_coords + torch.tensor([i, j], device=x.device)
                        hash_idx = self._hash_2d(corner)
                        w = (weights[:, 0] if i else (1 - weights[:, 0])) * \
                            (weights[:, 1] if j else (1 - weights[:, 1]))
                        features += w.unsqueeze(-1) * table[hash_idx]
                all_features.append(features)
        return torch.cat(all_features, dim=-1)

    def _hash_2d(self, coords: torch.Tensor) -> torch.Tensor:
        primes = [1, 2654435761]
        result = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for i in range(2):
            result ^= coords[:, i] * primes[i]
        return result % self.hashmap_size

    @property
    def output_dim(self):
        return self.n_planes * self.n_levels * self.n_features_per_level


class HybridGridPlaneEncoding(nn.Module):
    def __init__(self, n_levels: int = 16, n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19, base_resolution: int = 16,
                 finest_resolution: int = 512, plane_n_levels: int = 8,
                 plane_n_features_per_level: int = 2,
                 plane_log2_hashmap_size: int = 15,
                 plane_base_resolution: int = 16,
                 plane_finest_resolution: Optional[int] = None):
        super().__init__()
        self.grid_encoder = HashEncoding(n_levels, n_features_per_level, log2_hashmap_size,
                                         base_resolution, finest_resolution)
        self.plane_encoder = PlaneHashEncoding(
            plane_n_levels, plane_n_features_per_level, plane_log2_hashmap_size,
            plane_base_resolution, plane_finest_resolution or finest_resolution,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.grid_encoder(x), self.plane_encoder(x)], dim=-1)

    @property
    def output_dim(self):
        return self.grid_encoder.output_dim + self.plane_encoder.output_dim


class SirenLayer(nn.Module):
    """SIREN layer with sine activation"""
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, 
                 is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                           1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                    np.sqrt(6 / self.linear.in_features) / self.omega_0
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_last:
            return self.linear(x)
        return torch.sin(self.omega_0 * self.linear(x))


class DigitalRockINR(nn.Module):
    def __init__(self, encoding_type: str = 'hash', n_hidden_layers: int = 3,
                 hidden_dim: int = 64, n_levels: int = 16,
                 n_features_per_level: int = 2, log2_hashmap_size: int = 19,
                 base_resolution: int = 16, finest_resolution: int = 512,
                 n_frequencies: int = 10, plane_n_levels: int = 8,
                 plane_n_features_per_level: int = 2,
                 plane_log2_hashmap_size: int = 15,
                 plane_base_resolution: int = 16,
                 plane_finest_resolution: Optional[int] = None,
                 use_siren: bool = False, siren_omega_0: float = 30.0):
        super().__init__()
        self.encoding_type = encoding_type
        self.use_siren = use_siren
        
        # SIREN: no encoding, direct coordinate input
        if use_siren:
            self.encoder = None
            in_dim = 3
            layers = []
            layers.append(SirenLayer(in_dim, hidden_dim, omega_0=siren_omega_0, is_first=True))
            for _ in range(n_hidden_layers):
                layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=siren_omega_0))
            layers.append(SirenLayer(hidden_dim, 1, omega_0=siren_omega_0, is_last=True))
            layers.append(nn.Sigmoid())
            self.mlp = nn.Sequential(*layers)
        else:
            # Standard encoding + MLP
            if encoding_type == 'hash':
                self.encoder = HashEncoding(n_levels, n_features_per_level, log2_hashmap_size,
                                            base_resolution, finest_resolution)
            elif encoding_type == 'hybrid':
                self.encoder = HybridGridPlaneEncoding(
                    n_levels, n_features_per_level, log2_hashmap_size, base_resolution,
                    finest_resolution, plane_n_levels, plane_n_features_per_level,
                    plane_log2_hashmap_size, plane_base_resolution, plane_finest_resolution,
                )
            elif encoding_type == 'sinusoidal':
                self.encoder = SinusoidalEncoding(n_frequencies)
            else:
                raise ValueError(f"Unsupported encoding_type: {encoding_type}")

            layers = []
            in_dim = self.encoder.output_dim
            for _ in range(n_hidden_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            layers.append(nn.Sigmoid())
            self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_siren:
            return self.mlp(coords)
        else:
            return self.mlp(self.encoder(coords))

    def query_volume(self, resolution: Tuple[int, int, int], device: str = 'cuda',
                     chunk_size: int = 64) -> torch.Tensor:
        D, H, W = resolution
        volume = torch.zeros(D, H, W, dtype=torch.float32)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xy_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        with torch.no_grad():
            for z_start in range(0, D, chunk_size):
                z_end = min(z_start + chunk_size, D)
                z_vals = torch.linspace(z_start / D, (z_end - 1) / D, z_end - z_start, device=device)
                chunk_densities = []
                for z_val in z_vals:
                    z_coord = torch.full((H * W, 1), z_val.item(), device=device)
                    coords = torch.cat([xy_coords, z_coord], dim=-1)
                    layer_density = []
                    for i in range(0, coords.shape[0], 256 * 256):
                        layer_density.append(self(coords[i:i + 256 * 256]).cpu())
                    chunk_densities.append(torch.cat(layer_density, dim=0).reshape(H, W))
                volume[z_start:z_end] = torch.stack(chunk_densities, dim=0)
                if (z_start // chunk_size) % 5 == 0:
                    print(f"  Reconstructing: {z_end}/{D} slices ({100 * z_end / D:.1f}%)")
        if volume.numel() * volume.element_size() / (1024 ** 3) > 20:
            return volume
        return volume.to(device)


class DigitalRockINRTinyCuda(nn.Module):
    def __init__(self, n_levels: int = 16, n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19, base_resolution: int = 16,
                 finest_resolution: int = 512, n_hidden_layers: int = 2,
                 hidden_dim: int = 64):
        super().__init__()
        try:
            import tinycudann as tcnn
            self.use_tcnn = True
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp((np.log(finest_resolution) - np.log(base_resolution)) / max(n_levels - 1, 1)),
            }
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            }
            self.model = tcnn.NetworkWithInputEncoding(3, 1, encoding_config, network_config)
        except ImportError:
            print("警告: tiny-cuda-nn 未安装，使用纯 PyTorch 实现")
            self.use_tcnn = False
            self.model = DigitalRockINR('hash', n_hidden_layers, hidden_dim, n_levels,
                                        n_features_per_level, log2_hashmap_size,
                                        base_resolution, finest_resolution)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.model(coords)

    def query_volume(self, resolution: Tuple[int, int, int], device: str = 'cuda') -> torch.Tensor:
        return self.model.query_volume(resolution, device) if not self.use_tcnn else self._query_tcnn(resolution, device)

    def _query_tcnn(self, resolution: Tuple[int, int, int], device: str = 'cuda') -> torch.Tensor:
        D, H, W = resolution
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        densities = []
        with torch.no_grad():
            for i in range(0, coords.shape[0], 1024 * 1024):
                densities.append(self.model(coords[i:i + 1024 * 1024].float()).cpu())
        return torch.cat(densities, dim=0).reshape(D, H, W)
