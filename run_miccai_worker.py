#!/usr/bin/env python
"""
MICCAI Worker — Self-Contained Single-Encoding Trainer

Called by run_miccai_slurm.sbatch as an array job.
Each SLURM array task trains ONE encoding across ALL images for one split.

NO external src/ or configs/ dependencies — everything is embedded.

Usage:
    python run_miccai_worker.py --encoding fourier --split train
    python run_miccai_worker.py --encoding hybrid --split test --quick

Results are saved to:
    results/miccai/metrics/results_{split}_{encoding}.csv
    results/miccai/reconstructions_{split}/{slice}_{encoding}.npy
    results/miccai/convergence/{slice}_{encoding}.json
"""

import argparse
import copy
import json
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')


# ============================================================================
# EMBEDDED CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    'data': {
        'base_dir': 'data',
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'image_size': 'auto',
    },
    'encodings': {
        'num_frequencies': 256,
        'sigma': 10.0,
    },
    'network': {
        'hidden_dim': 256,
        'num_layers': 4,
        'skip_connection': True,
        'activation': 'relu',
    },
    'training': {
        'num_iterations': 15000,
        'batch_size': 4096,
        'learning_rate': 1e-4,
        'lr_decay': 0.1,
        'decay_steps': [10000],
        'seed': 42,
        'eval_checkpoint_every': 500,
        'snapshot_iterations': [500, 1000, 2000, 5000, 10000, 15000],
    },
    'evaluation': {
        'edge_threshold': 0.1,
        'edge_band_width': 5,
    },
    'experiment': {
        'device': 'cuda', #auto
    },
}


# ============================================================================
# EMBEDDED UTILITIES
# ============================================================================

def load_config(path=None):
    if path is not None and Path(path).exists():
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    return copy.deepcopy(DEFAULT_CONFIG)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================
# EMBEDDED ENCODINGS
# ============================================================================

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class TropicalFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    @staticmethod
    def triangle_cos(z):
        z_norm = z / np.pi
        return 1 - 2 * torch.abs(z_norm - 2 * torch.floor(z_norm / 2) - 1)

    @staticmethod
    def triangle_sin(z):
        z_shifted = z - np.pi / 2
        z_norm = z_shifted / np.pi
        return 1 - 2 * torch.abs(z_norm - 2 * torch.floor(z_norm / 2) - 1)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([self.triangle_cos(x_proj), self.triangle_sin(x_proj)], dim=-1)


class TrapezoidFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    @staticmethod
    def trapezoid_cos(z):
        z_norm = (z % (2 * np.pi)) / (2 * np.pi)
        result = torch.zeros_like(z_norm)
        result[z_norm <= 0.25] = 1.0
        mask2 = (z_norm > 0.25) & (z_norm <= 0.75)
        result[mask2] = 1.0 - 4.0 * (z_norm[mask2] - 0.25)
        result[z_norm > 0.75] = -1.0
        return result

    @staticmethod
    def trapezoid_sin(z):
        z_shifted = z - np.pi / 2
        z_norm = (z_shifted % (2 * np.pi)) / (2 * np.pi)
        result = torch.zeros_like(z_norm)
        result[z_norm <= 0.25] = 1.0
        mask2 = (z_norm > 0.25) & (z_norm <= 0.75)
        result[mask2] = 1.0 - 4.0 * (z_norm[mask2] - 0.25)
        result[z_norm > 0.75] = -1.0
        return result

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([self.trapezoid_cos(x_proj), self.trapezoid_sin(x_proj)], dim=-1)


class SquareWaveFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        z_cos = (x_proj % (2 * np.pi)) / (2 * np.pi)
        z_sin = ((x_proj - np.pi / 2) % (2 * np.pi)) / (2 * np.pi)
        cos_out = torch.where(z_cos < 0.5, torch.ones_like(z_cos), -torch.ones_like(z_cos))
        sin_out = torch.where(z_sin < 0.5, torch.ones_like(z_sin), -torch.ones_like(z_sin))
        return torch.cat([cos_out, sin_out], dim=-1)


class RampupFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        z_cos = (x_proj % (2 * np.pi)) / (2 * np.pi)
        z_sin = ((x_proj - np.pi / 2) % (2 * np.pi)) / (2 * np.pi)
        return torch.cat([2 * z_cos - 1, 2 * z_sin - 1], dim=-1)


class RampdownFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        z_cos = (x_proj % (2 * np.pi)) / (2 * np.pi)
        z_sin = ((x_proj - np.pi / 2) % (2 * np.pi)) / (2 * np.pi)
        return torch.cat([1 - 2 * z_cos, 1 - 2 * z_sin], dim=-1)


class NoEncoding(nn.Module):
    def __init__(self, in_dim=2, **kwargs):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x):
        return x


class SIRENEncoding(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.out_dim = num_frequencies * 2
        B = torch.empty(in_dim, num_frequencies).uniform_(-1.0 / in_dim, 1.0 / in_dim)
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = self.omega_0 * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class HashEncoding(nn.Module):
    """Multi-resolution hash encoding — Müller et al. 2022 (Instant-NGP).
    Indices and bilinear weights are pre-computed once per image via
    precompute(coords); each forward pass then only does table lookups.
    """
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0,
                 n_levels=16, features_per_level=2, log2_hashmap_size=14,
                 base_resolution=16, growth_factor=1.5):
        super().__init__()
        self.in_dim = in_dim
        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.out_dim = n_levels * features_per_level
        resolutions = [int(base_resolution * (growth_factor ** i)) for i in range(n_levels)]
        self.register_buffer('resolutions', torch.tensor(resolutions, dtype=torch.float32))
        self.hash_table = nn.Parameter(
            torch.randn(n_levels, self.hashmap_size, features_per_level) * 1e-4
        )
        self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))
        self._cached_idx = None
        self._cached_weights = None

    @torch.no_grad()
    def precompute(self, x):
        N, L = x.shape[0], self.n_levels
        x_unit  = (x + 1.0) / 2.0
        x_scaled = x_unit.unsqueeze(1) * self.resolutions.view(1, L, 1)
        x_floor  = x_scaled.long()
        x_ceil   = x_floor + 1
        w = x_scaled - x_floor.float()
        w0, w1 = w[..., 0:1], w[..., 1:2]
        corners = torch.stack([
            torch.stack([x_floor[..., 0], x_floor[..., 1]], dim=-1),
            torch.stack([x_ceil[..., 0],  x_floor[..., 1]], dim=-1),
            torch.stack([x_floor[..., 0], x_ceil[..., 1]],  dim=-1),
            torch.stack([x_ceil[..., 0],  x_ceil[..., 1]],  dim=-1),
        ], dim=2)
        idx = (corners[..., 0] * self.primes[0]) ^ (corners[..., 1] * self.primes[1])
        self._cached_idx     = (idx % self.hashmap_size).contiguous()
        self._cached_weights = torch.cat([
            (1-w0)*(1-w1), w0*(1-w1), (1-w0)*w1, w0*w1,
        ], dim=-1).contiguous()

    def clear_cache(self):
        self._cached_idx = None
        self._cached_weights = None

    # def forward(self, x):
    #     if self._cached_idx is not None:
    #         return self._forward_cached(x.shape[0])
    #     return self._forward_dynamic(x)
    
    def forward(self, x):
        T = self._cached_idx.shape[0] if self._cached_idx is not None else -1
        if self._cached_idx is not None and x.shape[0] == T:
            return self._forward_cached(T)
        return self._forward_dynamic(x)

    def _forward_cached(self, N):
        F   = self.features_per_level
        idx = self._cached_idx                                        # (N,L,4)
        idx_exp   = idx.unsqueeze(-1).expand(-1, -1, -1, F)
        table_exp = self.hash_table.unsqueeze(0).expand(N, -1, -1, -1)
        corner_features = table_exp.gather(2, idx_exp)               # (N,L,4,F)
        level_features  = (self._cached_weights.unsqueeze(-1) * corner_features).sum(2)
        return level_features.reshape(N, -1)

    def _forward_dynamic(self, x):
        N, L, F = x.shape[0], self.n_levels, self.features_per_level
        x_unit   = (x + 1.0) / 2.0
        x_scaled = x_unit.unsqueeze(1) * self.resolutions.view(1, L, 1)
        x_floor  = x_scaled.long()
        x_ceil   = x_floor + 1
        w = x_scaled - x_floor.float()
        w0, w1 = w[..., 0:1], w[..., 1:2]
        corners = torch.stack([
            torch.stack([x_floor[..., 0], x_floor[..., 1]], dim=-1),
            torch.stack([x_ceil[..., 0],  x_floor[..., 1]], dim=-1),
            torch.stack([x_floor[..., 0], x_ceil[..., 1]],  dim=-1),
            torch.stack([x_ceil[..., 0],  x_ceil[..., 1]],  dim=-1),
        ], dim=2)
        corner_weights = torch.cat([
            (1-w0)*(1-w1), w0*(1-w1), (1-w0)*w1, w0*w1,
        ], dim=-1)
        idx = (corners[..., 0] * self.primes[0]) ^ (corners[..., 1] * self.primes[1])
        idx = idx % self.hashmap_size
        idx_exp   = idx.unsqueeze(-1).expand(-1, -1, -1, F)
        table_exp = self.hash_table.unsqueeze(0).expand(N, -1, -1, -1)
        corner_features = table_exp.gather(2, idx_exp)
        level_features  = (corner_weights.unsqueeze(-1) * corner_features).sum(2)
        return level_features.reshape(N, -1)

class WaveletEncoding(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0, n_scales=8):
        super().__init__()
        self.n_scales = n_scales
        n_dirs = num_frequencies // n_scales
        self.n_dirs = n_dirs
        self.out_dim = n_scales * n_dirs * 2
        B = torch.randn(in_dim, n_dirs) * sigma
        self.register_buffer('B', B)
        self.register_buffer('scales', torch.logspace(-1, 1, n_scales))

    @staticmethod
    def mexican_hat(z):
        z2 = z ** 2
        return (1.0 - z2) * torch.exp(-z2 / 2.0)

    @staticmethod
    def mexican_hat_derivative(z):
        z2 = z ** 2
        return z * (z2 - 3.0) * torch.exp(-z2 / 2.0)

    def forward(self, x):
        proj = x @ self.B
        features = []
        for scale in self.scales:
            z = proj / scale
            features.append(self.mexican_hat(z))
            features.append(self.mexican_hat_derivative(z))
        return torch.cat(features, dim=-1)


def get_encoding(name, in_dim=2, num_frequencies=256, sigma=10.0):
    encodings = {
        'none': NoEncoding, 'fourier': FourierFeatures,
        'tropical': TropicalFeatures, 'trapezoid': TrapezoidFeatures,
        'square': SquareWaveFeatures, 'rampup': RampupFeatures,
        'rampdown': RampdownFeatures, 'siren': SIRENEncoding,
        'wavelet': WaveletEncoding,
        'hash': HashEncoding, 
        
    }
    if name not in encodings:
        raise ValueError(f"Unknown encoding: '{name}'. Available: {list(encodings.keys())}")
    return encodings[name](in_dim=in_dim, num_frequencies=num_frequencies, sigma=sigma)


# ============================================================================
# EMBEDDED MODEL
# ============================================================================

class INRMLP(nn.Module):
    def __init__(self, encoding_name='fourier', num_frequencies=256, sigma=10.0,
                 hidden_dim=256, num_layers=4, out_dim=1, skip_connection=True, activation='relu'):
        super().__init__()
        self.encoding_name = encoding_name
        self.encoding = get_encoding(encoding_name, 2, num_frequencies, sigma)
        self.skip_connection = skip_connection
        self.skip_layer = num_layers // 2 if skip_connection else -1
        in_dim = self.encoding.out_dim
        act_fn = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU()}.get(activation, nn.ReLU())
        layers = [nn.Linear(in_dim, hidden_dim), act_fn]
        for i in range(1, num_layers - 1):
            if i == self.skip_layer and skip_connection:
                layers.append(nn.Linear(hidden_dim + in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.in_dim = in_dim

    def forward(self, coords):
        x = self.encoding(coords)
        encoded = x
        layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if layer_idx == self.skip_layer and self.skip_connection:
                    x = torch.cat([x, encoded], dim=-1)
                x = layer(x)
                layer_idx += 1
            else:
                x = layer(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# EMBEDDED TRAINING
# ============================================================================

def train_inr(model, image, config, verbose=False):
    from tqdm import tqdm

    device = config['experiment']['device']
    model = model.to(device)
    H, W = image.shape
    num_pixels = H * W

    y_coords = torch.linspace(0, 1, H)
    x_coords = torch.linspace(0, 1, W)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)
    target = torch.from_numpy(image).float().reshape(-1, 1).to(device)

    # Pre-compute hash indices once for fixed image grid (Müller et al. 2022)
    if hasattr(model.encoding, 'precompute'):
        model.encoding.precompute(coords)

    num_iterations = config['training']['num_iterations']
    batch_size = min(config['training']['batch_size'], num_pixels)
    learning_rate = config['training']['learning_rate']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training'].get('decay_steps', []),
        gamma=config['training'].get('lr_decay', 0.1)
    )

    eval_every = config['training'].get('eval_checkpoint_every', 500)
    snapshot_iters = set(config['training'].get('snapshot_iterations', []))

    from skimage.metrics import structural_similarity

    loss_history = []
    all_metrics_over_time = {}   # {iteration: {psnr, ssim, edge_psnr, flat_psnr, ringing_score}}

    pbar = range(num_iterations)
    if verbose:
        pbar = tqdm(pbar, desc="Training INR")

    for iteration in pbar:
        model.train()
        indices = torch.randint(0, num_pixels, (batch_size,))
        optimizer.zero_grad()
        pred = model(coords[indices])
        loss = criterion(pred, target[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        is_checkpoint = (iteration > 0 and iteration % eval_every == 0) or (iteration == num_iterations - 1)
        is_snapshot = iteration in snapshot_iters

        if is_checkpoint or is_snapshot:
            model.eval()
            with torch.no_grad():
                recon_full = model(coords).cpu().numpy().reshape(H, W)
                recon_full = np.clip(recon_full, 0, 1)
            mse_val = float(np.mean((image - recon_full) ** 2))
            psnr_val = 100.0 if mse_val == 0 else float(20 * np.log10(1.0 / np.sqrt(mse_val)))
            ssim_val = float(structural_similarity(image, recon_full, data_range=1.0))
            edge_m   = _compute_metrics(image, recon_full, config)
            all_metrics_over_time[iteration] = {
                'psnr':          psnr_val,
                'ssim':          ssim_val,
                'edge_psnr':     edge_m.get('edge_psnr', 0.0),
                'flat_psnr':     edge_m.get('flat_psnr', 0.0),
                'ringing_score': edge_m.get('ringing_score', 0.0),
            }

    model.eval()
    with torch.no_grad():
        reconstruction = model(coords).cpu().numpy().reshape(H, W)
        reconstruction = np.clip(reconstruction, 0, 1)

    # Release pre-computed hash cache
    if hasattr(model.encoding, 'clear_cache'):
        model.encoding.clear_cache()

    metrics = _compute_metrics(image, reconstruction, config)

    training_info = {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else 0,
        'all_metrics_over_time': all_metrics_over_time,
    }

    return reconstruction, metrics, training_info


def _compute_metrics(original, reconstruction, config):
    mse = np.mean((original - reconstruction) ** 2)
    psnr = 100 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

    from skimage.metrics import structural_similarity
    ssim = structural_similarity(original, reconstruction, data_range=1.0)

    from scipy import ndimage
    from scipy.ndimage import binary_dilation

    edges_x = ndimage.sobel(original, axis=0)
    edges_y = ndimage.sobel(original, axis=1)
    edge_mag = np.hypot(edges_x, edges_y)

    edge_threshold = config['evaluation'].get('edge_threshold', 0.1)
    edge_band_width = config['evaluation'].get('edge_band_width', 5)

    edge_mask = edge_mag > edge_threshold
    struct = np.ones((edge_band_width, edge_band_width))
    edge_band = binary_dilation(edge_mask, structure=struct)

    edge_psnr = 0
    if edge_band.sum() > 0:
        edge_mse = np.mean((original[edge_band] - reconstruction[edge_band]) ** 2)
        edge_psnr = 20 * np.log10(1.0 / np.sqrt(edge_mse + 1e-10))

    flat_psnr = 0
    flat_mask = ~edge_band
    if flat_mask.sum() > 0:
        flat_mse = np.mean((original[flat_mask] - reconstruction[flat_mask]) ** 2)
        flat_psnr = 20 * np.log10(1.0 / np.sqrt(flat_mse + 1e-10))

    ringing_score = 0.0
    residual = np.abs(original - reconstruction)
    if edge_mask.sum() > 0:
        ringing_score = np.std(residual[edge_mask])

    try:
        fft_orig = np.fft.fft2(original)
        fft_recon = np.fft.fft2(reconstruction)
        spectral_mse = float(np.mean(np.abs(fft_orig - fft_recon) ** 2))
    except Exception:
        spectral_mse = 0.0

    return {
        'psnr': float(psnr), 'ssim': float(ssim), 'mse': float(mse),
        'edge_psnr': float(edge_psnr), 'flat_psnr': float(flat_psnr),
        'ringing_score': float(ringing_score), 'spectral_mse': float(spectral_mse),
    }


# ============================================================================
# HYBRID ENCODING + PATCHED MODEL CREATION
# ============================================================================

class HybridFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        half = num_frequencies // 2
        self.fourier = FourierFeatures(in_dim, half, sigma)
        self.tropical = TropicalFeatures(in_dim, half, sigma)
        self.out_dim = self.fourier.out_dim + self.tropical.out_dim

    def forward(self, x):
        return torch.cat([self.fourier(x), self.tropical(x)], dim=-1)


def _patched_get_encoding(name, in_dim=2, num_frequencies=256, sigma=10.0):
    if name == 'hybrid':
        return HybridFeatures(in_dim=in_dim, num_frequencies=num_frequencies, sigma=sigma)
    return get_encoding(name, in_dim, num_frequencies, sigma)


def _patched_create_model(config, encoding_name):
    model = INRMLP(
        encoding_name='fourier',
        num_frequencies=config['encodings']['num_frequencies'],
        sigma=config['encodings']['sigma'],
        hidden_dim=config['network']['hidden_dim'],
        num_layers=config['network']['num_layers'],
        skip_connection=config['network']['skip_connection'],
        activation=config['network']['activation'],
    )
    model.encoding = _patched_get_encoding(
        encoding_name, in_dim=2,
        num_frequencies=config['encodings']['num_frequencies'],
        sigma=config['encodings']['sigma'],
    )
    model.encoding_name = encoding_name
    new_in_dim = model.encoding.out_dim
    if new_in_dim != model.in_dim:
        old_first = model.layers[0]
        model.layers[0] = nn.Linear(new_in_dim, old_first.out_features)
        if model.skip_connection and model.skip_layer > 0:
            layer_idx = 0
            for i, layer in enumerate(model.layers):
                if isinstance(layer, nn.Linear):
                    if layer_idx == model.skip_layer:
                        expected_in = model.layers[i].in_features
                        corrected_in = expected_in - model.in_dim + new_in_dim
                        model.layers[i] = nn.Linear(corrected_in, model.layers[i].out_features)
                        break
                    layer_idx += 1
        model.in_dim = new_in_dim
    return model


# ============================================================================
# CONSTANTS
# ============================================================================

MICCAI_IMAGE_SIZE = 256
PSNR_CAP = 50.0
SEED = 42
CONVERGENCE_CHECKPOINTS = [1000, 2000, 2500, 3000, 3500]


def cap_psnr(val):
    return min(float(val), PSNR_CAP)


# ============================================================================
# MAIN WORKER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MICCAI Worker — single encoding trainer (self-contained)')
    parser.add_argument('--encoding', required=True,
                        help='Encoding name (none, fourier, tropical, hybrid, ...)')
    parser.add_argument('--split', default='train', choices=['train', 'test'],
                        help='Data split to train on')
    parser.add_argument('--config', default=None,
                        help='Optional YAML config (uses embedded defaults if not provided)')
    parser.add_argument('--quick', action='store_true', help='Fast test (2k iters)')
    parser.add_argument('--output-dir', default='results/miccai',
                        help='Output directory')
    args = parser.parse_args()

    enc_name = args.encoding
    split = args.split
    output_dir = Path(args.output_dir)

    print(f"=" * 60)
    print(f"MICCAI Worker (Self-Contained)")
    print(f"  Encoding: {enc_name}")
    print(f"  Split:    {split}")
    print(f"  Quick:    {args.quick}")
    print(f"=" * 60)

    config = load_config(args.config)
    config['data']['image_size'] = MICCAI_IMAGE_SIZE

    if args.quick:
        config['training']['num_iterations'] = 1000
        config['training']['batch_size'] = 4096
        config['training']['learning_rate'] = 1e-4
        config['training']['eval_checkpoint_every'] = 250
        config['training']['snapshot_iterations'] = [500, 1000]
    else:
        config['training']['num_iterations'] = 3500
        config['training']['batch_size'] = 4096
        config['training']['learning_rate'] = 1e-4
        config['training']['eval_checkpoint_every'] = 500
        config['training']['snapshot_iterations'] = [1000, 2000, 2500, 3000, 3500]

    config['training']['decay_steps'] = [2500]
    config['training']['lr_decay'] = 0.1
    config['training']['seed'] = SEED
    config['evaluation']['edge_threshold'] = 0.1
    config['evaluation']['edge_band_width'] = 5

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    config['experiment']['device'] = device
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(SEED)

    proc_dir = output_dir / 'processed'
    cohort_path = output_dir / 'cohort.csv'
    if not cohort_path.exists():
        print(f"ERROR: {cohort_path} not found. Run preprocess first.")
        sys.exit(1)

    cohort_df = pd.read_csv(cohort_path)
    if 'split' in cohort_df.columns:
        split_df = cohort_df[cohort_df['split'] == split]
    else:
        split_df = cohort_df
    print(f"Images for {split}: {len(split_df)}")

    if len(split_df) == 0:
        print(f"No images for split={split}. Exiting.")
        sys.exit(0)

    recon_dir = ensure_dir(output_dir / f'reconstructions_{split}')
    metrics_dir = ensure_dir(output_dir / 'metrics')
    convergence_dir = ensure_dir(output_dir / 'convergence')

    results_file = metrics_dir / f'results_{split}_{enc_name}.csv'

    completed = set()
    existing = pd.DataFrame()
    if results_file.exists():
        existing = pd.read_csv(results_file)
        completed = set(existing['slice_name'].values)
        print(f"Resuming: {len(completed)} already done for {enc_name}")

    all_results = []
    total = len(split_df)
    t_total_start = time.time()

    for idx, (_, row) in enumerate(split_df.iterrows()):
        slice_path = proc_dir / row['output_file']
        if not slice_path.exists():
            print(f"  Skip {row['output_file']} (not found)")
            continue

        sname = slice_path.stem
        if sname in completed:
            continue

        image = np.load(slice_path).astype(np.float32)
        print(f"  [{idx+1}/{total}] {row['patient_id']} | {enc_name} | "
              f"{row.get('modality', '?')} | {image.shape}")

        try:
            set_seed(SEED)
            cfg = copy.deepcopy(config)

            model = _patched_create_model(cfg, enc_name)
            t_start = time.time()
            recon, metrics, tinfo = train_inr(model, image, cfg, verbose=False)
            train_time_sec = time.time() - t_start

            np.save(recon_dir / f"{sname}_{enc_name}.npy", recon)

            full_cp_metrics = tinfo.get('all_metrics_over_time', {})
            psnr_curve = {it: cap_psnr(m['psnr']) for it, m in full_cp_metrics.items()}

            conv_path = convergence_dir / f"{sname}_{enc_name}.json"
            with open(conv_path, 'w') as f:
                json.dump({'psnr_curve': psnr_curve,
                           'full_metrics': {str(k): v for k, v in full_cp_metrics.items()}}, f)

            H, W = image.shape
            coords = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij'
            ), dim=-1).reshape(-1, 2).to(device)
            model.eval()
            with torch.no_grad():
                _ = model(coords[:1000])
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3):
                    _ = model(coords)
                if device == 'cuda':
                    torch.cuda.synchronize()
                inference_time = (time.time() - t0) / 3.0

            result = {
                'slice_name': sname, 'encoding': enc_name,
                'patient_id': row['patient_id'],
                'modality': row.get('modality', 'Unknown'),
                'split': split, 'rank': row.get('rank', 1),
                'psnr': cap_psnr(metrics['psnr']),
                'psnr_raw': float(metrics['psnr']),
                'ssim': float(metrics['ssim']),
                'mse': float(metrics['mse']),
                'edge_psnr': cap_psnr(metrics.get('edge_psnr', 0)),
                'edge_psnr_raw': float(metrics.get('edge_psnr', 0)),
                'flat_psnr': cap_psnr(metrics.get('flat_psnr', 0)),
                'flat_psnr_raw': float(metrics.get('flat_psnr', 0)),
                'ringing_score': float(metrics.get('ringing_score', 0)),
                'spectral_mse': float(metrics.get('spectral_mse', 0)),
                'num_params': model.count_parameters(),
                'encoding_dim': model.in_dim,
                'train_time_sec': train_time_sec,
                'inference_time_sec': inference_time,
                'final_loss': tinfo.get('final_loss', 0),
            }

            for cp in CONVERGENCE_CHECKPOINTS:
                best_it = None
                for it_key in sorted(full_cp_metrics.keys()):
                    if it_key <= cp:
                        best_it = it_key
                if best_it is not None:
                    m = full_cp_metrics[best_it]
                    result[f'psnr_at_{cp}']       = cap_psnr(m.get('psnr', np.nan))
                    result[f'ssim_at_{cp}']        = float(m.get('ssim', np.nan))
                    result[f'edge_psnr_at_{cp}']   = float(m.get('edge_psnr', np.nan))
                    result[f'flat_psnr_at_{cp}']   = float(m.get('flat_psnr', np.nan))
                    result[f'ringing_at_{cp}']     = float(m.get('ringing_score', np.nan))
                else:
                    for col in [f'psnr_at_{cp}', f'ssim_at_{cp}', f'edge_psnr_at_{cp}',
                                f'flat_psnr_at_{cp}', f'ringing_at_{cp}']:
                        result[col] = np.nan

            all_results.append(result)

            psnr_str = f"{result['psnr']:.1f}"
            print(f"    -> PSNR={psnr_str} dB | SSIM={result['ssim']:.3f} | "
                  f"time={train_time_sec:.1f}s")

            if len(all_results) % 5 == 0:
                _save(all_results, existing, results_file)

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()

    df = _save(all_results, existing, results_file)

    elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"Worker complete: {enc_name} ({split})")
    print(f"  Experiments: {len(df)}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Results: {results_file}")
    print(f"{'='*60}")


def _save(new_results, existing, path):
    df = pd.DataFrame(new_results)
    if len(existing) > 0:
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    main()
