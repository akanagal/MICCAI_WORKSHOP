#!/usr/bin/env python
"""
MICCAI Publication Pipeline — Self-Contained Single Script (v2)
================================================================

Complete pipeline for "Tropical Positional Encodings for Implicit Neural
Representations of Medical Images" — MICCAI 2025 submission.

This is a SELF-CONTAINED version — no external src/ or configs/ dependencies.
Everything is embedded in this file. Just place it alongside your data/ folder
and run.

Features:
  1. HELD-OUT EVALUATION: Train on rank-1 slice per volume, evaluate
     generalization on rank-2/3 slices (unseen during training).
  2. CONVERGENCE SPEED: Tracks PSNR at [500, 1k, 2k, 5k, 10k, 15k] iters.
     Generates Table 3 (iterations to reach threshold) + Figure 7.
  3. MODEL SIZE & TIMING: Table 4 with parameter counts, encoding dim,
     forward pass time, and per-pixel inference speed.
  4. BASELINE CITATIONS: Proper references to SIREN, Instant-NGP, wavelet.
  5. 30+ IMAGES PER MODALITY: Updated expectations and warnings.

Usage:
    python run_miccai.py                          # Full pipeline
    python run_miccai.py --skip-training          # Re-generate tables/figures
    python run_miccai.py --quick                  # Fast test (2000 iters)
    python run_miccai.py --encodings fourier tropical hybrid
    python run_miccai.py --skip-heldout           # Skip held-out eval
    python run_miccai.py --preprocess-only        # SLURM: preprocess then exit

Expects data in: data/raw/ (NIfTI, DICOM, SVS, PNG, JPEG, etc.)
Outputs to:      results/miccai/
"""

import argparse
import copy
import hashlib
import json
import random
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image as _PIL_Image
from scipy.ndimage import sobel, uniform_filter
from skimage.transform import resize as sk_resize

warnings.filterwarnings('ignore')


# ============================================================================
# EMBEDDED CONFIG (replaces configs/default.yaml)
# ============================================================================

DEFAULT_CONFIG = {
    'data': {
        'base_dir': 'data',
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'cohort_file': 'data/cohort.csv',
        'image_size': 'auto',
        'num_patients': 5000,
        'resize': {'enabled': True, 'min_size': 64, 'max_size': 512, 'preserve_aspect': True},
        'normalization': {'method': 'auto', 'clip_percentile': 1},
        'modality': 'auto',
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
        'num_iterations': 'auto',
        'batch_size': 'auto',
        'learning_rate': 'auto',
        'lr_decay': 0.1,
        'decay_steps': [8000],
        'seed': 42,
        'eval_checkpoint_every': 500,
        'snapshot_iterations': [100, 500, 1000, 2500, 5000, 10000],
    },
    'evaluation': {
        'edge_threshold': 0.1,
        'edge_band_width': 'auto',
        'significance_level': 0.05,
        'correction_method': 'bonferroni',
    },
    'experiment': {
        'device': 'auto',
        'num_workers': 4,
        'save_reconstructions': True,
        'generate_visualizations': True,
    },
}


# ============================================================================
# EMBEDDED UTILITIES (replaces src/utils.py)
# ============================================================================

def load_config(path=None):
    """Load config from YAML file or return embedded default."""
    if path is not None and Path(path).exists():
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    return copy.deepcopy(DEFAULT_CONFIG)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


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
# EMBEDDED ENCODINGS (replaces src/encodings.py)
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
        mask1 = z_norm <= 0.25
        result[mask1] = 1.0
        mask2 = (z_norm > 0.25) & (z_norm <= 0.75)
        result[mask2] = 1.0 - 4.0 * (z_norm[mask2] - 0.25)
        mask3 = z_norm > 0.75
        result[mask3] = -1.0
        return result

    @staticmethod
    def trapezoid_sin(z):
        z_shifted = z - np.pi / 2
        z_norm = (z_shifted % (2 * np.pi)) / (2 * np.pi)
        result = torch.zeros_like(z_norm)
        mask1 = z_norm <= 0.25
        result[mask1] = 1.0
        mask2 = (z_norm > 0.25) & (z_norm <= 0.75)
        result[mask2] = 1.0 - 4.0 * (z_norm[mask2] - 0.25)
        mask3 = z_norm > 0.75
        result[mask3] = -1.0
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

    @staticmethod
    def square_cos(z):
        z_norm = (z % (2 * np.pi)) / (2 * np.pi)
        return torch.where(z_norm < 0.5, torch.ones_like(z_norm), -torch.ones_like(z_norm))

    @staticmethod
    def square_sin(z):
        z_shifted = z - np.pi / 2
        z_norm = (z_shifted % (2 * np.pi)) / (2 * np.pi)
        return torch.where(z_norm < 0.5, torch.ones_like(z_norm), -torch.ones_like(z_norm))

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([self.square_cos(x_proj), self.square_sin(x_proj)], dim=-1)


class RampupFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    @staticmethod
    def rampup_cos(z):
        z_norm = (z % (2 * np.pi)) / (2 * np.pi)
        return 2 * z_norm - 1

    @staticmethod
    def rampup_sin(z):
        z_shifted = z - np.pi / 2
        z_norm = (z_shifted % (2 * np.pi)) / (2 * np.pi)
        return 2 * z_norm - 1

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([self.rampup_cos(x_proj), self.rampup_sin(x_proj)], dim=-1)


class RampdownFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        self.out_dim = num_frequencies * 2
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    @staticmethod
    def rampdown_cos(z):
        z_norm = (z % (2 * np.pi)) / (2 * np.pi)
        return 1 - 2 * z_norm

    @staticmethod
    def rampdown_sin(z):
        z_shifted = z - np.pi / 2
        z_norm = (z_shifted % (2 * np.pi)) / (2 * np.pi)
        return 1 - 2 * z_norm

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([self.rampdown_cos(x_proj), self.rampdown_sin(x_proj)], dim=-1)


class NoEncoding(nn.Module):
    def __init__(self, in_dim=2, **kwargs):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x):
        return x


class SIRENEncoding(nn.Module):
    """SIREN-inspired positional encoding. Uses omega_0 scaling with
    SIREN-style initialization, but feeds into standard ReLU MLP."""
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
    """Multi-resolution hash encoding from Müller et al. 2022 (Instant-NGP).

    For image fitting the coordinate grid is fixed, so hash indices and
    bilinear weights are pre-computed once via precompute(coords) before the
    training loop.  Each forward pass then does only table lookups +
    interpolation — skipping all floor/XOR/modulo arithmetic.
    Gradient flows to self.hash_table exactly as in the original formulation.
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

        # Pre-computed lookup tables (set by precompute(); None = not yet cached)
        self._cached_idx = None       # (N, L, 4)  long, no grad
        self._cached_weights = None   # (N, L, 4)  float, no grad

    # ------------------------------------------------------------------
    # Pre-computation (call once per image before training)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def precompute(self, x):
        """Cache hash indices and bilinear weights for a fixed coordinate set.

        XOR arithmetic runs on CPU regardless of device — MPS does not support
        int64 bitwise ops.  Result indices fit in int32 (hashmap_size ≤ 2^19)
        and are moved to x.device for fast gather in the training loop.
        """
        device = x.device
        N = x.shape[0]
        L = self.n_levels
        # bilinear weights: computed on device (float ops are fine everywhere)
        x_unit   = (x + 1.0) / 2.0
        x_scaled = x_unit.unsqueeze(1) * self.resolutions.to(device).view(1, L, 1)
        x_floor_f = x_scaled.floor()
        w  = x_scaled - x_floor_f
        w0, w1 = w[..., 0:1], w[..., 1:2]
        weights = torch.cat([
            (1-w0)*(1-w1), w0*(1-w1), (1-w0)*w1, w0*w1,
        ], dim=-1).contiguous()                                        # (N,L,4) on device

        # hash index arithmetic: done on CPU (int64 XOR not supported on MPS)
        x_floor = x_floor_f.long().cpu()
        x_ceil  = x_floor + 1
        primes  = self.primes.cpu()
        corners = torch.stack([
            torch.stack([x_floor[..., 0], x_floor[..., 1]], dim=-1),
            torch.stack([x_ceil[..., 0],  x_floor[..., 1]], dim=-1),
            torch.stack([x_floor[..., 0], x_ceil[..., 1]],  dim=-1),
            torch.stack([x_ceil[..., 0],  x_ceil[..., 1]],  dim=-1),
        ], dim=2)                                                       # (N,L,4,2) on CPU
        idx = (corners[..., 0] * primes[0]) ^ (corners[..., 1] * primes[1])
        idx = (idx % self.hashmap_size).to(device)                     # move to device

        self._cached_idx     = idx.contiguous()
        self._cached_weights = weights

    def clear_cache(self):
        """Release pre-computed buffers (call after training an image)."""
        self._cached_idx = None
        self._cached_weights = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        T = self._cached_idx.shape[0] if self._cached_idx is not None else -1
        if self._cached_idx is not None and x.shape[0] == T:
            return self._forward_cached(T)
        return self._forward_dynamic(x)

    def _forward_cached(self, N):
        """Fast path: indices and weights pre-computed, only table lookup here."""
        L = self.n_levels
        F = self.features_per_level
        idx     = self._cached_idx      # (N, L, 4)  — no grad
        weights = self._cached_weights  # (N, L, 4)  — no grad

        # Index into hash_table (L, H, F) for all levels and corners at once.
        # hash_table has grad; indexing creates a differentiable path via autograd.
        # Expand idx to (N, L, 4, F) for gathering along the hashmap dimension.
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, F)            # (N,L,4,F)
        table_exp = self.hash_table.unsqueeze(0).expand(N, -1, -1, -1)  # (N,L,H,F)
        corner_features = table_exp.gather(2, idx_exp)               # (N,L,4,F)

        level_features = (weights.unsqueeze(-1) * corner_features).sum(dim=2)  # (N,L,F)
        return level_features.reshape(N, -1)

    def _forward_dynamic(self, x):
        """Fallback path: compute indices on-the-fly (used without precompute).
        XOR runs on CPU for MPS compatibility; everything else stays on device.
        """
        device = x.device
        N = x.shape[0]
        L = self.n_levels
        F = self.features_per_level
        x_unit   = (x + 1.0) / 2.0
        x_scaled = x_unit.unsqueeze(1) * self.resolutions.view(1, L, 1)
        x_floor_f = x_scaled.floor()
        w  = x_scaled - x_floor_f
        w0, w1 = w[..., 0:1], w[..., 1:2]
        corner_weights = torch.cat([
            (1-w0)*(1-w1), w0*(1-w1), (1-w0)*w1, w0*w1,
        ], dim=-1)                                                      # on device
        # hash on CPU (MPS int64 XOR unsupported)
        x_floor = x_floor_f.long().cpu()
        x_ceil  = x_floor + 1
        primes  = self.primes.cpu()
        corners = torch.stack([
            torch.stack([x_floor[..., 0], x_floor[..., 1]], dim=-1),
            torch.stack([x_ceil[..., 0],  x_floor[..., 1]], dim=-1),
            torch.stack([x_floor[..., 0], x_ceil[..., 1]],  dim=-1),
            torch.stack([x_ceil[..., 0],  x_ceil[..., 1]],  dim=-1),
        ], dim=2)
        idx = ((corners[..., 0] * primes[0]) ^ (corners[..., 1] * primes[1]))
        idx = (idx % self.hashmap_size).to(device)
        idx_exp   = idx.unsqueeze(-1).expand(-1, -1, -1, F)
        table_exp = self.hash_table.unsqueeze(0).expand(N, -1, -1, -1)
        corner_features = table_exp.gather(2, idx_exp)
        level_features  = (corner_weights.unsqueeze(-1) * corner_features).sum(dim=2)
        return level_features.reshape(N, -1)



class WaveletEncoding(nn.Module):
    """Multi-scale wavelet positional encoding using Mexican hat wavelets."""
    def __init__(self, in_dim=2, num_frequencies=256, sigma=10.0, n_scales=8):
        super().__init__()
        self.in_dim = in_dim
        self.n_scales = n_scales
        n_dirs = num_frequencies // n_scales
        self.n_dirs = n_dirs
        self.out_dim = n_scales * n_dirs * 2

        B = torch.randn(in_dim, n_dirs) * sigma
        self.register_buffer('B', B)
        scales = torch.logspace(-1, 1, n_scales)
        self.register_buffer('scales', scales)

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
# EMBEDDED MODEL (replaces src/models.py)
# ============================================================================

class INRMLP(nn.Module):
    """Standard INR MLP — the ONLY model in this pipeline.
    Architecture: encoding -> Linear -> ReLU -> ... -> Linear -> output
    With optional skip connection at the midpoint.
    """
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

    def get_first_layer_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                return layer.weight.detach().cpu().numpy()
        return None


# ============================================================================
# EMBEDDED TRAINING (replaces src/train.py)
# ============================================================================

def train_inr(model, image, config, verbose=True):
    """Train an Implicit Neural Representation on a single image.

    Returns:
        reconstruction: (H, W) array
        metrics: dict with PSNR, SSIM, etc.
        training_info: dict with loss history, PSNR over time, snapshots
    """
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

    # Pre-compute hash indices and bilinear weights once for HashEncoding.
    # Coordinates are fixed (same image grid every iteration), so this avoids
    # repeating floor/XOR/modulo arithmetic 3500× per image.
    # Mathematical formulation (Müller et al. 2022) is unchanged.
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
    snapshot_iters = set(config['training'].get('snapshot_iterations',
                                                [100, 500, 1000, 2500, 5000, 10000]))

    loss_history = []
    psnr_over_time = []
    ssim_over_time = []
    mse_over_time = []
    lr_history = []
    reconstruction_snapshots = {}
    # Full metrics (PSNR, SSIM, edge_psnr, flat_psnr, ringing) at each snapshot
    all_metrics_over_time = {}

    from skimage.metrics import structural_similarity

    pbar = range(num_iterations)
    if verbose:
        pbar = tqdm(pbar, desc="Training INR")

    for iteration in pbar:
        model.train()
        indices = torch.randint(0, num_pixels, (batch_size,))
        batch_coords = coords[indices]
        batch_target = target[indices]

        optimizer.zero_grad()
        pred = model(batch_coords)
        loss = criterion(pred, batch_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append((iteration, current_lr))

        if verbose and iteration % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

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
            edge_m = _compute_edge_metrics(image, recon_full, config)

            psnr_over_time.append((iteration, psnr_val))
            ssim_over_time.append((iteration, ssim_val))
            mse_over_time.append((iteration, mse_val))

            # store full snapshot metrics for checkpoint reporting
            all_metrics_over_time[iteration] = {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'mse': mse_val,
                'edge_psnr': edge_m.get('edge_psnr', 0.0),
                'flat_psnr': edge_m.get('flat_psnr', 0.0),
                'ringing_score': edge_m.get('ringing_score', 0.0),
            }

            if is_snapshot:
                reconstruction_snapshots[iteration] = recon_full.copy()

    model.eval()
    with torch.no_grad():
        reconstruction = model(coords).cpu().numpy().reshape(H, W)
        reconstruction = np.clip(reconstruction, 0, 1)

    # Release pre-computed hash cache — frees GPU memory between images
    if hasattr(model.encoding, 'clear_cache'):
        model.encoding.clear_cache()

    metrics = _compute_all_metrics(image, reconstruction, config)

    training_info = {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else 0,
        'num_iterations': num_iterations,
        'psnr_over_time': psnr_over_time,
        'ssim_over_time': ssim_over_time,
        'mse_over_time': mse_over_time,
        'lr_history': lr_history,
        'reconstruction_snapshots': reconstruction_snapshots,
        'all_metrics_over_time': all_metrics_over_time,
    }

    return reconstruction, metrics, training_info


def _compute_all_metrics(original, reconstruction, config):
    """Compute image quality metrics."""
    mse = np.mean((original - reconstruction) ** 2)
    psnr = 100 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

    from skimage.metrics import structural_similarity
    ssim = structural_similarity(original, reconstruction, data_range=1.0)

    edge_metrics = _compute_edge_metrics(original, reconstruction, config)
    spectral_mse = _compute_spectral_mse(original, reconstruction)

    return {
        'psnr': float(psnr), 'ssim': float(ssim), 'mse': float(mse),
        'spectral_mse': float(spectral_mse), **edge_metrics
    }


def _compute_spectral_mse(original, reconstruction):
    try:
        fft_orig = np.fft.fft2(original)
        fft_recon = np.fft.fft2(reconstruction)
        return float(np.mean(np.abs(fft_orig - fft_recon) ** 2))
    except Exception:
        return 0.0


def _compute_edge_metrics(original, reconstruction, config):
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

    if edge_band.sum() > 0:
        edge_mse = np.mean((original[edge_band] - reconstruction[edge_band]) ** 2)
        edge_psnr = 20 * np.log10(1.0 / np.sqrt(edge_mse + 1e-10))
    else:
        edge_psnr = 0

    flat_mask = ~edge_band
    if flat_mask.sum() > 0:
        flat_mse = np.mean((original[flat_mask] - reconstruction[flat_mask]) ** 2)
        flat_psnr = 20 * np.log10(1.0 / np.sqrt(flat_mse + 1e-10))
    else:
        flat_psnr = 0

    residual = np.abs(original - reconstruction)
    ringing_score = np.std(residual[edge_mask]) if edge_mask.sum() > 0 else 0.0

    return {
        'edge_psnr': float(edge_psnr),
        'flat_psnr': float(flat_psnr),
        'ringing_score': float(ringing_score),
    }


# ============================================================================
# EMBEDDED DATASET HELPERS (replaces src/datasets.py)
# ============================================================================

def apply_hu_window(ct, center=40, width=400):
    low, high = center - width / 2, center + width / 2
    return (np.clip(ct, low, high) - low) / (high - low)


# ============================================================================
# CONSTANTS
# ============================================================================

MICCAI_IMAGE_SIZE = 256
PSNR_CAP = 50.0  # used only as reference label; analysis runs at 50, 100, and uncapped
MIN_SAMPLES_PER_MODALITY = 10
RECOMMENDED_PER_MODALITY = 30
SEED = 42
DPI = 600

CONVERGENCE_CHECKPOINTS = [1000, 2000, 2500, 3000, 3500]

CONVERGENCE_THRESHOLDS = {'CT': 30.0, 'MRI': 28.0, 'XRay': 28.0,
                          'Ultrasound': 25.0, 'Pathology': 32.0, 'default': 28.0}

MAIN_ENCODINGS = ['none', 'fourier', 'tropical', 'hybrid',
                  'trapezoid', 'square', 'rampup', 'rampdown']
BASELINE_ENCODINGS = ['siren',  'wavelet', 'hash']

ENC_COLORS = {
    'none': '#888888', 'fourier': '#4dabf7', 'tropical': '#51cf66',
    'hybrid': '#cc5de8', 'trapezoid': '#ffd43b', 'square': '#f06595',
    'rampup': '#69db7c', 'rampdown': '#ff8787',
    'siren': '#845ef7', 'wavelet': '#20c997', 'hash': '#ff922b', 
    
}
ENC_MARKERS = {
    'none': 'x', 'fourier': 's', 'tropical': 'o', 'hybrid': '^',
    'trapezoid': 'v', 'square': 'D', 'rampup': '>', 'rampdown': '<',
    'siren': 'p',  'wavelet': '*', 'hash': 'h',
}

BASELINE_CITATIONS = {
    'siren': r'Sitzmann et al.\ \cite{sitzmann2020siren}',
    'hash': r'M\"uller et al.\ \cite{mueller2022instant}',
    'wavelet': r'Fathony et al.\ \cite{fathony2021wavelet}',
}


# ============================================================================
# HYBRID ENCODING
# ============================================================================

class HybridFeatures(nn.Module):
    """Concatenation of Fourier and Tropical features (equal split)."""
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
# PHASE 1: PREPROCESSING
# ============================================================================

def preprocess_all(config, output_dir):
    """Preprocess all images to standardized 256x256 grayscale [0,1]."""
    from skimage.transform import resize as sk_resize

    raw_dir = Path(config['data']['raw_dir'])
    proc_dir = ensure_dir(output_dir / 'processed')

    extensions = ['*.nii.gz', '*.nii', '*.dcm', '*.dicom',
                  '*.svs', '*.png', '*.PNG',
                  '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
                  '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.bmp', '*.BMP', '*.npy']

    all_files = []
    for ext in extensions:
        all_files.extend(raw_dir.rglob(ext))

    dcm_folders = set()
    non_dcm = []
    for f in all_files:
        if f.suffix.lower() in ['.dcm', '.dicom']:
            dcm_folders.add(f.parent)
        else:
            non_dcm.append(f)

    print(f"Found {len(non_dcm)} non-DICOM files + {len(dcm_folders)} DICOM series")

    cohort = []
    for dcm_folder in sorted(dcm_folders):
        results = _process_dicom_series(dcm_folder, config, proc_dir)
        if results:
            cohort.extend(results)

    for fpath in sorted(non_dcm):
        results = _process_file(fpath, config, proc_dir)
        if results:
            if isinstance(results, list):
                cohort.extend(results)
            else:
                cohort.append(results)

    cohort_df = pd.DataFrame(cohort)

    # Assign 80/20 train/test split stratified per modality with fixed seed
    rng = np.random.default_rng(42)
    split_col = []
    for _, grp in cohort_df.groupby('modality'):
        pids = grp['patient_id'].tolist()
        pids_sorted = sorted(pids)
        rng_local = np.random.default_rng(42)
        idx = rng_local.permutation(len(pids_sorted))
        n_test = max(1, round(len(pids_sorted) * 0.2))
        test_pids = set(pids_sorted[i] for i in idx[:n_test])
        split_col.extend(['test' if pid in test_pids else 'train'
                          for pid in grp['patient_id']])
    cohort_df['split'] = split_col

    cohort_path = output_dir / 'cohort.csv'
    cohort_df.to_csv(cohort_path, index=False)

    n_train = len(cohort_df[cohort_df['split'] == 'train']) if 'split' in cohort_df.columns else len(cohort_df)
    n_test = len(cohort_df[cohort_df['split'] == 'test']) if 'split' in cohort_df.columns else 0
    print(f"\nPreprocessed {len(cohort_df)} slices from "
          f"{cohort_df['patient_id'].nunique()} patients")
    print(f"  Train split: {n_train} slices (rank-1)")
    print(f"  Test split:  {n_test} slices (rank-1, held-out patients 20%)")

    if 'modality' in cohort_df.columns:
        print("\nModality breakdown:")
        for mod in sorted(cohort_df['modality'].unique()):
            n_total = len(cohort_df[cohort_df['modality'] == mod])
            n_tr = len(cohort_df[(cohort_df['modality'] == mod) & (cohort_df['split'] == 'train')])
            status = ""
            if n_tr < MIN_SAMPLES_PER_MODALITY:
                status = f" *** BELOW MINIMUM ({MIN_SAMPLES_PER_MODALITY})"
            elif n_tr < RECOMMENDED_PER_MODALITY:
                status = f" (recommend {RECOMMENDED_PER_MODALITY}+)"
            print(f"  {mod}: {n_total} total ({n_tr} train, {n_total - n_tr} test){status}")

    return cohort_df


def _detect_modality(file_path):
    path_str = str(file_path).lower()
    suffix = file_path.suffix.lower()

    if suffix in ['.nii', '.gz'] or str(file_path).endswith('.nii.gz'):
        if any(k in path_str for k in ['mri', 't1w', 't2w', 'flair']):
            return 'MRI'
        return 'CT'
    if suffix in ['.dcm', '.dicom']:
        if any(k in path_str for k in ['ct', 'chest_ct', 'head_ct']):
            return 'CT'
        return 'MRI'
    if suffix in ['.svs', '.ndpi', '.mrxs']:
        return 'Pathology'

    if any(k in path_str for k in ['patholog', 'histol', 'wsi', 'biopsy',
                                    'stain', 'tumor', 'tcga', 'hcm']):
        return 'Pathology'
    if any(k in path_str for k in ['ultrasound', 'ultra', 'sonograph',
                                    'benign', 'malignant']):
        return 'Ultrasound'
    if any(k in path_str for k in ['xray', 'x-ray', 'x_ray', 'cxr',
                                    'radiograph', 'osteoporosis']):
        return 'XRay'
    if any(k in path_str for k in ['mri', 'brain', 'gbm']):
        return 'MRI'
    if any(k in path_str for k in ['ct_', 'ct/', '_ct']):
        return 'CT'
    if any(k in path_str for k in ['pet', 'fdg']):
        return 'PET'

    if suffix in ['.jpg', '.jpeg']:
        return 'XRay'
    if suffix in ['.png']:
        return 'Ultrasound'

    return 'Unknown'


def _load_and_standardize(image_2d):
    from skimage.transform import resize as sk_resize

    if image_2d.ndim == 3:
        image_2d = 0.299 * image_2d[:, :, 0] + 0.587 * image_2d[:, :, 1] + 0.114 * image_2d[:, :, 2]

    image_2d = np.nan_to_num(image_2d, nan=0.0, posinf=1.0, neginf=0.0)

    vmin, vmax = np.percentile(image_2d, [1, 99])
    if vmax - vmin > 1e-8:
        image_2d = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    else:
        image_2d = np.clip(image_2d, 0, 1)

    image_2d = sk_resize(image_2d, (MICCAI_IMAGE_SIZE, MICCAI_IMAGE_SIZE),
                         anti_aliasing=True, preserve_range=True).astype(np.float32)
    return image_2d


def _compute_edge_score(image):
    from skimage.filters import sobel
    return float(np.mean(sobel(image)))


def _process_file(fpath, config, proc_dir):
    try:
        suffix = fpath.suffix.lower()
        modality = _detect_modality(fpath)

        if suffix in ['.nii', '.gz'] or str(fpath).endswith('.nii.gz'):
            import nibabel as nib
            vol = nib.load(str(fpath)).get_fdata()
            return _extract_slices_from_volume(vol, fpath, modality, config, proc_dir)

        if suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            from PIL import Image
            img = Image.open(str(fpath))
            arr = np.array(img, dtype=np.float32)
            if arr.max() > 1:
                arr = arr / 255.0
            image = _load_and_standardize(arr)
            if image.std() < 0.001:
                return None
            patient_id = fpath.stem
            out_name = f"{patient_id}_rank1.npy"
            np.save(proc_dir / out_name, image)
            split = 'test' if (int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 10) < 2 else 'train'
            return {
                'patient_id': patient_id, 'modality': modality,
                'output_file': out_name, 'edge_score': _compute_edge_score(image),
                'shape': str(image.shape), 'rank': 1, 'split': split,
            }

        if suffix in ['.svs', '.ndpi', '.mrxs']:
            try:
                from openslide import OpenSlide
                slide = OpenSlide(str(fpath))
                level = min(2, slide.level_count - 1)
                w, h = slide.level_dimensions[level]
                region = slide.read_region((0, 0), level, (w, h))
                arr = np.array(region.convert('RGB'), dtype=np.float32) / 255.0
                slide.close()
            except Exception:
                from PIL import Image
                arr = np.array(Image.open(str(fpath)).convert('RGB'), dtype=np.float32) / 255.0

            image = _load_and_standardize(arr)
            if image.std() < 0.001:
                return None
            patient_id = fpath.stem
            out_name = f"{patient_id}_rank1.npy"
            np.save(proc_dir / out_name, image)
            split = 'test' if (int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 10) < 2 else 'train'
            return {
                'patient_id': patient_id, 'modality': 'Pathology',
                'output_file': out_name, 'edge_score': _compute_edge_score(image),
                'shape': str(image.shape), 'rank': 1, 'split': split,
            }

        if suffix in ['.npy', '.npz']:
            data = np.load(str(fpath))
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data[list(data.keys())[0]]
            if data.ndim == 3 and data.shape[2] >= 3 and data.shape[2] != 3:
                return _extract_slices_from_volume(data, fpath, modality, config, proc_dir)
            if data.ndim == 3 and data.shape[2] == 3:
                data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
            image = _load_and_standardize(data.astype(np.float32))
            if image.std() < 0.001:
                return None
            patient_id = fpath.stem
            out_name = f"{patient_id}_rank1.npy"
            np.save(proc_dir / out_name, image)
            split = 'test' if (int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 10) < 2 else 'train'
            return {
                'patient_id': patient_id, 'modality': modality,
                'output_file': out_name, 'edge_score': _compute_edge_score(image),
                'shape': str(image.shape), 'rank': 1, 'split': split,
            }

    except Exception as e:
        print(f"  Skip {fpath.name}: {e}")
    return None


def _extract_slices_from_volume(volume, fpath, modality, config, proc_dir):
    if volume.ndim != 3 or volume.shape[2] < 3:
        return None

    patient_id = fpath.stem.replace('.nii', '').replace('.gz', '')

    if modality == 'CT':
        try:
            volume = apply_hu_window(volume, center=40, width=400)
        except Exception:
            pass

    n_slices = volume.shape[2]
    candidates = np.linspace(int(0.2 * n_slices), int(0.8 * n_slices),
                             num=min(30, n_slices)).astype(int)

    scored_slices = []
    for z in candidates:
        sl = volume[:, :, z].astype(np.float32)
        sl = _load_and_standardize(sl)
        if sl.std() < 0.005:
            continue
        score = _compute_edge_score(sl)
        scored_slices.append((score, z, sl))

    if not scored_slices:
        return None

    scored_slices.sort(key=lambda x: x[0], reverse=True)
    score, z, sl = scored_slices[0]

    split = 'test' if (int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 10) < 2 else 'train'
    out_name = f"{patient_id}_z{z:03d}_rank1.npy"
    np.save(proc_dir / out_name, sl)
    return [{
        'patient_id': patient_id, 'modality': modality,
        'output_file': out_name, 'edge_score': score,
        'slice_index': z, 'shape': str(sl.shape),
        'rank': 1, 'split': split,
    }]


def _process_dicom_series(dcm_folder, config, proc_dir):
    try:
        import pydicom
    except ImportError:
        return None

    dcm_files = sorted(dcm_folder.glob('*.dcm')) + sorted(dcm_folder.glob('*.dicom'))
    if len(dcm_files) < 2:
        return None

    slices = []
    for dcm_path in dcm_files:
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            arr = dcm.pixel_array.astype(np.float32)
            if arr.ndim != 2:
                continue
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                arr = arr * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
            sort_key = 0
            if hasattr(dcm, 'InstanceNumber') and dcm.InstanceNumber is not None:
                sort_key = int(dcm.InstanceNumber)
            slices.append((sort_key, arr))
        except Exception:
            continue

    if len(slices) < 2:
        return None

    slices.sort(key=lambda x: x[0])
    from collections import Counter
    shapes = Counter(a.shape for _, a in slices)
    common_shape = shapes.most_common(1)[0][0]
    arrays = [a for _, a in slices if a.shape == common_shape]
    volume = np.stack(arrays, axis=2)

    modality = _detect_modality(dcm_folder)
    return _extract_slices_from_volume(volume, dcm_folder, modality, config, proc_dir)


# ============================================================================
# PHASE 2: TRAINING (with convergence tracking)
# ============================================================================

def cap_psnr(val):
    return min(float(val), PSNR_CAP)


def apply_psnr_cap(df, cap):
    """Return a copy of df with PSNR columns at the requested cap level.

    psnr/edge_psnr/flat_psnr are stored already capped at PSNR_CAP (50 dB).
    For cap > 50 or uncapped analysis, *_raw columns are used instead.
    """
    df = df.copy()
    if cap == PSNR_CAP:
        return df  # already stored capped at 50
    for base in ['psnr', 'edge_psnr', 'flat_psnr']:
        raw_col = f'{base}_raw'
        if raw_col in df.columns:
            df[base] = df[raw_col].clip(upper=cap) if cap is not None else df[raw_col]
    return df


def run_training(cohort_df, config, encodings, output_dir, resume=False, split='train'):
    from tqdm import tqdm

    proc_dir = output_dir / 'processed'
    recon_dir = ensure_dir(output_dir / f'reconstructions_{split}')
    metrics_dir = ensure_dir(output_dir / 'metrics')
    convergence_dir = ensure_dir(output_dir / 'convergence')

    split_df = cohort_df[cohort_df['split'] == split] if 'split' in cohort_df.columns else cohort_df
    if len(split_df) == 0:
        print(f"  No slices for split={split}, skipping.")
        return pd.DataFrame()

    results_file = metrics_dir / f'results_{split}.csv'
    completed = set()
    existing = pd.DataFrame()
    if resume and results_file.exists():
        existing = pd.read_csv(results_file)
        completed = set(zip(existing['slice_name'], existing['encoding']))
        print(f"Resuming {split}: {len(completed)} experiments already done")

    all_results = []
    total = len(split_df) * len(encodings)
    pbar = tqdm(total=total, desc=f"Training ({split})")

    for _, row in split_df.iterrows():
        slice_path = proc_dir / row['output_file']
        if not slice_path.exists():
            pbar.update(len(encodings))
            continue

        image = np.load(slice_path).astype(np.float32)
        sname = slice_path.stem

        for enc_name in encodings:
            if (sname, enc_name) in completed:
                pbar.update(1)
                continue

            pbar.set_postfix(patient=row['patient_id'], enc=enc_name)

            try:
                set_seed(SEED)
                cfg = copy.deepcopy(config)
                model = _patched_create_model(cfg, enc_name)

                t_start = time.time()
                recon, metrics, tinfo = train_inr(model, image, cfg, verbose=False)
                train_time_sec = time.time() - t_start

                np.save(recon_dir / f"{sname}_{enc_name}.npy", recon)

                psnr_curve = {}
                if 'psnr_over_time' in tinfo and tinfo['psnr_over_time']:
                    for it, psnr_val in tinfo['psnr_over_time']:
                        psnr_curve[int(it)] = cap_psnr(psnr_val)

                # full metrics at each checkpoint iteration
                full_cp_metrics = tinfo.get('all_metrics_over_time', {})

                conv_path = convergence_dir / f"{sname}_{enc_name}.json"
                with open(conv_path, 'w') as f:
                    json.dump({'psnr_curve': psnr_curve,
                               'full_metrics': {str(k): v for k, v in full_cp_metrics.items()},
                               'loss_history': tinfo.get('loss_history', [])[-10:]}, f)

                device = cfg['experiment']['device']
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
                    # find closest iteration at or before this checkpoint
                    best_it = None
                    for it_key in sorted(full_cp_metrics.keys()):
                        if it_key <= cp:
                            best_it = it_key
                    if best_it is not None:
                        cp_m = full_cp_metrics[best_it]
                        result[f'psnr_at_{cp}'] = cap_psnr(cp_m.get('psnr', np.nan))
                        result[f'ssim_at_{cp}'] = float(cp_m.get('ssim', np.nan))
                        result[f'edge_psnr_at_{cp}'] = float(cp_m.get('edge_psnr', np.nan))
                        result[f'flat_psnr_at_{cp}'] = float(cp_m.get('flat_psnr', np.nan))
                        result[f'ringing_at_{cp}'] = float(cp_m.get('ringing_score', np.nan))
                    else:
                        result[f'psnr_at_{cp}'] = np.nan
                        result[f'ssim_at_{cp}'] = np.nan
                        result[f'edge_psnr_at_{cp}'] = np.nan
                        result[f'flat_psnr_at_{cp}'] = np.nan
                        result[f'ringing_at_{cp}'] = np.nan

                all_results.append(result)

                if len(all_results) % 10 == 0:
                    _save_results(all_results, existing, results_file)

            except Exception as e:
                print(f"\n  Error {row['patient_id']}/{enc_name}: {e}")
                traceback.print_exc()

            pbar.update(1)

    pbar.close()
    df = _save_results(all_results, existing, results_file)
    print(f"\nTraining ({split}) complete: {len(df)} experiments")
    return df


def _save_results(new_results, existing, path):
    df = pd.DataFrame(new_results)
    if len(existing) > 0:
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


# ============================================================================
# PHASE 3: STATISTICAL EVALUATION
# ============================================================================

def run_evaluation(train_df, test_df, output_dir, cap=None):
    from scipy import stats

    train_df = apply_psnr_cap(train_df, cap)
    if test_df is not None and len(test_df) > 0:
        test_df = apply_psnr_cap(test_df, cap)

    metrics_dir = ensure_dir(output_dir / 'metrics')
    tables_dir = ensure_dir(output_dir / 'tables')

    metrics_cols = ['psnr', 'ssim', 'edge_psnr', 'flat_psnr', 'ringing_score']
    available = [m for m in metrics_cols if m in train_df.columns]

    # TABLE 1
    print("\n" + "=" * 70)
    print("TABLE 1: Summary Statistics (Train Split)")
    print("=" * 70)

    summary_rows = []
    for enc in sorted(train_df['encoding'].unique()):
        edf = train_df[train_df['encoding'] == enc]
        row = {'Encoding': enc, 'N': len(edf)}
        for m in available:
            vals = edf[m].values
            row[f'{m}_mean'] = np.mean(vals)
            row[f'{m}_std'] = np.std(vals)
            row[f'{m}_str'] = f"{np.mean(vals):.2f} $\\pm$ {np.std(vals):.2f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metrics_dir / 'table1_summary.csv', index=False)

    for _, row in summary_df.iterrows():
        parts = [f"{row['Encoding']:12s}"]
        for m in available:
            parts.append(f"{row[f'{m}_str']:>20s}")
        print("  ".join(parts))

    _generate_latex_table1(summary_df, available, tables_dir / 'table1.tex', cap=cap)

    # Per-modality
    if 'modality' in train_df.columns:
        print("\n" + "=" * 70)
        print("PER-MODALITY PSNR (Train Split)")
        print("=" * 70)

        mod_rows = []
        for mod in sorted(train_df['modality'].unique()):
            mdf = train_df[train_df['modality'] == mod]
            n = mdf['patient_id'].nunique()
            print(f"\n  {mod} (N={n} patients):")
            for enc in sorted(mdf['encoding'].unique()):
                vals = mdf[mdf['encoding'] == enc]['psnr'].values
                print(f"    {enc:15s}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")
                mod_rows.append({
                    'modality': mod, 'encoding': enc, 'n': len(vals),
                    'psnr_mean': np.mean(vals), 'psnr_std': np.std(vals),
                    'ssim_mean': np.mean(mdf[mdf['encoding'] == enc]['ssim'].values),
                })
        pd.DataFrame(mod_rows).to_csv(metrics_dir / 'per_modality_summary.csv', index=False)

    # TABLE 2
    print("\n" + "=" * 70)
    print("TABLE 2: Pairwise Comparisons (Wilcoxon signed-rank, Bonferroni)")
    print("=" * 70)

    key_pairs = [
        ('fourier', 'tropical'), ('fourier', 'hybrid'), ('fourier', 'siren'), 
        ('fourier', 'wavelet'), ('fourier', 'square'), ('fourier', 'rampup'), 
        ('fourier', 'rampdown'), ('fourier', 'trapezoid'), ('fourier', 'hash'),
        
        ('square', 'siren'), ('square', 'wavelet'), ('square', 'tropical'), 
        ('square', 'hybrid'), ('square', 'trapezoid'), ('square', 'rampup'), 
        ('square', 'rampdown'), ('square', 'hash'),
        
        ('tropical', 'hybrid'), ('tropical', 'siren'), ('tropical', 'wavelet'), 
        ('tropical', 'trapezoid'), ('tropical', 'rampup'), ('tropical', 'rampdown'), 
        ('tropical', 'hash'),
        
        ('trapezoid', 'hybrid'), ('trapezoid', 'siren'), ('trapezoid', 'wavelet'), 
        ('trapezoid', 'rampup'), ('trapezoid', 'rampdown'), ('trapezoid', 'hash'),
        
        ('rampup', 'siren'), ('rampup', 'wavelet'), ('rampup', 'hybrid'),
        ('rampup', 'rampdown'), ('rampup', 'hash'),
        
        ('rampdown', 'siren'), ('rampdown', 'wavelet'), ('rampdown', 'hybrid'),
        ('rampdown', 'hash'),
        
        ('hybrid', 'siren'), ('hybrid', 'wavelet'),('hybrid', 'hash'),
        
        ('siren', 'wavelet'), ('siren', 'hash'),
        
        ('hash', 'wavelet')
        
    ]

    sig_rows = []
    for metric in ['psnr', 'ssim', 'mse', 'edge_psnr', 'ringing_score']:
        if metric not in train_df.columns:
            continue
        for enc_a, enc_b in key_pairs:
            if enc_a not in train_df['encoding'].values or enc_b not in train_df['encoding'].values:
                continue
            df_a = train_df[train_df['encoding'] == enc_a].set_index('slice_name')[metric]
            df_b = train_df[train_df['encoding'] == enc_b].set_index('slice_name')[metric]
            common = df_a.index.intersection(df_b.index)
            if len(common) < 5:
                continue
            vals_a = df_a.loc[common].values
            vals_b = df_b.loc[common].values
            try:
                stat, p_val = stats.wilcoxon(vals_a, vals_b)
            except Exception:
                stat, p_val = np.nan, np.nan
            diff = vals_a - vals_b
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            sig_rows.append({
                'metric': metric, 'encoding_a': enc_a, 'encoding_b': enc_b,
                'mean_a': np.mean(vals_a), 'mean_b': np.mean(vals_b),
                'mean_diff': np.mean(diff), 'p_value': p_val,
                'cohens_d': d, 'n_pairs': len(common),
            })

    sig_df = pd.DataFrame(sig_rows)
    if len(sig_df) > 0:
        n_tests = len(sig_df)
        sig_df['p_adjusted'] = np.minimum(sig_df['p_value'] * n_tests, 1.0)
        sig_df['significant'] = sig_df['p_adjusted'] < 0.05
        sig_df.to_csv(metrics_dir / 'table2_significance.csv', index=False)

        for _, row in sig_df.iterrows():
            star = "*" if row['significant'] else ""
            print(f"  {row['metric']:15s} | {row['encoding_a']:10s} vs {row['encoding_b']:10s} | "
                  f"Delta={row['mean_diff']:+.3f} | p={row['p_adjusted']:.4f}{star} | d={row['cohens_d']:.2f}")

        _generate_latex_table2(sig_df, tables_dir / 'table2.tex')

    # TABLE 3
    print("\n" + "=" * 70)
    print("TABLE 3: Convergence Speed")
    print("=" * 70)
    conv_rows = _compute_convergence_table(train_df, metrics_dir)
    if conv_rows:
        _generate_latex_table3(conv_rows, tables_dir / 'table3.tex')

    # TABLE 4
    print("\n" + "=" * 70)
    print("TABLE 4: Model Efficiency")
    print("=" * 70)
    eff_rows = _compute_efficiency_table(train_df, metrics_dir)
    if eff_rows:
        _generate_latex_table4(eff_rows, tables_dir / 'table4.tex')

    # TABLE 5
    if test_df is not None and len(test_df) > 0:
        print("\n" + "=" * 70)
        print("TABLE 5: Held-Out Generalization (Test Split: held-out patients 20%)")
        print("=" * 70)

        gen_rows = []
        for enc in sorted(test_df['encoding'].unique()):
            test_vals = test_df[test_df['encoding'] == enc]['psnr'].values
            train_vals = train_df[train_df['encoding'] == enc]['psnr'].values
            if len(test_vals) > 0 and len(train_vals) > 0:
                delta = np.mean(test_vals) - np.mean(train_vals)
                print(f"  {enc:15s}: train={np.mean(train_vals):.2f}  test={np.mean(test_vals):.2f}  "
                      f"delta={delta:+.2f} dB")
                gen_rows.append({
                    'encoding': enc,
                    'train_psnr': np.mean(train_vals), 'test_psnr': np.mean(test_vals),
                    'delta_psnr': delta,
                    'train_ssim': np.mean(train_df[train_df['encoding'] == enc]['ssim'].values),
                    'test_ssim': np.mean(test_df[test_df['encoding'] == enc]['ssim'].values),
                    'n_train': len(train_vals), 'n_test': len(test_vals),
                })

        if gen_rows:
            gen_df = pd.DataFrame(gen_rows)
            gen_df.to_csv(metrics_dir / 'held_out_generalization.csv', index=False)
            _generate_latex_table_heldout(gen_df, tables_dir / 'table5_heldout.tex')

    # Best encoding per metric
    print("\n" + "=" * 70)
    print("BEST ENCODING PER METRIC")
    print("=" * 70)
    for metric in available:
        means = train_df.groupby('encoding')[metric].mean()
        best = means.idxmin() if metric == 'ringing_score' else means.idxmax()
        print(f"  {metric:15s}: {best} ({means[best]:.3f})")

    # Patient-level wins
    if 'hybrid' in train_df['encoding'].values and 'fourier' in train_df['encoding'].values:
        print("\n" + "=" * 70)
        print("PATIENT-LEVEL WINS (Hybrid vs Fourier, Train Split)")
        print("=" * 70)
        wins = {'hybrid': 0, 'fourier': 0, 'tie': 0}
        for sname in train_df['slice_name'].unique():
            pdf = train_df[train_df['slice_name'] == sname]
            h = pdf[pdf['encoding'] == 'hybrid']['psnr'].values
            f = pdf[pdf['encoding'] == 'fourier']['psnr'].values
            if len(h) > 0 and len(f) > 0:
                if h[0] > f[0] + 0.1:
                    wins['hybrid'] += 1
                elif f[0] > h[0] + 0.1:
                    wins['fourier'] += 1
                else:
                    wins['tie'] += 1
        total = sum(wins.values())
        if total > 0:
            print(f"  Hybrid wins:  {wins['hybrid']:3d} ({100*wins['hybrid']/total:.1f}%)")
            print(f"  Fourier wins: {wins['fourier']:3d} ({100*wins['fourier']/total:.1f}%)")
            print(f"  Ties:         {wins['tie']:3d} ({100*wins['tie']/total:.1f}%)")

    return sig_df if len(sig_rows) > 0 else pd.DataFrame()


def _compute_convergence_table(train_df, metrics_dir):
    conv_cp_cols = sorted([c for c in train_df.columns if c.startswith('psnr_at_')])
    if not conv_cp_cols:
        print("  No convergence checkpoint data found.")
        return []

    rows = []
    for enc in sorted(train_df['encoding'].unique()):
        edf = train_df[train_df['encoding'] == enc]
        row = {'encoding': enc}
        for cp_col in conv_cp_cols:
            vals = edf[cp_col].dropna().values
            row[cp_col + '_mean'] = np.mean(vals) if len(vals) > 0 else np.nan

        iters_to_thresh = []
        for mod in edf['modality'].unique():
            thresh = CONVERGENCE_THRESHOLDS.get(mod, CONVERGENCE_THRESHOLDS['default'])
            mod_df = edf[edf['modality'] == mod]
            for _, srow in mod_df.iterrows():
                found_iter = None
                for cp_col in conv_cp_cols:
                    cp_iter = int(cp_col.split('_')[-1])
                    val = srow.get(cp_col, np.nan)
                    if pd.notna(val) and val >= thresh:
                        found_iter = cp_iter
                        break
                if found_iter is not None:
                    iters_to_thresh.append(found_iter)

        row['mean_iters_to_threshold'] = np.mean(iters_to_thresh) if iters_to_thresh else np.nan
        row['pct_reached_threshold'] = len(iters_to_thresh) / max(len(edf), 1) * 100
        rows.append(row)

        if not np.isnan(row['mean_iters_to_threshold']):
            print(f"  {enc:15s}: avg iters to threshold = {row['mean_iters_to_threshold']:.0f}, "
                  f"{row['pct_reached_threshold']:.0f}% reached")
        else:
            print(f"  {enc:15s}: threshold not reached")

    pd.DataFrame(rows).to_csv(metrics_dir / 'table3_convergence.csv', index=False)
    return rows


def _compute_efficiency_table(train_df, metrics_dir):
    rows = []
    for enc in sorted(train_df['encoding'].unique()):
        edf = train_df[train_df['encoding'] == enc]
        row = {
            'encoding': enc,
            'num_params': int(edf['num_params'].iloc[0]) if 'num_params' in edf.columns else 0,
            'encoding_dim': int(edf['encoding_dim'].iloc[0]) if 'encoding_dim' in edf.columns else 0,
            'mean_train_time': edf['train_time_sec'].mean() if 'train_time_sec' in edf.columns else 0,
            'mean_inference_time': edf['inference_time_sec'].mean() if 'inference_time_sec' in edf.columns else 0,
            'mean_psnr': edf['psnr'].mean(),
            'psnr_per_param': edf['psnr'].mean() / max(int(edf['num_params'].iloc[0]), 1) * 1e6 if 'num_params' in edf.columns else 0,
        }
        rows.append(row)
        print(f"  {enc:15s}: {row['num_params']:>8d} params | enc_dim={row['encoding_dim']:>4d} | "
              f"train={row['mean_train_time']:.1f}s | infer={row['mean_inference_time']*1000:.1f}ms | "
              f"PSNR={row['mean_psnr']:.2f}")

    pd.DataFrame(rows).to_csv(metrics_dir / 'table4_efficiency.csv', index=False)
    return rows


# ==== LaTeX generators ====

def _generate_latex_table1(summary_df, metrics, path, cap=None):
    nice_names = {
        'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'edge_psnr': 'Edge PSNR (dB)',
        'flat_psnr': 'Flat PSNR (dB)', 'ringing_score': 'Ringing',
    }
    cols = " & ".join([nice_names.get(m, m) for m in metrics])
    cap_str = f"PSNR capped at {cap}\\,dB." if cap is not None else "PSNR uncapped."
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Reconstruction quality across all modalities (mean $\\pm$ std, "
        f"train split). Best in \\textbf{{bold}}. {cap_str}}}",
        "\\label{tab:results}",
        f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}", "\\toprule",
        f"Encoding & {cols} \\\\", "\\midrule",
    ]
    best = {}
    for m in metrics:
        col = f'{m}_mean'
        best[m] = summary_df[col].idxmin() if m == 'ringing_score' else summary_df[col].idxmax()

    for idx, row in summary_df.iterrows():
        enc_name = row['Encoding'].replace('_', '\\_').title()
        if row['Encoding'] in BASELINE_CITATIONS:
            enc_name += f" {BASELINE_CITATIONS[row['Encoding']]}"
        cells = [enc_name]
        for m in metrics:
            val = row[f'{m}_str']
            if idx == best[m]:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  LaTeX Table 1 -> {path}")


def _generate_latex_table2(sig_df, path):
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Pairwise comparisons (Wilcoxon signed-rank, Bonferroni corrected). "
        "$^*$\\,significant at $p<0.05$.}",
        "\\label{tab:significance}",
        "\\begin{tabular}{llccccc}", "\\toprule",
        "Metric & Comparison & $\\Delta$ & $p_{adj}$ & Cohen's $d$ & $N$ & Sig. \\\\",
        "\\midrule",
    ]
    for _, row in sig_df.iterrows():
        star = "$^*$" if row['significant'] else ""
        lines.append(
            f"{row['metric']} & {row['encoding_a']} vs {row['encoding_b']} & "
            f"{row['mean_diff']:+.3f} & {row['p_adjusted']:.4f} & "
            f"{row['cohens_d']:.2f} & {row['n_pairs']} & {star} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  LaTeX Table 2 -> {path}")


def _generate_latex_table3(conv_rows, path):
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Convergence speed. Avg.\\ iterations to reach modality-specific "
        "PSNR threshold and percentage of images reaching threshold within budget.}",
        "\\label{tab:convergence}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Encoding & Iters to Threshold & \\% Reached & PSNR@5k \\\\", "\\midrule",
    ]
    for row in conv_rows:
        iters = f"{row['mean_iters_to_threshold']:.0f}" if not np.isnan(row['mean_iters_to_threshold']) else "---"
        psnr_5k = row.get('psnr_at_5000_mean', np.nan)
        psnr_str = f"{psnr_5k:.1f}" if not np.isnan(psnr_5k) else "---"
        lines.append(f"{row['encoding'].title()} & {iters} & {row['pct_reached_threshold']:.0f}\\% & {psnr_str} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  LaTeX Table 3 -> {path}")


def _generate_latex_table4(eff_rows, path):
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Model efficiency. All encodings use the same 4-layer MLP with "
        "256 hidden units. Encoding dimension and parameter count vary.}",
        "\\label{tab:efficiency}",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        "Encoding & Enc.~Dim & Params & Train (s) & Infer (ms) & PSNR & PSNR/M-param \\\\", "\\midrule",
    ]
    for row in eff_rows:
        lines.append(
            f"{row['encoding'].title()} & {row['encoding_dim']} & "
            f"{row['num_params']:,} & {row['mean_train_time']:.1f} & "
            f"{row['mean_inference_time']*1000:.1f} & "
            f"{row['mean_psnr']:.2f} & {row['psnr_per_param']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  LaTeX Table 4 -> {path}")


def _generate_latex_table_heldout(gen_df, path):
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Held-out generalization. 80/20 patient-level split: "
        "models trained on 80\\% of patients (rank-1 slice each); "
        "evaluated on 20\\% withheld patients (rank-1 slice each). "
        "$\\Delta$ = test $-$ train PSNR.}",
        "\\label{tab:heldout}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "Encoding & Train PSNR & Test PSNR & $\\Delta$ PSNR & Train SSIM & Test SSIM \\\\", "\\midrule",
    ]
    for _, row in gen_df.iterrows():
        lines.append(
            f"{row['encoding'].title()} & {row['train_psnr']:.2f} & {row['test_psnr']:.2f} & "
            f"{row['delta_psnr']:+.2f} & {row['train_ssim']:.3f} & {row['test_ssim']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  LaTeX Table 5 -> {path}")


# ============================================================================
# PHASE 4: PUBLICATION FIGURES
# ============================================================================
    # ---- Inner helpers (defined once, used by all panels below) ----

def _save_fig(fig, path):
    fig.savefig(str(path) + '.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(str(path) + '.pdf', bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    

def _fig_psnr_bars(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 1: PSNR bar chart...")
    per_mod = df.groupby(['modality', 'encoding'])['psnr'].mean().reset_index()
    balanced = per_mod.groupby('encoding')['psnr'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(balanced))
    bars = ax.bar(x, balanced.values,
                  color=[ENC_COLORS.get(e, '#888') for e in balanced.index],
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in balanced.index], rotation=30, ha='right')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Modality-Balanced Reconstruction Quality')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, balanced.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig1_psnr_balanced')

def _fig_mse_bars(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 2: MSE bar chart...")
    if 'mse' not in df.columns:
        return
    per_mod = df.groupby(['modality', 'encoding'])['mse'].mean().reset_index()
    balanced = per_mod.groupby('encoding')['mse'].mean().sort_values(ascending=True)  # Lower is better
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(balanced))
    bars = ax.bar(x, balanced.values,
                  color=[ENC_COLORS.get(e, '#888') for e in balanced.index],
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in balanced.index], rotation=30, ha='right')
    ax.set_ylabel('MSE (lower = better)')
    ax.set_title('Mean Squared Error Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, balanced.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + val*0.02,
                f'{val:.5f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig2_mse_balanced')


def _fig_edge_vs_flat(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 2: Edge vs Flat scatter...")
    if 'edge_psnr' not in df.columns or 'flat_psnr' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for enc in df['encoding'].unique():
        edf = df[df['encoding'] == enc]
        ax.scatter(edf['flat_psnr'], edf['edge_psnr'],
                   c=ENC_COLORS.get(enc, '#888'), marker=ENC_MARKERS.get(enc, 'o'),
                   label=enc.title(), alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5, zorder=0)
    ax.set_xlabel('Flat Region PSNR (dB)')
    ax.set_ylabel('Edge Region PSNR (dB)')
    ax.set_title('Edge vs Flat Region Quality')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig2_edge_vs_flat')


def _fig_modality_boxplots(df, fig_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("  Figure 3: Per-modality boxplots...")
    if 'modality' not in df.columns:
        return
    modalities = sorted(df['modality'].unique())
    if not modalities:
        return
    n_mods = len(modalities)
    fig, axes = plt.subplots(1, n_mods, figsize=(4.5 * n_mods, 5), squeeze=False)
    axes = axes[0]
    for ax, mod in zip(axes, modalities):
        mdf = df[df['modality'] == mod]
        order = mdf.groupby('encoding')['psnr'].mean().sort_values(ascending=False).index
        sns.boxplot(data=mdf, x='encoding', y='psnr', ax=ax, order=order,
                    palette=ENC_COLORS, showfliers=False)
        ax.set_title(f'{mod} (N={mdf["patient_id"].nunique()})')
        ax.set_xlabel('')
        ax.set_ylabel('PSNR (dB)' if mod == modalities[0] else '')
        ax.set_xticklabels([t.get_text().title() for t in ax.get_xticklabels()],
                           rotation=45, ha='right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle('Reconstruction Quality by Modality', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig3_modality_boxplots')


def _fig_reconstruction_grid(results_df, proc_dir, recon_dir, fig_dir, psnr_cap=None, nohash_index=None):
    import matplotlib.pyplot as plt
    print("  Figure 4: Reconstruction grids...")
    if not proc_dir.exists() or not recon_dir.exists():
        return
    show_encodings = ['fourier', 'tropical', 'hybrid', 'square', 'trapezoid', 'rampup', 'rampdown']
    available_enc = [e for e in show_encodings if e in results_df['encoding'].values]
    if not available_enc:
        available_enc = sorted(results_df['encoding'].unique())[:3]
    modalities = results_df['modality'].unique() if 'modality' in results_df.columns else ['Unknown']
    examples = []
    for mod in modalities:
        mdf = results_df[results_df['modality'] == mod] if 'modality' in results_df.columns else results_df
        if len(mdf) == 0:
            continue
        # If No_Hash index provided, find a row whose output_file matches the pinned slice
        if nohash_index:
            matched = mdf[mdf.apply(
                lambda r: nohash_index.get(r.get('patient_id', ''), '') == r.get('output_file', ''),
                axis=1
            )]
            row = matched.iloc[0] if len(matched) > 0 else mdf.iloc[0]
        else:
            row = mdf.iloc[0]
        sname = row['slice_name']
        gt_path = proc_dir / row.get('output_file', f"{sname}.npy")
        if not gt_path.exists():
            gt_path = proc_dir / f"{sname}.npy"
        if gt_path.exists():
            examples.append((mod, sname, gt_path))
    if not examples:
        return
    n_rows = len(examples)
    n_cols = len(available_enc) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for i, (mod, sname, gt_path) in enumerate(examples):
        gt = np.load(gt_path)
        axes[i, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            axes[i, 0].set_title('Ground Truth', fontsize=14, fontweight='bold', pad=6)
        axes[i, 0].set_ylabel(mod, fontsize=14, fontweight='bold', labelpad=8)
        axes[i, 0].axis('off')
        recons = {}
        for j, enc in enumerate(available_enc):
            rf = recon_dir / f"{sname}_{enc}.npy"
            ax = axes[i, j + 1]
            if rf.exists():
                r = np.load(rf).astype(np.float32)
                recons[enc] = r
                mse = np.mean((gt - r) ** 2)
                raw_psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
                psnr = min(raw_psnr, psnr_cap) if psnr_cap is not None else raw_psnr
                ax.imshow(r, cmap='gray', vmin=0, vmax=1)
                if i == 0:
                    ax.set_title(f'{enc.title()}\n{psnr:.1f} dB', fontsize=13, fontweight='bold', pad=6)
                else:
                    ax.set_title(f'{psnr:.1f} dB', fontsize=13, pad=6)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=13)
            ax.axis('off')
        ax = axes[i, -1]
        enc_a = 'fourier' if 'fourier' in recons else available_enc[0]
        enc_b = 'square' if 'square' in recons else (available_enc[-1] if len(available_enc) > 1 else available_enc[0])
        if enc_a in recons and enc_b in recons:
            err_a = np.abs(recons[enc_a] - gt)
            err_b = np.abs(recons[enc_b] - gt)
            improvement = err_a - err_b
            ax.imshow(improvement, cmap='RdBu', vmin=-0.05, vmax=0.05)
            if i == 0:
                ax.set_title(f'{enc_b.title()} advantage', fontsize=13, fontweight='bold', pad=6)
        ax.axis('off')
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig4_reconstruction_grid')


def _fig_ringing_bars(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 5: Ringing comparison...")
    if 'ringing_score' not in df.columns:
        return
    means = df.groupby('encoding')['ringing_score'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(means)), means.values,
                  color=[ENC_COLORS.get(e, '#888') for e in means.index],
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([e.title() for e in means.index], rotation=30, ha='right')
    ax.set_ylabel('Ringing Score (lower = better)')
    ax.set_title('Ringing Artifact Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig5_ringing')


def _fig_radar(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 6: Radar charts...")
    if 'modality' not in df.columns:
        return
    metrics = ['psnr', 'ssim', 'edge_psnr', 'flat_psnr']
    avail = [m for m in metrics if m in df.columns]
    if len(avail) < 3:
        return
    for mod in df['modality'].unique():
        mdf = df[df['modality'] == mod]
        means = mdf.groupby('encoding')[avail].mean()
        if len(means) < 2:
            continue
        norm = (means - means.min()) / (means.max() - means.min() + 1e-8)
        angles = np.linspace(0, 2 * np.pi, len(avail), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for enc in norm.index:
            vals = norm.loc[enc].tolist() + [norm.loc[enc].iloc[0]]
            ax.plot(angles, vals, 'o-', label=enc.title(),
                    color=ENC_COLORS.get(enc, '#888'), linewidth=2)
            ax.fill(angles, vals, alpha=0.08, color=ENC_COLORS.get(enc, '#888'))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in avail], fontsize=9)
        ax.set_title(f'{mod}', fontsize=13, pad=20)
        ax.legend(bbox_to_anchor=(1.3, 1), fontsize=8)
        plt.tight_layout()
        _save_fig(fig, fig_dir / f'fig6_radar_{mod.lower()}')


def _fig_convergence(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 7: Convergence curves...")
    cp_cols = sorted([c for c in df.columns if c.startswith('psnr_at_')])
    if not cp_cols:
        return
    iters = [int(c.split('_')[-1]) for c in cp_cols]
    fig, ax = plt.subplots(figsize=(10, 6))
    for enc in sorted(df['encoding'].unique()):
        edf = df[df['encoding'] == enc]
        means = [edf[c].dropna().mean() for c in cp_cols]
        stds = [edf[c].dropna().std() for c in cp_cols]
        valid = [(i, m, s) for i, m, s in zip(iters, means, stds) if not np.isnan(m)]
        if not valid:
            continue
        vi, vm, vs = zip(*valid)
        ax.plot(vi, vm, 'o-', label=enc.title(), color=ENC_COLORS.get(enc, '#888'),
                marker=ENC_MARKERS.get(enc, 'o'), linewidth=2, markersize=6)
        ax.fill_between(vi, [m - s for m, s in zip(vm, vs)],
                        [m + s for m, s in zip(vm, vs)],
                        alpha=0.1, color=ENC_COLORS.get(enc, '#888'))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Convergence Speed')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig7_convergence')


def _fig_heldout_comparison(train_df, test_df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 8: Held-out comparison...")
    encodings = sorted(set(train_df['encoding'].unique()) & set(test_df['encoding'].unique()))
    encodings = [e for e in encodings if e != 'none']
    if not encodings:
        return
    train_means = [train_df[train_df['encoding'] == e]['psnr'].mean() for e in encodings]
    test_means = [test_df[test_df['encoding'] == e]['psnr'].mean() for e in encodings]
    x = np.arange(len(encodings))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, train_means, width, label='Train (80% patients)',
           color=[ENC_COLORS.get(e, '#888') for e in encodings], alpha=0.9,
           edgecolor='white', linewidth=1.2)
    ax.bar(x + width / 2, test_means, width, label='Test (20% held-out patients)',
           color=[ENC_COLORS.get(e, '#888') for e in encodings], alpha=0.5,
           edgecolor='white', linewidth=1.2, hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in encodings], rotation=30, ha='right')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Held-Out Generalization: Train vs Test Split')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (tr, te) in enumerate(zip(train_means, test_means)):
        delta = te - tr
        color = 'green' if delta > -0.5 else 'red'
        ax.annotate(f'{delta:+.1f}', xy=(x[i] + width / 2, te),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7, color=color)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig8_heldout_comparison')


def _fig_efficiency_scatter(df, fig_dir):
    import matplotlib.pyplot as plt
    print("  Figure 9: Efficiency scatter...")
    if 'num_params' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for enc in df['encoding'].unique():
        edf = df[df['encoding'] == enc]
        ax.scatter(edf['num_params'].iloc[0], edf['psnr'].mean(),
                   c=ENC_COLORS.get(enc, '#888'), marker=ENC_MARKERS.get(enc, 'o'),
                   s=150, label=enc.title(), edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Mean PSNR (dB)')
    ax.set_title('Quality vs Model Size')
    ax.legend(loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig9_efficiency')


def _fig_multimetric_convergence(df, fig_dir):
    """Figure 10: Convergence of ALL metrics (PSNR, SSIM, edge_psnr, flat_psnr, ringing) over checkpoints."""
    import matplotlib.pyplot as plt
    print("  Figure 10: Multi-metric convergence curves...")

    metric_cols_map = {
        'psnr': ('psnr_at_{}', 'PSNR (dB)'),
        'ssim': ('ssim_at_{}', 'SSIM'),
        'edge_psnr': ('edge_psnr_at_{}', 'Edge PSNR (dB)'),
        'flat_psnr': ('flat_psnr_at_{}', 'Flat PSNR (dB)'),
        'ringing': ('ringing_at_{}', 'Ringing Score'),
    }

    available_cps = sorted([
        int(c.split('_')[-1]) for c in df.columns if c.startswith('psnr_at_')
    ])
    if not available_cps:
        return

    n_metrics = len(metric_cols_map)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_key, (col_tmpl, ylabel)) in zip(axes, metric_cols_map.items()):
        cp_cols = [col_tmpl.format(cp) for cp in available_cps]
        cp_cols = [c for c in cp_cols if c in df.columns]
        if not cp_cols:
            continue
        iters = [int(c.split('_')[-1]) for c in cp_cols]

        for enc in sorted(df['encoding'].unique()):
            edf = df[df['encoding'] == enc]
            means = [edf[c].dropna().mean() for c in cp_cols]
            stds = [edf[c].dropna().std() for c in cp_cols]
            valid = [(i, m, s) for i, m, s in zip(iters, means, stds) if not np.isnan(m)]
            if not valid:
                continue
            vi, vm, vs = zip(*valid)
            ax.plot(vi, vm, 'o-', label=enc.title(),
                    color=ENC_COLORS.get(enc, '#888'),
                    marker=ENC_MARKERS.get(enc, 'o'), linewidth=2, markersize=5)
            ax.fill_between(vi, [m - s for m, s in zip(vm, vs)],
                            [m + s for m, s in zip(vm, vs)],
                            alpha=0.08, color=ENC_COLORS.get(enc, '#888'))

        ax.set_xlabel('Iterations')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    plt.suptitle('All Metrics vs Training Iterations', fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig10_multimetric_convergence')


def _fig_checkpoint_heatmap(df, fig_dir):
    """Figure 11: Heatmap of PSNR at each checkpoint for all encodings."""
    import matplotlib.pyplot as plt
    print("  Figure 11: Checkpoint heatmap...")

    cp_cols = sorted([c for c in df.columns if c.startswith('psnr_at_')],
                     key=lambda c: int(c.split('_')[-1]))
    if not cp_cols:
        return

    encodings = [e for e in df['encoding'].unique() if e != 'none']
    matrix = []
    for enc in encodings:
        edf = df[df['encoding'] == enc]
        matrix.append([edf[c].dropna().mean() for c in cp_cols])

    matrix = np.array(matrix)
    col_labels = [f"{int(c.split('_')[-1])}" for c in cp_cols]

    fig, ax = plt.subplots(figsize=(len(cp_cols) * 1.5 + 2, len(encodings) * 0.6 + 1.5))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(encodings)))
    ax.set_yticklabels([e.title() for e in encodings])
    ax.set_xlabel('Iterations')
    ax.set_title('PSNR (dB) at Each Checkpoint')
    plt.colorbar(im, ax=ax, label='PSNR (dB)')

    for i in range(len(encodings)):
        for j in range(len(cp_cols)):
            ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                    fontsize=8, color='black')
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig11_checkpoint_heatmap')


def _fig_psnr_per_sec(df, fig_dir):
    """Figure 12: PSNR per second bar chart (efficiency)."""
    import matplotlib.pyplot as plt
    print("  Figure 12: PSNR/sec efficiency...")
    if 'train_time_sec' not in df.columns:
        return
    encodings = [e for e in df['encoding'].unique() if e != 'none']
    psnr_per_sec = []
    for enc in encodings:
        edf = df[df['encoding'] == enc]
        avg_psnr = edf['psnr'].dropna().mean()
        avg_time = edf['train_time_sec'].dropna().mean()
        psnr_per_sec.append(avg_psnr / avg_time if avg_time > 0 else 0)

    x = np.arange(len(encodings))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, psnr_per_sec,
           color=[ENC_COLORS.get(e, '#888') for e in encodings],
           edgecolor='white', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([e.title() for e in encodings], rotation=30, ha='right')
    ax.set_ylabel('PSNR (dB) / Training Second')
    ax.set_title('Encoding Efficiency: PSNR per Second')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig12_psnr_per_sec')


def run_visualization(train_df, test_df, output_dir, cap=None, data_dir=None, nohash_cohort=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': DPI,
    })

    train_df = apply_psnr_cap(train_df, cap)
    if test_df is not None and len(test_df) > 0:
        test_df = apply_psnr_cap(test_df, cap)

    if data_dir is None:
        data_dir = output_dir
    fig_dir = ensure_dir(output_dir / 'figures')
    proc_dir = data_dir / 'processed'
    recon_dir = data_dir / 'reconstructions_train'
    # Load No_Hash cohort index to pin reconstruction figures to the same slices.
    nohash_index = None
    if nohash_cohort:
        try:
            nh = pd.read_csv(nohash_cohort)
            # Build {patient_id -> output_file} for the train split
            nohash_index = (
                nh[nh['split'] == 'train']
                .drop_duplicates('patient_id')
                .set_index('patient_id')['output_file']
                .to_dict()
            )
            print(f"  Using No_Hash cohort index: {len(nohash_index)} patients pinned")
        except Exception as e:
            print(f"  Warning: could not load --nohash-cohort: {e}")

    plot_df = train_df[train_df['encoding'] != 'none'].copy()

    _fig_psnr_bars(plot_df, fig_dir)
    _fig_mse_bars(plot_df, fig_dir)
    _fig_edge_vs_flat(plot_df, fig_dir)
    _fig_modality_boxplots(plot_df, fig_dir)
    _fig_reconstruction_grid(train_df, proc_dir, recon_dir, fig_dir, psnr_cap=cap, nohash_index=nohash_index)
    _fig_ringing_bars(plot_df, fig_dir)
    _fig_radar(plot_df, fig_dir)
    _fig_convergence(plot_df, fig_dir)
    

    if test_df is not None and len(test_df) > 0:
        _fig_heldout_comparison(train_df, test_df, fig_dir)

    _fig_efficiency_scatter(plot_df, fig_dir)
    _fig_multimetric_convergence(plot_df, fig_dir)
    _fig_checkpoint_heatmap(plot_df, fig_dir)
    _fig_psnr_per_sec(plot_df, fig_dir)

    # ---- Shared config for new reconstruction figures ----
    enc_order = ['none', 'siren', 'wavelet', 'hash', 'fourier', 'square']
    ref_enc_name = 'square'   # reference for advantage maps

    def _psnr(gt, r):
        mse = np.mean((gt - r) ** 2)
        raw = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
        return min(raw, cap) if cap is not None else raw

    def _err(gt, r, scale=5.0):
        return np.clip(np.abs(gt - r) * scale, 0, 1)

    def _adv(gt, ref, cmp):
        return np.abs(gt - ref) - np.abs(gt - cmp)

    def _zoom(img, cx, cy, frac=0.25):
        H, W = img.shape
        h, w = max(int(H * frac), 16), max(int(W * frac), 16)
        r0 = int(np.clip(int(cy * H) - h // 2, 0, H - h))
        c0 = int(np.clip(int(cx * W) - w // 2, 0, W - w))
        return sk_resize(img[r0:r0+h, c0:c0+w], (H, W),
                         anti_aliasing=True, preserve_range=True).astype(np.float32)

    def _zoom_center(gt):
        dens = uniform_filter(np.hypot(sobel(gt, 0), sobel(gt, 1)),
                              size=max(8, gt.shape[0] // 8))
        idx = np.unravel_index(dens.argmax(), dens.shape)
        return float(np.clip(idx[1]/gt.shape[1], 0.2, 0.8)), float(np.clip(idx[0]/gt.shape[0], 0.2, 0.8))

    def _zbox(ax, cx, cy, frac, H, W):
        import matplotlib.patches as patches
        h, w = max(int(H*frac), 16), max(int(W*frac), 16)
        r0 = max(0, min(int(cy*H) - h//2, H-h))
        c0 = max(0, min(int(cx*W) - w//2, W-w))
        ax.add_patch(patches.Rectangle((c0, r0), w, h,
                                        linewidth=1.5, edgecolor='yellow', facecolor='none'))

    def _load_examples(df, n_per_mod=1, modality=None):
        """Load gt + recon arrays for the best n_per_mod slices per modality.

        If nohash_index is set (from --nohash-cohort), slices are chosen by
        matching the patient_ids in the No_Hash cohort so reconstruction figures
        show the exact same slice indexes for a fair comparison. All files are
        loaded from this run's own proc_dir / recon_dir.
        """
        enc_list = [e for e in enc_order if e in df['encoding'].values]
        if not enc_list:
            return [], []
        sub = df[df['modality'] == modality] if modality and 'modality' in df.columns else df
        groups = (
            {m: sub[sub['modality'] == m] for m in sorted(sub['modality'].unique())}
            if 'modality' in sub.columns and modality is None
            else {modality or '__all__': sub}
        )

        examples = []
        for grp, gdf in groups.items():
            candidates = []
            for sname in (gdf['slice_name'].unique() if 'slice_name' in gdf.columns else []):
                row = gdf[gdf['slice_name'] == sname].iloc[0]
                pid = row.get('patient_id', '')

                # If a No_Hash cohort index is provided, prefer the slice whose
                # output_file matches what No_Hash selected for this patient.
                if nohash_index and pid in nohash_index:
                    pinned_file = nohash_index[pid]
                    gt_path = proc_dir / pinned_file
                    # Only consider this sname if its output_file matches the pinned slice
                    if row.get('output_file', '') != pinned_file:
                        continue
                else:
                    gt_path = proc_dir / row.get('output_file', f'{sname}.npy')
                    if not gt_path.exists():
                        gt_path = proc_dir / f'{sname}.npy'

                if not gt_path.exists():
                    continue
                n_avail = sum(1 for e in enc_list if (recon_dir/f'{sname}_{e}.npy').exists())
                avg_psnr = gdf[gdf['slice_name']==sname]['psnr'].mean() if 'psnr' in gdf.columns else 0
                candidates.append((n_avail, avg_psnr, sname, gt_path, row))
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _, _, sname, gt_path, row in candidates[:n_per_mod]:
                try:
                    gt = np.load(str(gt_path)).astype(np.float32)
                    recons = {e: np.load(str(recon_dir/f'{sname}_{e}.npy')).astype(np.float32)
                              for e in enc_list if (recon_dir/f'{sname}_{e}.npy').exists()}
                    mod = row.get('modality', grp) if hasattr(row, 'get') else grp
                    examples.append((gt, recons, sname, mod))
                except Exception as e:
                    print(f'    skip {sname}: {e}')
        return examples, enc_list
        

    def _vstack(figs, title=''):
        rendered = []
        for f in figs:
            bio = BytesIO()
            f.savefig(bio, format='png', dpi=DPI, bbox_inches='tight')
            bio.seek(0)
            rendered.append(np.array(_PIL_Image.open(bio).convert('RGB'), dtype=np.uint8))
        if not rendered:
            return plt.figure()
        max_w = max(a.shape[1] for a in rendered)
        stacked = np.concatenate([
            np.concatenate([a, np.ones((a.shape[0], max_w-a.shape[1], 3), dtype=np.uint8)*255], axis=1)
            if a.shape[1] < max_w else a for a in rendered
        ], axis=0)
        fig, ax = plt.subplots(figsize=(stacked.shape[1]/DPI, stacked.shape[0]/DPI + 0.4))
        ax.imshow(stacked); ax.axis('off')
        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold', y=1.002)
        fig.tight_layout(pad=0)
        return fig

    def _error_adv_panel(gt, recons, enc_list, sname, modality):
        """3-row panel: recons | error maps | advantage vs ref."""
        import matplotlib.gridspec as gridspec
        avail = ['__gt__'] + [e for e in enc_list if e in recons]
        ref = ref_enc_name if ref_enc_name in recons else (avail[1] if len(avail) > 1 else None)
        fig = plt.figure(figsize=(max(3.2*len(avail), 14), 9))
        gs = gridspec.GridSpec(3, len(avail), figure=fig, hspace=0.05, wspace=0.03)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(len(avail))] for r in range(3)]
        for ci, enc in enumerate(avail):
            ax0, ax1, ax2 = axes[0][ci], axes[1][ci], axes[2][ci]
            if enc == '__gt__':
                ax0.imshow(gt, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
                ax0.set_title('Ground\nTruth', fontsize=9, fontweight='bold', pad=3)
                for ax in (ax1, ax2): ax.axis('off')
                ax1.text(0.5, 0.5, 'Error Map\n(×5)', ha='center', va='center',
                         fontsize=8, transform=ax1.transAxes, color='#555')
                ax2.text(0.5, 0.5, f'Advantage\nvs {ref.title() if ref else "—"}',
                         ha='center', va='center', fontsize=8, transform=ax2.transAxes, color='#555')
            else:
                r = recons[enc]; pv = _psnr(gt, r); col = ENC_COLORS.get(enc, '#888')
                ax0.imshow(r, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
                ax0.set_title(f'{enc.title()}\n{pv:.1f} dB', fontsize=9, color=col, fontweight='bold', pad=3)
                ax1.imshow(_err(gt, r), cmap='hot', vmin=0, vmax=1, interpolation='lanczos')
                if ref and enc != ref and ref in recons:
                    adv = _adv(gt, recons[ref], r)
                    ax2.imshow(adv, cmap='RdBu', vmin=-0.05, vmax=0.05, interpolation='lanczos')
                    m = adv.mean()
                    ax2.set_title(f'{"+" if m>=0 else ""}{m:.4f}', fontsize=7, pad=2,
                                   color='navy' if m>=0 else 'darkred')
                elif enc == ref:
                    ax2.imshow(np.zeros_like(gt), cmap='gray', vmin=0, vmax=1)
                    ax2.set_title('(reference)', fontsize=7, pad=2, color='gray')
                else:
                    ax2.axis('off')
            for ax in (ax0, ax1, ax2):
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
        fig.suptitle(f'{modality}  ·  {sname}', fontsize=11, y=0.998, fontweight='bold')
        return fig

    def _zoom_panel(gt, recons, enc_list, sname, modality, frac=0.25):
        """3-row panel: recons with zoom box | error maps (×5) | zoomed crops."""
        import matplotlib.gridspec as gridspec
        avail = ['__gt__'] + [e for e in enc_list if e in recons]
        cx, cy = _zoom_center(gt); H, W = gt.shape
        fig = plt.figure(figsize=(max(3.2*len(avail), 14), 9))
        gs = gridspec.GridSpec(3, len(avail), figure=fig, hspace=0.05, wspace=0.03)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(len(avail))] for r in range(3)]
        for ci, enc in enumerate(avail):
            ax0, ax1, ax2 = axes[0][ci], axes[1][ci], axes[2][ci]
            if enc == '__gt__':
                ax0.imshow(gt, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
                ax0.set_title('Ground\nTruth', fontsize=9, fontweight='bold', pad=3)
                _zbox(ax0, cx, cy, frac, H, W)
                ax1.axis('off')
                ax1.text(0.5, 0.5, 'Error Map\n(×5)', ha='center', va='center',
                         fontsize=8, transform=ax1.transAxes, color='#555')
                ax2.imshow(_zoom(gt, cx, cy, frac), cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
                ax2.set_title('GT Zoom', fontsize=8, pad=2)
            else:
                r = recons[enc]; pv = _psnr(gt, r); col = ENC_COLORS.get(enc, '#888')
                ax0.imshow(r, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
                _zbox(ax0, cx, cy, frac, H, W)
                ax0.set_title(f'{enc.title()}\n{pv:.1f} dB', fontsize=9, color=col, fontweight='bold', pad=3)
                ax1.imshow(_err(gt, r, scale=5.0), cmap='hot', vmin=0, vmax=1, interpolation='lanczos')
                ax2.imshow(_zoom(r, cx, cy, frac), cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
                ax2.set_title(f'Zoom  {pv:.1f} dB', fontsize=7, pad=2, color=col)
            for ax in (ax0, ax1, ax2):
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
        fig.suptitle(f'{modality}  ·  {sname}', fontsize=11, y=0.998, fontweight='bold')
        return fig

    # ---- figA1: error+advantage, all modalities combined ----
    print("\n  --- Reconstruction figures (A: error+advantage, B: zoom) ---")
    examples, enc_list = _load_examples(train_df, n_per_mod=1)
    if examples:
        figs = [_error_adv_panel(gt, recons, enc_list, sname, mod)
                for gt, recons, sname, mod in examples]
        combined = _vstack(figs, title='Reconstruction · Error Maps · Advantage (All Modalities)')
        for f in figs: plt.close(f)
        _save_fig(combined, fig_dir / 'figA1_recon_error_adv_ALL'); plt.close(combined)
        print("    -> figA1_recon_error_adv_ALL")

    # ---- figA2: error+advantage, per modality ----
    for mod in (sorted(train_df['modality'].unique()) if 'modality' in train_df.columns else []):
        examples, enc_list = _load_examples(train_df, n_per_mod=1, modality=mod)
        if not examples: continue
        figs = [_error_adv_panel(gt, recons, enc_list, sname, mod)
                for gt, recons, sname, _ in examples]
        combined = _vstack(figs, title=f'{mod} — Error Maps & Advantage')
        for f in figs: plt.close(f)
        _save_fig(combined, fig_dir / f'figA2_recon_error_adv_{mod}'); plt.close(combined)
        print(f"    -> figA2_recon_error_adv_{mod}")

    # ---- figB1: zoom, all modalities combined ----
    examples, enc_list = _load_examples(train_df, n_per_mod=1)
    if examples:
        figs = [_zoom_panel(gt, recons, enc_list, sname, mod)
                for gt, recons, sname, mod in examples]
        combined = _vstack(figs, title='Zoomed Reconstructions (All Modalities)')
        for f in figs: plt.close(f)
        _save_fig(combined, fig_dir / 'figB1_recon_zoom_ALL'); plt.close(combined)
        print("    -> figB1_recon_zoom_ALL")

    # ---- figB2: zoom, per modality ----
    for mod in (sorted(train_df['modality'].unique()) if 'modality' in train_df.columns else []):
        examples, enc_list = _load_examples(train_df, n_per_mod=1, modality=mod)
        if not examples: continue
        figs = [_zoom_panel(gt, recons, enc_list, sname, mod)
                for gt, recons, sname, _ in examples]
        combined = _vstack(figs, title=f'{mod} — Zoomed Reconstructions')
        for f in figs: plt.close(f)
        _save_fig(combined, fig_dir / f'figB2_recon_zoom_{mod}'); plt.close(combined)
        print(f"    -> figB2_recon_zoom_{mod}")

    print(f"\nAll figures saved to {fig_dir}/")

# ============================================================================
# PHASE 5: REQUIREMENTS
# ============================================================================

def generate_requirements(output_dir):
    reqs = [
        "torch>=2.0.0", "numpy>=1.24.0", "scipy>=1.10.0",
        "scikit-image>=0.20.0", "pandas>=2.0.0", "matplotlib>=3.7.0",
        "seaborn>=0.12.0", "nibabel>=5.0.0", "tqdm>=4.65.0",
        "Pillow>=9.5.0", "PyYAML>=6.0", "pydicom>=2.4.0",
        "openslide-python>=1.2.0",
    ]
    path = output_dir / 'requirements.txt'
    with open(path, 'w') as f:
        f.write("\n".join(reqs) + "\n")
    print(f"  requirements.txt saved to {path}")

    root_path = Path(__file__).parent / 'requirements.txt'
    with open(root_path, 'w') as f:
        f.write("\n".join(reqs) + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MICCAI Publication Pipeline (v2 — Self-Contained)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', default=None,
                        help='Path to optional YAML config file (uses embedded defaults if not provided)')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing (use existing cohort)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (use existing results)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Skip baseline encodings (siren, wavelet, hash)')
    parser.add_argument('--skip-heldout', action='store_true',
                        help='Skip held-out evaluation on test slices')
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Only run preprocessing, then exit (for SLURM parallelization)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (2000 iterations)')
    parser.add_argument('--encodings', nargs='+', default=None,
                        help='Override encoding list')
    parser.add_argument('--skip-downstream', action='store_true',
                        help='Skip downstream task evaluation')
    parser.add_argument('--downstream-only', action='store_true',
                        help='Only run downstream tasks (skip main training)')
    parser.add_argument('--nohash-cohort', default=None,
                        help='Path to No_Hash cohort.csv. When set, reconstruction '
                             'figures use the same patient/slice indexes as the No_Hash '
                             'run for a fair side-by-side comparison.')
    parser.add_argument('--skip-collect', action='store_true')
    args = parser.parse_args()

    start_time = time.time()

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

    output_dir = ensure_dir('results/miccai')

    if args.encodings:
        encodings = args.encodings
    else:
        encodings = MAIN_ENCODINGS.copy()
        if not args.skip_baselines:
            encodings.extend(BASELINE_ENCODINGS)

    print(f"\n{'='*70}")
    print(f"MICCAI PUBLICATION PIPELINE v2 (Self-Contained)")
    print(f"{'='*70}")
    print(f"Encodings: {encodings}")
    print(f"Image size: {MICCAI_IMAGE_SIZE}x{MICCAI_IMAGE_SIZE}")
    print(f"Iterations: {config['training']['num_iterations']}")
    print(f"PSNR analysis: cap@50, cap@100, uncapped (3 separate output dirs)")
    print(f"Held-out eval: {'skip' if args.skip_heldout else 'enabled'}")
    print(f"Convergence checkpoints: {CONVERGENCE_CHECKPOINTS}")
    print()

    set_seed(SEED)

    # Step 1: Preprocessing
    cohort_path = output_dir / 'cohort.csv'
    if not args.skip_preprocess:
        print(f"\n{'='*70}")
        print("STEP 1/5: PREPROCESSING")
        print(f"{'='*70}")
        cohort_df = preprocess_all(config, output_dir)
    else:
        if not cohort_path.exists():
            print("ERROR: No cohort.csv found. Run without --skip-preprocess first.")
            sys.exit(1)
        cohort_df = pd.read_csv(cohort_path)
        print(f"Loaded existing cohort: {len(cohort_df)} slices")

    if len(cohort_df) == 0:
        print("ERROR: No images were preprocessed. Check data/raw/ directory.")
        sys.exit(1)

    if 'modality' in cohort_df.columns:
        train_cohort = cohort_df[cohort_df['split'] == 'train'] if 'split' in cohort_df.columns else cohort_df
        for mod in train_cohort['modality'].unique():
            n = len(train_cohort[train_cohort['modality'] == mod])
            if n < MIN_SAMPLES_PER_MODALITY:
                print(f"\n  WARNING: {mod} has only {n} train images. "
                      f"Need >= {MIN_SAMPLES_PER_MODALITY} for statistical testing.")
            elif n < RECOMMENDED_PER_MODALITY:
                print(f"\n  NOTE: {mod} has {n} train images. "
                      f"MICCAI reviewers prefer {RECOMMENDED_PER_MODALITY}+.")

    if args.preprocess_only:
        print(f"\n{'='*70}")
        print("PREPROCESS COMPLETE (--preprocess-only)")
        print(f"{'='*70}")
        print(f"Cohort: {len(cohort_df)} slices -> {output_dir / 'cohort.csv'}")
        print("Run training with run_miccai_worker.py or continue without --preprocess-only")
        sys.exit(0)

    # Step 2: Training (train split)
    train_results_path = output_dir / 'metrics' / 'results_train.csv'
    if not args.skip_training:
        print(f"\n{'='*70}")
        print("STEP 2/5: TRAINING (Train Split - rank-1 slices)")
        print(f"{'='*70}")
        train_df = run_training(cohort_df, config, encodings, output_dir,
                                resume=args.resume, split='train')
    else:
        if not train_results_path.exists():
            old_path = output_dir / 'metrics' / 'all_results.csv'
            if old_path.exists():
                train_df = pd.read_csv(old_path)
            else:
                print("ERROR: No results found. Run without --skip-training first.")
                sys.exit(1)
        else:
            train_df = pd.read_csv(train_results_path)
        print(f"Loaded existing train results: {len(train_df)} experiments")

    # Step 3: Training (test split — held-out)
    test_df = pd.DataFrame()
    test_results_path = output_dir / 'metrics' / 'results_test.csv'
    if not args.skip_heldout and not args.skip_training:
        has_test = 'split' in cohort_df.columns and len(cohort_df[cohort_df['split'] == 'test']) > 0
        if has_test:
            print(f"\n{'='*70}")
            print("STEP 3/5: TRAINING (Test Split - held-out patients, 80/20 patient split)")
            print(f"{'='*70}")
            test_df = run_training(cohort_df, config, encodings, output_dir,
                                   resume=args.resume, split='test')
        else:
            print("\n  No test split slices found (2D images only?). Skipping held-out.")
    elif not args.skip_heldout and args.skip_training:
        if test_results_path.exists():
            test_df = pd.read_csv(test_results_path)
            print(f"Loaded existing test results: {len(test_df)} experiments")
    else:
        print("\n  Held-out evaluation skipped (--skip-heldout).")

   
    # Steps 4 & 5: Evaluation + Figures — run for each PSNR cap
    cap_configs = [('cap50', 50), ('cap100', 100), ('nocap', None)]
    if args.skip_collect:
        print("\n  Collect (evaluation + figures) skipped (--skip-collect).")
    for cap_label, cap_val in cap_configs:
        if args.skip_collect:
            break
        cap_dir = ensure_dir(output_dir / f'analysis_{cap_label}')
        cap_str = f"{cap_val} dB" if cap_val is not None else "uncapped"
        print(f"\n{'='*70}")
        print(f"STEP 4/5: STATISTICAL EVALUATION  [PSNR {cap_str}]")
        print(f"{'='*70}")
        _test = test_df if len(test_df) > 0 else None
        sig_df = run_evaluation(train_df, _test, cap_dir, cap=cap_val)

        print(f"\n{'='*70}")
        print(f"STEP 5/5: GENERATING FIGURES  [PSNR {cap_str}]")
        print(f"{'='*70}")
        run_visualization(train_df, _test, cap_dir, cap=cap_val, data_dir=output_dir,
                          nohash_cohort=args.nohash_cohort)

    # Step 6: Downstream tasks
    ds_df = pd.DataFrame()
    downstream_results_path = output_dir / 'downstream' / 'results_downstream.csv'
    if not args.skip_downstream:
        print(f"\n{'='*70}")
        print("STEP 6/6: DOWNSTREAM TASKS (super-resolution, segmentation, inpainting)")
        print(f"  {N_DOWNSTREAM_PER_MODALITY} images per modality")
        print(f"{'='*70}")
        if args.skip_training and downstream_results_path.exists():
            ds_df = pd.read_csv(downstream_results_path)
            print(f"Loaded existing downstream results: {len(ds_df)} rows")
        else:
            ds_df = run_downstream_tasks(cohort_df, config, encodings, output_dir,
                                         n_per_modality=N_DOWNSTREAM_PER_MODALITY)

        if len(ds_df) > 0:
            # generate downstream tables and figures for each cap
            for cap_label, cap_val in cap_configs:
                cap_dir = ensure_dir(output_dir / f'analysis_{cap_label}')
                tables_dir = ensure_dir(cap_dir / 'tables')
                fig_dir = ensure_dir(cap_dir / 'figures')
                _generate_downstream_table(ds_df, tables_dir)
                _fig_downstream(ds_df, fig_dir)
                _fig_downstream_modality(ds_df, fig_dir)
    else:
        print("\n  Downstream tasks skipped (--skip-downstream).")

    # Requirements
    generate_requirements(output_dir)

    # Summary
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {h}h {m}m {s}s")
    print(f"\nResults directory: {output_dir}/")
    print(f"  cohort.csv                  - {len(cohort_df)} preprocessed slices")
    print(f"  metrics/                    - Raw CSV results (uncapped)")
    print(f"  reconstructions_train/      - Train split reconstructions")
    if len(test_df) > 0:
        print(f"  reconstructions_test/       - Held-out test reconstructions")
    print(f"  convergence/                - Per-experiment convergence data")
    print(f"  analysis_cap50/             - Tables + figures with PSNR capped at 50 dB")
    print(f"  analysis_cap100/            - Tables + figures with PSNR capped at 100 dB")
    print(f"  analysis_nocap/             - Tables + figures with uncapped PSNR")
    print(f"  downstream/                 - Downstream task results CSV")
    print(f"  requirements.txt            - Python dependencies")

    n_train = len(train_df)
    n_test = len(test_df) if len(test_df) > 0 else 0
    n_patients = train_df['patient_id'].nunique()
    n_encodings = train_df['encoding'].nunique()
    print(f"\n  Train experiments: {n_train} ({n_patients} patients x {n_encodings} encodings)")
    if n_test > 0:
        print(f"  Test experiments:  {n_test} (held-out generalization)")

    print(f"\nLaTeX tables and figures (per PSNR cap):")
    for cap_label, _ in cap_configs:
        cap_dir = output_dir / f'analysis_{cap_label}'
        tables_dir = cap_dir / 'tables'
        fig_dir = cap_dir / 'figures'
        n_tex = len(list(tables_dir.glob('*.tex'))) if tables_dir.exists() else 0
        n_fig = len(list(fig_dir.glob('*.pdf'))) if fig_dir.exists() else 0
        print(f"  {cap_label}/  -> {n_tex} tables, {n_fig} figures")

    if n_patients < 20:
        print(f"\n  NOTE: {n_patients} patients is below typical MICCAI standards (50+).")

    print(f"\nBaseline citations for your paper:")
    print(f"  SIREN: Sitzmann et al., 'Implicit Neural Representations with "
          f"Periodic Activation Functions', NeurIPS 2020")
    print(f"  Instant-NGP (hash): Mueller et al., 'Instant Neural Graphics "
          f"Primitives with a Multiresolution Hash Encoding', SIGGRAPH 2022")
    print(f"  Wavelet: Fathony et al., 'Multiplicative Filter Networks', ICLR 2021")


# ============================================================================
# DOWNSTREAM TASKS
# ============================================================================
# Three tasks that test whether PPE representations generalise beyond fitting:
#   1. Super-Resolution: train on LR (4× down), query at HR coords
#   2. Segmentation:     train at full-res, Otsu-threshold recon, measure Dice
#   3. Inpainting:       mask centre 30%, train on visible pixels, eval on masked
#
# Each task runs on N_DOWNSTREAM_PER_MODALITY images per modality.
# ============================================================================

N_DOWNSTREAM_PER_MODALITY = 5
DOWNSTREAM_TASKS = ['image_fitting', 'super_resolution', 'segmentation', 'inpainting']


def _dice(mask_a, mask_b):
    """Dice coefficient between two binary masks."""
    intersection = (mask_a & mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return 2.0 * intersection / denom if denom > 0 else 1.0


def _otsu_threshold(arr):
    """Binary threshold via Otsu's method (numpy only, no skimage dependency)."""
    arr_u8 = (arr * 255).astype(np.uint8)
    counts, bins = np.histogram(arr_u8, bins=256, range=(0, 255))
    total = counts.sum()
    best_t, best_var = 0, 0.0
    sum_all = np.dot(np.arange(256), counts)
    w0, sum0 = 0, 0
    for t in range(256):
        w0 += counts[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        sum0 += t * counts[t]
        mu0 = sum0 / w0
        mu1 = (sum_all - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return arr > (best_t / 255.0)


def _run_downstream_superres(image, config, scale=4):
    """
    Super-resolution: train on a scale× downsampled image, reconstruct at
    original resolution. Returns PSNR and SSIM vs original.
    """
    from skimage.metrics import structural_similarity
    from skimage.transform import resize as sk_resize

    H, W = image.shape
    Hlr, Wlr = max(H // scale, 8), max(W // scale, 8)
    lr = sk_resize(image, (Hlr, Wlr), anti_aliasing=True,
                   preserve_range=True).astype(np.float32)
    lr = np.clip(lr, 0, 1)

    device = config['experiment']['device']
    enc_name = config['_downstream_enc']
    set_seed(SEED)
    model = _patched_create_model(config, enc_name)
    model = model.to(device)

    # train on LR coords
    y_lr = torch.linspace(0, 1, Hlr)
    x_lr = torch.linspace(0, 1, Wlr)
    yy, xx = torch.meshgrid(y_lr, x_lr, indexing='ij')
    coords_lr = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)
    target_lr = torch.from_numpy(lr).float().reshape(-1, 1).to(device)

    n_iter = config['training']['num_iterations']
    lr_rate = config['training']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['training'].get('decay_steps', []), gamma=0.1)

    batch_size = min(config['training']['batch_size'], Hlr * Wlr)
    n_pix = Hlr * Wlr
    for it in range(n_iter):
        model.train()
        idx = torch.randint(0, n_pix, (batch_size,))
        optimizer.zero_grad()
        pred = model(coords_lr[idx])
        loss = criterion(pred, target_lr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # query at HR coords
    model.eval()
    y_hr = torch.linspace(0, 1, H)
    x_hr = torch.linspace(0, 1, W)
    yy, xx = torch.meshgrid(y_hr, x_hr, indexing='ij')
    coords_hr = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)
    with torch.no_grad():
        hr_recon = model(coords_hr).cpu().numpy().reshape(H, W)
    hr_recon = np.clip(hr_recon, 0, 1)

    mse = np.mean((image - hr_recon) ** 2)
    psnr = 100.0 if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
    ssim = float(structural_similarity(image, hr_recon, data_range=1.0))
    return psnr, ssim, hr_recon


def _run_downstream_segmentation(image, recon):
    """
    Segmentation: Otsu-threshold both ground-truth image and reconstruction.
    Returns Dice coefficient between the two binary masks.
    """
    gt_mask = _otsu_threshold(image)
    recon_mask = _otsu_threshold(recon)
    return float(_dice(gt_mask, recon_mask))


def _run_downstream_inpainting(image, config, mask_frac=0.30):
    """
    Inpainting: mask the central (mask_frac × mask_frac) region.
    Train the INR on visible (unmasked) pixels only.
    Evaluate PSNR and SSIM on the masked region.
    """
    from skimage.metrics import structural_similarity

    H, W = image.shape
    device = config['experiment']['device']
    enc_name = config['_downstream_enc']

    # build mask (True = visible for training)
    mask = np.ones((H, W), dtype=bool)
    r0 = int(H * (0.5 - mask_frac / 2))
    r1 = int(H * (0.5 + mask_frac / 2))
    c0 = int(W * (0.5 - mask_frac / 2))
    c1 = int(W * (0.5 + mask_frac / 2))
    mask[r0:r1, c0:c1] = False
    hole_mask = ~mask

    y_all = torch.linspace(0, 1, H)
    x_all = torch.linspace(0, 1, W)
    yy, xx = torch.meshgrid(y_all, x_all, indexing='ij')
    coords_all = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    # only train on visible pixels
    visible_idx = torch.from_numpy(mask.reshape(-1)).nonzero(as_tuple=True)[0]
    coords_vis = coords_all[visible_idx].to(device)
    target_vis = torch.from_numpy(image.reshape(-1)[visible_idx.numpy()]).float().unsqueeze(1).to(device)

    set_seed(SEED)
    model = _patched_create_model(config, enc_name)
    model = model.to(device)

    n_iter = config['training']['num_iterations']
    lr_rate = config['training']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['training'].get('decay_steps', []), gamma=0.1)

    n_vis = len(coords_vis)
    batch_size = min(config['training']['batch_size'], n_vis)
    for it in range(n_iter):
        model.train()
        idx = torch.randint(0, n_vis, (batch_size,))
        optimizer.zero_grad()
        pred = model(coords_vis[idx])
        loss = criterion(pred, target_vis[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        recon = model(coords_all.to(device)).cpu().numpy().reshape(H, W)
    recon = np.clip(recon, 0, 1)

    # evaluate only on the masked (hole) region
    gt_hole = image[hole_mask]
    rc_hole = recon[hole_mask]
    mse = np.mean((gt_hole - rc_hole) ** 2)
    psnr = 100.0 if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))

    # SSIM on full image (with inpainted region)
    ssim = float(structural_similarity(image, recon, data_range=1.0))
    return psnr, ssim, recon


def run_downstream_tasks(cohort_df, config, encodings, output_dir, n_per_modality=N_DOWNSTREAM_PER_MODALITY):
    """
    Run downstream tasks for N_DOWNSTREAM_PER_MODALITY images per modality.

    Tasks: image_fitting, super_resolution, segmentation, inpainting
    Results saved to: results/miccai/downstream/results_downstream.csv
    """
    from tqdm import tqdm

    proc_dir = output_dir / 'processed'
    recon_dir = output_dir / 'reconstructions_train'
    downstream_dir = ensure_dir(output_dir / 'downstream')
    results_file = downstream_dir / 'results_downstream.csv'

    # Load existing results to allow resume
    completed = set()
    existing_rows = []
    if results_file.exists():
        _ex = pd.read_csv(results_file)
        existing_rows = _ex.to_dict('records')
        completed = set(zip(_ex['slice_name'], _ex['encoding'], _ex['task']))
        print(f"  Resuming downstream: {len(completed)} experiments already done")

    # Pick N images per modality from train split
    train_df = cohort_df[cohort_df['split'] == 'train'] if 'split' in cohort_df.columns else cohort_df
    selected_rows = []
    for mod, mdf in train_df.groupby('modality'):
        # prefer images that actually exist in processed dir
        avail = [r for _, r in mdf.iterrows()
                 if (proc_dir / r['output_file']).exists()]
        selected_rows.extend(avail[:n_per_modality])

    if not selected_rows:
        print("  No images found for downstream tasks.")
        return pd.DataFrame()

    print(f"  Downstream tasks: {len(selected_rows)} images × {len(encodings)} encodings × {len(DOWNSTREAM_TASKS)} tasks")

    all_results = list(existing_rows)
    total = len(selected_rows) * len(encodings) * len(DOWNSTREAM_TASKS)
    pbar = tqdm(total=total, desc="Downstream tasks")

    for row in selected_rows:
        image = np.load(proc_dir / row['output_file']).astype(np.float32)
        sname = Path(row['output_file']).stem

        for enc_name in encodings:
            # --- image_fitting: reuse existing reconstruction if available ---
            task = 'image_fitting'
            if (sname, enc_name, task) not in completed:
                recon_path = recon_dir / f"{sname}_{enc_name}.npy"
                if recon_path.exists():
                    recon = np.load(recon_path).astype(np.float32)
                    from skimage.metrics import structural_similarity
                    mse = np.mean((image - recon) ** 2)
                    psnr = 100.0 if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
                    ssim = float(structural_similarity(image, recon, data_range=1.0))
                    all_results.append({
                        'slice_name': sname, 'encoding': enc_name,
                        'patient_id': row['patient_id'],
                        'modality': row.get('modality', 'Unknown'),
                        'task': task,
                        'psnr': cap_psnr(psnr), 'psnr_raw': psnr,
                        'ssim': ssim, 'dice': np.nan,
                    })
                    completed.add((sname, enc_name, task))
            pbar.update(1)

            # --- super_resolution ---
            task = 'super_resolution'
            if (sname, enc_name, task) not in completed:
                try:
                    cfg = copy.deepcopy(config)
                    cfg['_downstream_enc'] = enc_name
                    psnr, ssim, _ = _run_downstream_superres(image, cfg, scale=4)
                    all_results.append({
                        'slice_name': sname, 'encoding': enc_name,
                        'patient_id': row['patient_id'],
                        'modality': row.get('modality', 'Unknown'),
                        'task': task,
                        'psnr': cap_psnr(psnr), 'psnr_raw': psnr,
                        'ssim': ssim, 'dice': np.nan,
                    })
                    completed.add((sname, enc_name, task))
                except Exception as e:
                    print(f"\n  Error {sname}/{enc_name}/superres: {e}")
            pbar.update(1)

            # --- segmentation: use existing recon (image_fitting output) ---
            task = 'segmentation'
            if (sname, enc_name, task) not in completed:
                recon_path = recon_dir / f"{sname}_{enc_name}.npy"
                if recon_path.exists():
                    try:
                        recon = np.load(recon_path).astype(np.float32)
                        dice = _run_downstream_segmentation(image, recon)
                        all_results.append({
                            'slice_name': sname, 'encoding': enc_name,
                            'patient_id': row['patient_id'],
                            'modality': row.get('modality', 'Unknown'),
                            'task': task,
                            'psnr': np.nan, 'psnr_raw': np.nan,
                            'ssim': np.nan, 'dice': dice,
                        })
                        completed.add((sname, enc_name, task))
                    except Exception as e:
                        print(f"\n  Error {sname}/{enc_name}/segmentation: {e}")
            pbar.update(1)

            # --- inpainting ---
            task = 'inpainting'
            if (sname, enc_name, task) not in completed:
                try:
                    cfg = copy.deepcopy(config)
                    cfg['_downstream_enc'] = enc_name
                    psnr, ssim, _ = _run_downstream_inpainting(image, cfg, mask_frac=0.30)
                    all_results.append({
                        'slice_name': sname, 'encoding': enc_name,
                        'patient_id': row['patient_id'],
                        'modality': row.get('modality', 'Unknown'),
                        'task': task,
                        'psnr': cap_psnr(psnr), 'psnr_raw': psnr,
                        'ssim': ssim, 'dice': np.nan,
                    })
                    completed.add((sname, enc_name, task))
                except Exception as e:
                    print(f"\n  Error {sname}/{enc_name}/inpainting: {e}")
            pbar.update(1)

            # checkpoint save every 20 results
            if len(all_results) % 20 == 0:
                pd.DataFrame(all_results).to_csv(results_file, index=False)

    pbar.close()
    df = pd.DataFrame(all_results)
    df.to_csv(results_file, index=False)
    print(f"\n  Downstream results: {len(df)} rows -> {results_file}")
    return df


def _generate_downstream_table(ds_df, tables_dir):
    """LaTeX table: one row per encoding, columns = task metrics."""
    if ds_df is None or len(ds_df) == 0:
        return

    task_metric = {
        'image_fitting': ('psnr', 'PSNR (dB)'),
        'super_resolution': ('psnr', 'SR PSNR (dB)'),
        'segmentation': ('dice', 'Dice'),
        'inpainting': ('psnr', 'Inpaint PSNR (dB)'),
    }

    encodings = [e for e in ds_df['encoding'].unique() if e != 'none']
    encodings_sorted = sorted(encodings)

    # compute means
    rows_data = {}
    for enc in encodings_sorted:
        edf = ds_df[ds_df['encoding'] == enc]
        rows_data[enc] = {}
        for task, (metric, _) in task_metric.items():
            tdf = edf[edf['task'] == task][metric].dropna()
            rows_data[enc][task] = (tdf.mean(), tdf.std()) if len(tdf) > 0 else (np.nan, np.nan)

    # find best per column
    best = {}
    for task, (metric, _) in task_metric.items():
        vals = {e: rows_data[e][task][0] for e in encodings_sorted if not np.isnan(rows_data[e][task][0])}
        best[task] = max(vals, key=vals.get) if vals else None

    col_header = ' & '.join([label for _, label in task_metric.values()])
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Downstream task evaluation (5 images per modality). "
        "Image fitting: PSNR of INR at 3500 iterations. "
        "Super-resolution: train on $4\\times$ downsampled, reconstruct at full resolution. "
        "Segmentation: Dice score between Otsu-thresholded reconstruction and ground truth. "
        "Inpainting: PSNR in masked centre region (30\\% of image). Best per column in \\textbf{bold}.}",
        "\\label{tab:downstream}",
        f"\\begin{{tabular}}{{l{'c' * len(task_metric)}}}",
        "\\toprule",
        f"Encoding & {col_header} \\\\",
        "\\midrule",
    ]
    for enc in encodings_sorted:
        cells = []
        for task, (metric, _) in task_metric.items():
            mu, sd = rows_data[enc][task]
            if np.isnan(mu):
                cells.append('--')
            else:
                val_str = f"{mu:.3f}$\\pm${sd:.3f}" if metric == 'dice' else f"{mu:.1f}$\\pm${sd:.1f}"
                if best.get(task) == enc:
                    val_str = f"\\textbf{{{val_str}}}"
                cells.append(val_str)
        lines.append(f"{enc.title()} & {' & '.join(cells)} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = tables_dir / 'table_downstream.tex'
    path.write_text('\n'.join(lines))
    print(f"    Saved {path}")

    # also save CSV
    rows_csv = []
    for enc in encodings_sorted:
        r = {'encoding': enc}
        for task, (metric, label) in task_metric.items():
            mu, sd = rows_data[enc][task]
            r[f'{task}_mean'] = round(mu, 4) if not np.isnan(mu) else np.nan
            r[f'{task}_std'] = round(sd, 4) if not np.isnan(sd) else np.nan
        rows_csv.append(r)
    pd.DataFrame(rows_csv).to_csv(tables_dir / 'table_downstream.csv', index=False)


def _fig_downstream(ds_df, fig_dir):
    """Figures for downstream tasks: grouped bar charts per task."""
    import matplotlib.pyplot as plt
    if ds_df is None or len(ds_df) == 0:
        return
    print("  Downstream figures...")

    task_cfg = {
        'image_fitting':    ('psnr',  'PSNR (dB)',        'Image Fitting'),
        'super_resolution': ('psnr',  'PSNR (dB)',        'Super-Resolution (4×)'),
        'segmentation':     ('dice',  'Dice Score',       'Segmentation (Otsu)'),
        'inpainting':       ('psnr',  'PSNR (dB)',        'Inpainting (30% masked)'),
    }

    encodings = [e for e in ds_df['encoding'].unique() if e != 'none']
    encodings_sorted = sorted(encodings)
    x = np.arange(len(encodings_sorted))

    fig, axes = plt.subplots(1, len(task_cfg), figsize=(5 * len(task_cfg), 5), sharey=False)
    if len(task_cfg) == 1:
        axes = [axes]

    for ax, (task, (metric, ylabel, title)) in zip(axes, task_cfg.items()):
        tdf = ds_df[ds_df['task'] == task]
        means, stds = [], []
        for enc in encodings_sorted:
            vals = tdf[tdf['encoding'] == enc][metric].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 0 else 0)

        colors = [ENC_COLORS.get(e, '#888') for e in encodings_sorted]
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color=colors, edgecolor='white', linewidth=1.2, error_kw={'linewidth': 1})
        ax.set_xticks(x)
        ax.set_xticklabels([e.title() for e in encodings_sorted], rotation=40, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

        # annotate best
        if means:
            best_i = int(np.argmax(means))
            ax.bar(x[best_i:best_i+1], means[best_i:best_i+1],
                   color=colors[best_i], edgecolor='gold', linewidth=2.5)

    plt.suptitle('Downstream Task Evaluation', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, fig_dir / 'fig_downstream_tasks')


def _fig_downstream_modality(ds_df, fig_dir):
    """Per-modality breakdown of downstream task performance."""
    import matplotlib.pyplot as plt
    if ds_df is None or len(ds_df) == 0 or 'modality' not in ds_df.columns:
        return
    print("  Downstream per-modality figure...")

    task_cfg = {
        'super_resolution': ('psnr', 'SR PSNR (dB)'),
        'segmentation':     ('dice', 'Dice'),
        'inpainting':       ('psnr', 'Inpaint PSNR (dB)'),
    }
    modalities = sorted(ds_df['modality'].dropna().unique())
    highlight_encs = ['square', 'fourier', 'hash', 'siren']

    for task, (metric, ylabel) in task_cfg.items():
        tdf = ds_df[ds_df['task'] == task]
        if len(tdf) == 0:
            continue
        encs = [e for e in highlight_encs if e in tdf['encoding'].unique()]
        if not encs:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(modalities) * 1.5), 5))
        width = 0.8 / len(encs)
        x = np.arange(len(modalities))
        for i, enc in enumerate(encs):
            edf = tdf[tdf['encoding'] == enc]
            means = [edf[edf['modality'] == m][metric].dropna().mean() for m in modalities]
            stds  = [edf[edf['modality'] == m][metric].dropna().std()  for m in modalities]
            offset = (i - len(encs) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=2,
                   label=enc.title(), color=ENC_COLORS.get(enc, '#888'),
                   edgecolor='white', linewidth=1, error_kw={'linewidth': 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(modalities, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{task.replace("_", " ").title()} — Per Modality')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, fig_dir / f'fig_downstream_{task}_modality')


if __name__ == '__main__':
    main()
