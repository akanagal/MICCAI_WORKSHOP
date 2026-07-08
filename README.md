# Piecewise Positional Encodings

---

## Overview

This pipeline trains and evaluates Implicit Neural Representations (INRs) with multiple positional encodings across five medical imaging modalities (CT, MRI, Pathology, Ultrasound, XRay). It produces LaTeX tables, convergence figures, per-modality bar charts, harmonic density plots, and reconstruction grids at three PSNR cap levels (50 dB, 100 dB, uncapped).

---

## Encodings

**Main (PPE variants):** `none`, `fourier`, `tropical`, `hybrid`, `square`, `trapezoid`, `rampup`, `rampdown`

**Baselines:** `siren`, `wavelet`, `hash` (Instant-NGP)

---

## Data

Place raw data in `data/raw/`. Accepted formats: NIfTI (`.nii`, `.nii.gz`), DICOM, SVS (pathology), PNG, JPEG.

- Images are resized to **256×256**
- 3D volumes: best slice selected by Sobel edge score (rank-1 only)
- Split: **80% train / 20% test** by patient ID (deterministic hash)
- Minimum 10 samples per modality recommended (30+ for statistical power)

---

## Pipeline

### Option 1 — Single machine (sequential)

```bash
# Full pipeline
python run_miccai.py

# Re-generate tables/figures only (skip training)
python run_miccai.py --skip-training

# Fast test run (2000 iterations)
python run_miccai.py --quick

# Preprocess only
python run_miccai.py --preprocess-only

# Specific encodings
python run_miccai.py --encodings fourier tropical hybrid
```

### Option 2 — SLURM (parallelized, one job per encoding)

```bash
# Full pipeline
sbatch run_miccai_slurm.sbatch

# Fast test
sbatch run_miccai_slurm.sbatch --quick
```

**SLURM job structure:**
1. Job 1: Preprocess (serial, ~5 min)
2. Job 2: Train array — one task per encoding, trains all images
3. Job 3: Collect + evaluate + visualize (serial, ~5 min)

### Option 3 — Worker (single encoding, manual / GPU)

```bash
# Train one encoding on train split
python run_miccai_worker.py --encoding fourier --split train

# With explicit GPU
python run_miccai_worker.py --encoding hash --split train --device cuda

# Multi-GPU targeting
python run_miccai_worker.py --encoding hash --split train --device cuda:1
```

Worker results are saved to:
- `results/miccai/metrics/results_{split}_{encoding}.csv`
- `results/miccai/reconstructions_{split}/{slice}_{encoding}.npy`
- `results/miccai/convergence/{slice}_{encoding}.json`

### Option 4 — Collect (merge + evaluate after workers finish)

```bash
# Merge all per-encoding CSVs, run evaluation and figures
python run_miccai_collect.py

# Re-run evaluation/figures from existing merged CSVs (no re-merge)
python run_miccai_collect.py --eval-only
```

---

## Outputs

All outputs go to `results/miccai/` (or `--output-dir`).

```
results/miccai/
  cohort.csv                        # preprocessed slice index
  metrics/
    results_train.csv               # merged training results
    results_test.csv                # merged test results
  analysis_cap50/                   # PSNR capped at 50 dB
    tables/                         # LaTeX .tex files
    figures/                        # PDF + PNG figures
    metrics/                        # per-modality / convergence CSVs
  analysis_cap100/                  # PSNR capped at 100 dB
  analysis_nocap/                   # PSNR uncapped
  reconstructions_train/            # .npy reconstructions (train split)
  reconstructions_test/             # .npy reconstructions (test split)
```

---

## Standalone Figures

### Figures A, B, C (convergence / modality bars / harmonic density)

Reads from `analysis_cap50/metrics/`. Saves to `analysis_cap50/figures/`.

```bash
python make_figures.py
```

Outputs: `figA_convergence`, `figB_modality_bars`, `figC_harmonic_density`, `figABC_combined` — each as `.png` (600 DPI) and `.pdf`.

### Figure 4 (reconstruction grid)

Reads from `processed/` and `reconstructions_train/`.

```bash
python make_fig4.py
```

Output: `fig4_reconstruction_grid.png` / `.pdf` in the working directory.

---

## Requirements

Core dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-image`, `Pillow`

A `requirements.txt` is auto-generated in the output directory after the pipeline runs.
