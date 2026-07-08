#!/usr/bin/env python3
"""Regenerate fig4_reconstruction_grid with larger fonts."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DPI = 600
BASE      = Path('.')
PROC_DIR  = BASE / 'processed'
RECON_DIR = BASE / 'reconstructions_train'
OUT_DIR   = BASE
PSNR_CAP  = 50.0

cohort = pd.read_csv(BASE / 'cohort.csv')

# Discover available encodings from files in recon dir
all_encs = ['fourier', 'tropical', 'hybrid', 'square', 'trapezoid', 'rampup', 'rampdown',
            'hash', 'siren', 'wavelet', 'none']
show_encs = ['fourier', 'tropical', 'hybrid', 'square', 'trapezoid', 'rampup', 'rampdown']

# Build index: stem -> set of encodings present
stem_to_encs = {}
for f in RECON_DIR.iterdir():
    for enc in all_encs:
        if f.name.endswith(f'_{enc}.npy'):
            stem = f.name[:-len(f'_{enc}.npy')]
            stem_to_encs.setdefault(stem, set()).add(enc)
            break

available_enc = [e for e in show_encs if any(e in encs for encs in stem_to_encs.values())]
if not available_enc:
    available_enc = show_encs

# Match cohort rows to available stems
cohort['stem'] = cohort['output_file'].apply(lambda x: Path(x).stem)
cohort_indexed = cohort[cohort['stem'].isin(stem_to_encs)].copy()

mod_order = ['CT', 'MRI', 'Pathology', 'Ultrasound', 'XRay']
examples = []
for mod in mod_order:
    mdf = cohort_indexed[cohort_indexed['modality'] == mod]
    if len(mdf) == 0:
        continue
    # pick the slice with the most show_encs available and highest edge score
    best_row = None
    best_score = (-1, -1)
    for _, row in mdf.iterrows():
        stem = row['stem']
        gt_path = PROC_DIR / row['output_file']
        if not gt_path.exists():
            continue
        count = sum(1 for e in available_enc if e in stem_to_encs.get(stem, set()))
        score = (count, row['edge_score'])
        if score > best_score:
            best_score = score
            best_row = (mod, stem, gt_path)
    if best_row:
        examples.append(best_row)

if not examples:
    print("No examples found — check processed/ and reconstructions_train/ paths.")
    exit(1)

print(f"Plotting {len(examples)} modalities x {len(available_enc)} encodings")

n_rows = len(examples)
n_cols = len(available_enc) + 2   # GT + encodings + advantage map

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.2 * n_cols, 4.2 * n_rows),
                         squeeze=False)

for i, (mod, sname, gt_path) in enumerate(examples):
    gt = np.load(gt_path).astype(np.float32)

    # Ground truth
    axes[i, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    if i == 0:
        axes[i, 0].set_title('Ground Truth', fontsize=14, fontweight='bold', pad=6)
    axes[i, 0].set_ylabel(mod, fontsize=14, fontweight='bold', labelpad=8)
    axes[i, 0].axis('off')

    recons = {}
    for j, enc in enumerate(available_enc):
        rf = RECON_DIR / f'{sname}_{enc}.npy'
        ax = axes[i, j + 1]
        if rf.exists():
            r = np.load(rf).astype(np.float32)
            recons[enc] = r
            mse = float(np.mean((gt - r) ** 2))
            raw_psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
            psnr = min(raw_psnr, PSNR_CAP)
            ax.imshow(r, cmap='gray', vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f'{enc.title()}\n{psnr:.1f} dB',
                             fontsize=13, fontweight='bold', pad=6)
            else:
                ax.set_title(f'{psnr:.1f} dB', fontsize=13, pad=6)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    fontsize=13, transform=ax.transAxes)
        ax.axis('off')

    # Advantage map (Fourier vs Square)
    ax = axes[i, -1]
    enc_a = 'fourier' if 'fourier' in recons else (available_enc[0] if available_enc else None)
    enc_b = 'square'  if 'square'  in recons else (available_enc[-1] if len(available_enc) > 1 else enc_a)
    if enc_a and enc_b and enc_a in recons and enc_b in recons:
        improvement = np.abs(recons[enc_a] - gt) - np.abs(recons[enc_b] - gt)
        ax.imshow(improvement, cmap='RdBu', vmin=-0.05, vmax=0.05)
        if i == 0:
            ax.set_title(f'{enc_b.title()} advantage',
                         fontsize=13, fontweight='bold', pad=6)
    ax.axis('off')

plt.tight_layout()
out_path = OUT_DIR / 'fig4_reconstruction_grid'
fig.savefig(str(out_path) + '.png', dpi=DPI, bbox_inches='tight')
fig.savefig(str(out_path) + '.pdf', bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}.png / .pdf')
