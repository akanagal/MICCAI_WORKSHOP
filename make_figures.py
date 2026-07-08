#!/usr/bin/env python3
"""
Generate Figures A, B, C individually at 600 DPI + combined ABC panel.
Reads from analysis_cap50/metrics/ CSVs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

DPI = 600
BASE = Path('analysis_cap50/metrics')
OUT  = Path('analysis_cap50/figures')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── colours matching the screenshots ──────────────────────────────────────────
C = {
    'square':    '#c0392b',   # red
    'trapezoid': '#16a085',   # teal-green
    'hash':      '#1a7a1a',   # dark green
    'wavelet':   '#888888',   # gray
    'rampup':    '#8e44ad',   # purple (Sawtooth)
    'fourier':   '#2980b9',   # blue
    'siren':     '#e67e22',   # orange
    'tropical':  '#2980b9',   # reused for triangle in fig C
}

LABELS = {
    'square':    'Square (PPE)',
    'trapezoid': 'Trapezoid (PPE)',
    'hash':      'Hash (Instant-NGP)',
    'wavelet':   'Wavelet',
    'rampup':    'Sawtooth (PPE)',
    'fourier':   'Fourier',
    'siren':     'SIREN',
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Convergence curves
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig_A(ax=None):
    conv = pd.read_csv(BASE / 'table3_convergence.csv').set_index('encoding')
    cp_cols = [c for c in conv.columns if c.startswith('psnr_at_') and c.endswith('_mean')]
    iters = [int(c.replace('psnr_at_', '').replace('_mean', '')) for c in cp_cols]

    show_encs = ['square', 'trapezoid', 'hash', 'wavelet', 'rampup', 'fourier', 'siren']

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    for enc in show_encs:
        if enc not in conv.index:
            continue
        vals = [conv.loc[enc, c] for c in cp_cols]
        ax.plot(iters, vals, 'o-', color=C[enc], label=LABELS[enc],
                linewidth=2, markersize=5, zorder=3)

    # annotation — placed in the gap between Fourier and SIREN lines
    sq_vals     = [conv.loc['square',  c] for c in cp_cols]
    siren_vals  = [conv.loc['siren',   c] for c in cp_cols]
    fourier_vals= [conv.loc['fourier', c] for c in cp_cols]
    # mid-point between fourier and siren at the last checkpoint
    text_y = (fourier_vals[-1] + siren_vals[-1]) / 2  # ~33.6 dB
    text_x = iters[-1]
    ax.annotate(
        'Square: best of the\nfixed encodings',
        xy=(iters[-1], sq_vals[-1]),
        xytext=(text_x, text_y),
        color='#c0392b', fontsize=9.5, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4,
                        connectionstyle='arc3,rad=-0.25'),
        zorder=10,
    )

    ax.set_xlabel('Training iterations')
    ax.set_ylabel('PSNR (dB, capped at 50)')
    ax.set_ylim(15, 53)
    ax.set_xticks(iters)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', ncol=2, framealpha=0.9,
              handlelength=1.8, columnspacing=1.0)

    if standalone:
        fig.tight_layout()
        fig.savefig(OUT / 'figA_convergence.png', dpi=DPI, bbox_inches='tight')
        fig.savefig(OUT / 'figA_convergence.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved figA_convergence  (.png + .pdf)')
        return None
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Per-modality bar chart (Fourier vs Square vs Hash ceiling)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig_B(ax=None):
    mod_df = pd.read_csv(BASE / 'per_modality_summary.csv')

    mod_order = ['CT', 'MRI', 'Pathology', 'Ultrasound', 'XRay']
    mod_labels = ['CT', 'MRI', 'Path', 'US', 'X-ray']

    fourier_means, fourier_stds = [], []
    square_means,  square_stds  = [], []
    hash_means,    hash_stds    = [], []

    for mod in mod_order:
        mdf = mod_df[mod_df['modality'] == mod]
        def _get(enc):
            row = mdf[mdf['encoding'] == enc]
            return (float(row['psnr_mean']), float(row['psnr_std'])) if len(row) else (0, 0)
        fm, fs = _get('fourier');   fourier_means.append(fm); fourier_stds.append(fs)
        sm, ss = _get('square');    square_means.append(sm);  square_stds.append(ss)
        hm, hs = _get('hash');      hash_means.append(hm);    hash_stds.append(hs)

    x = np.arange(len(mod_order))
    w = 0.25

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(x - w, fourier_means, w, yerr=fourier_stds, capsize=3,
           color='#2980b9', label='Fourier', zorder=3)
    ax.bar(x,     square_means,  w, yerr=square_stds,  capsize=3,
           color='#c0392b', label='Square (PPE)', zorder=3)
    ax.bar(x + w, hash_means,    w, yerr=hash_stds,    capsize=3,
           color='#16a085', alpha=0.55, hatch='//', label='Hash (ceiling, capped)', zorder=3)

    # gain label overrides (user-specified corrections)
    gain_overrides = {'Pathology': 8.3, 'XRay': -2.1}

    for i, (mod, fm, sm) in enumerate(zip(mod_order, fourier_means, square_means)):
        delta = gain_overrides.get(mod, round(sm - fm, 1))
        sign  = '+' if delta >= 0 else ''
        ax.text(x[i], max(sm, fm) + 0.8, f'{sign}{delta} dB',
                ha='center', va='bottom', fontsize=9, color='#c0392b', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(mod_labels)
    ax.set_ylabel('PSNR (dB)')
    ax.set_ylim(28, 57)
    ax.set_title('Square–Fourier gain (labelled) tracks boundary sharpness')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # legend between title and bars — upper right
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')

    if standalone:
        fig.tight_layout()
        fig.savefig(OUT / 'figB_modality_bars.png', dpi=DPI, bbox_inches='tight')
        fig.savefig(OUT / 'figB_modality_bars.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved figB_modality_bars  (.png + .pdf)')
        return None
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Harmonic density: PSNR + Ringing (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig_C(axes=None):
    t1 = pd.read_csv(BASE / 'table1_summary.csv').set_index('Encoding')

    # encoding → display label, x-label, color
    entries = [
        ('tropical',  'Triangle\n$1/n^2$',     '#7fb3d3'),
        ('trapezoid', 'Trapezoid\nmixed',        '#1a7a1a'),
        ('rampup',    'Sawtooth\n$1/n$',         '#8e44ad'),
        ('square',    'Square\n$1/n$ (odd)',      '#c0392b'),
    ]
    fourier_psnr    = float(t1.loc['fourier', 'psnr_mean'])
    fourier_ringing = float(t1.loc['fourier', 'ringing_score_mean'])

    psnr_vals    = [float(t1.loc[enc, 'psnr_mean'])          for enc, _, _ in entries]
    ringing_vals = [float(t1.loc[enc, 'ringing_score_mean']) for enc, _, _ in entries]
    colors       = [c for _, _, c in entries]
    xlabels      = [lbl for _, lbl, _ in entries]

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax_p, ax_r = axes

    # ── PSNR panel ──────────────────────────────────────────────
    x = np.arange(len(entries))
    bars = ax_p.bar(x, psnr_vals, color=colors, zorder=3, width=0.6)
    ax_p.axhline(fourier_psnr, color='#2980b9', linestyle='--', linewidth=1.5, zorder=2)
    ax_p.text(len(entries) - 0.5, fourier_psnr + 0.15,
              'Fourier baseline', color='#2980b9', fontsize=8.5, va='bottom')
    for bar, val, (enc, _, _) in zip(bars, psnr_vals, entries):
        weight = 'bold' if enc == 'square' else 'normal'
        label  = f'$\\bf{{{val:.1f}}}$' if enc == 'square' else f'{val:.1f}'
        ax_p.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                  label, ha='center', va='bottom', fontsize=9, fontweight=weight)
    ax_p.set_xticks(x); ax_p.set_xticklabels(xlabels, ha='center')
    ax_p.set_ylabel('PSNR (dB)')
    ax_p.set_ylim(38, 46.5)
    ax_p.set_title('Fidelity rises with harmonic density')
    ax_p.grid(axis='y', alpha=0.3, linestyle='--')

    # ── Ringing panel ────────────────────────────────────────────
    bars_r = ax_r.bar(x, ringing_vals, color=colors, zorder=3, width=0.6)
    ax_r.axhline(fourier_ringing, color='#2980b9', linestyle='--', linewidth=1.5, zorder=2)
    # Fourier label with white background so it's readable over the dashed line
    ax_r.text(len(entries) - 0.52, fourier_ringing + 0.00035,
              f'Fourier\n{fourier_ringing:.4f}', color='#2980b9', fontsize=8.5, va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))
    sq_idx = [enc for enc, _, _ in entries].index('square')
    for j, (bar, val) in enumerate(zip(bars_r, ringing_vals)):
        weight = 'bold' if j == sq_idx else 'normal'
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.0001,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight=weight)

    ax_r.set_xticks(x); ax_r.set_xticklabels(xlabels, ha='center')
    ax_r.set_ylabel('Ringing score (lower is better)')
    ax_r.set_title('Ringing falls with harmonic density')
    ax_r.grid(axis='y', alpha=0.3, linestyle='--')

    if standalone:
        fig.tight_layout()
        fig.savefig(OUT / 'figC_harmonic_density.png', dpi=DPI, bbox_inches='tight')
        fig.savefig(OUT / 'figC_harmonic_density.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved figC_harmonic_density  (.png + .pdf)')
        return None
    return axes


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED ABC PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def make_combined():
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                  height_ratios=[1, 1])

    ax_a  = fig.add_subplot(gs[0, :])       # A spans full top row
    ax_b  = fig.add_subplot(gs[1, 0])       # B bottom-left
    ax_c1 = fig.add_subplot(gs[1, 1])       # C-left  bottom-right (PSNR)
    # C needs two side-by-side axes inside gs[1,1] — use inset_axes trick
    ax_c1.remove()
    gs_inner = gs[1, 1].subgridspec(1, 2, wspace=0.45)
    ax_c1 = fig.add_subplot(gs_inner[0])
    ax_c2 = fig.add_subplot(gs_inner[1])

    make_fig_A(ax=ax_a)
    make_fig_B(ax=ax_b)
    make_fig_C(axes=(ax_c1, ax_c2))

    # Panel labels A, B, C
    for ax, label in [(ax_a, 'A'), (ax_b, 'B'), (ax_c1, 'C')]:
        ax.text(-0.07, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    fig.savefig(OUT / 'figABC_combined.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT / 'figABC_combined.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved figABC_combined  (.png + .pdf)')


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating figures...')
    make_fig_A()
    make_fig_B()
    make_fig_C()
    make_combined()
    print(f'\nDone. All figures saved to {OUT}/')
