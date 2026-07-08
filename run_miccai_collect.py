#!/usr/bin/env python
"""
MICCAI Collect — Merge per-encoding results and generate tables/figures.

Called after all parallel worker jobs complete.
Merges results/miccai/metrics/results_{split}_{encoding}.csv files into
single results_{split}.csv, then runs evaluation and visualization.

Self-contained — imports from run_miccai.py in the SAME directory.

Usage:
    python run_miccai_collect.py
    python run_miccai_collect.py --output-dir results/miccai
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

# Import evaluation and visualization from the main script (same directory)
from run_miccai import (
    run_evaluation, run_visualization, generate_requirements,
    apply_psnr_cap, ensure_dir, _fig_reconstruction_grid, _save_fig,
    PSNR_CAP, MICCAI_IMAGE_SIZE, MIN_SAMPLES_PER_MODALITY,
    RECOMMENDED_PER_MODALITY, DPI,
)


def merge_results(metrics_dir, split):
    """Merge all per-encoding CSV files into one results_{split}.csv."""
    pattern = f"results_{split}_*.csv"
    files = sorted(metrics_dir.glob(pattern))

    if not files:
        print(f"  No files matching {pattern} in {metrics_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                enc = f.stem.replace(f'results_{split}_', '')
                print(f"  {f.name}: {len(df)} rows ({enc})")
                dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of restarts)
    if 'slice_name' in merged.columns and 'encoding' in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=['slice_name', 'encoding'], keep='last')
        if len(merged) < before:
            print(f"  Removed {before - len(merged)} duplicates")

    # Save merged file
    merged_path = metrics_dir / f'results_{split}.csv'
    merged.to_csv(merged_path, index=False)
    print(f"  -> Merged: {merged_path} ({len(merged)} total experiments)")

    return merged


def main():
    parser = argparse.ArgumentParser(description='MICCAI Collect — merge and evaluate')
    parser.add_argument('--output-dir', default='results/miccai')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip merging — load existing results_train/test.csv and re-run eval+figures')
    parser.add_argument('--fig4-only', action='store_true',
                        help='Regenerate only fig4_reconstruction_grid using existing results + processed/ recons')
    parser.add_argument('--nohash-cohort', default=None,
                        help='Path to No_Hash cohort.csv to pin reconstruction figures to the same slice indexes')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / 'metrics'

    start_time = time.time()

    print("=" * 70)
    print("MICCAI COLLECT — Merging parallel results")
    print("=" * 70)

    if args.eval_only:
        # Load existing merged CSVs directly
        train_path = metrics_dir / 'results_train.csv'
        test_path = metrics_dir / 'results_test.csv'
        if not train_path.exists():
            print(f"ERROR: {train_path} not found. Run without --eval-only first.")
            sys.exit(1)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
        print(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")
    else:
        # ---- Merge train results ----
        print("\nMerging TRAIN results:")
        train_df = merge_results(metrics_dir, 'train')

        # ---- Remove test slices from train (guard against split leakage) ----
        cohort_path = output_dir / 'cohort.csv'
        if cohort_path.exists():
            cohort = pd.read_csv(cohort_path)
            test_patients = set(cohort[cohort['split'] == 'test']['patient_id'])
            before = len(train_df)
            train_df = train_df[~train_df['patient_id'].isin(test_patients)].reset_index(drop=True)
            dropped = before - len(train_df)
            if dropped > 0:
                print(f"  Removed {dropped} rows from train that belong to test patients")
            train_df.to_csv(metrics_dir / 'results_train.csv', index=False)

        # ---- Merge test results ----
        print("\nMerging TEST results:")
        test_df = merge_results(metrics_dir, 'test')

    if len(train_df) == 0:
        print("\nERROR: No train results found. Check worker logs.")
        sys.exit(1)

    # ---- Summary ----
    n_patients = train_df['patient_id'].nunique()
    n_encodings = train_df['encoding'].nunique()
    print(f"\nTotal train experiments: {len(train_df)} "
          f"({n_patients} patients x {n_encodings} encodings)")
    if len(test_df) > 0:
        print(f"Total test experiments:  {len(test_df)}")

    print(f"\nEncodings found: {sorted(train_df['encoding'].unique())}")
    if 'modality' in train_df.columns:
        print(f"Modalities: {sorted(train_df['modality'].unique())}")
        for mod in sorted(train_df['modality'].unique()):
            n = train_df[train_df['modality'] == mod]['patient_id'].nunique()
            status = ""
            if n < MIN_SAMPLES_PER_MODALITY:
                status = " *** LOW"
            elif n < RECOMMENDED_PER_MODALITY:
                status = f" (recommend {RECOMMENDED_PER_MODALITY}+)"
            print(f"  {mod}: {n} patients{status}")

    # ---- Run evaluation + figures for each PSNR cap ----
    cap_configs = [('cap50', 50), ('cap100', 100), ('nocap', None)]
    _test = test_df if len(test_df) > 0 else None
    for cap_label, cap_val in cap_configs:
        cap_dir = ensure_dir(output_dir / f'analysis_{cap_label}')
        cap_str = f"{cap_val} dB" if cap_val is not None else "uncapped"
        print(f"\n{'='*70}")
        print(f"STATISTICAL EVALUATION  [PSNR {cap_str}]")
        print(f"{'='*70}")
        run_evaluation(train_df, _test, cap_dir, cap=cap_val)
        print(f"\n{'='*70}")
        print(f"GENERATING FIGURES  [PSNR {cap_str}]")
        print(f"{'='*70}")
        run_visualization(train_df, _test, cap_dir, cap=cap_val, data_dir=output_dir,
                          nohash_cohort=args.nohash_cohort)

    # ---- Requirements ----
    generate_requirements(output_dir)

    # ---- Final summary ----
    elapsed = time.time() - start_time
    m, s = int(elapsed // 60), int(elapsed % 60)

    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {m}m {s}s")
    print(f"\nOutputs:")
    print(f"  {output_dir}/metrics/results_train.csv  ({len(train_df)} experiments)")
    if len(test_df) > 0:
        print(f"  {output_dir}/metrics/results_test.csv   ({len(test_df)} experiments)")

    print(f"\nLaTeX tables and figures (per PSNR cap):")
    for cap_label, _ in cap_configs:
        cap_dir = output_dir / f'analysis_{cap_label}'
        n_tex = len(list((cap_dir / 'tables').glob('*.tex'))) if (cap_dir / 'tables').exists() else 0
        n_fig = len(list((cap_dir / 'figures').glob('*.pdf'))) if (cap_dir / 'figures').exists() else 0
        print(f"  {cap_label}/  -> {n_tex} tables, {n_fig} figures")


if __name__ == '__main__':
    main()
