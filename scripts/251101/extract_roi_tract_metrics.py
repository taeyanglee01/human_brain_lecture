#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract white matter metrics within each ROI for all subjects
각 ROI 내부의 백질 특성 측정 (streamline count, FA, volume)
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

def resample_atlas_to_native(atlas_data, atlas_affine, target_shape, target_affine):
    """Resample atlas to match native space using nearest neighbor"""
    scale = np.array(atlas_data.shape) / np.array(target_shape)
    resampled = ndimage.zoom(atlas_data, 1.0/scale, order=0, mode='constant', cval=0)

    # Ensure exact match
    if resampled.shape != target_shape:
        # Adjust if there's a small mismatch
        resampled_fixed = np.zeros(target_shape, dtype=atlas_data.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        resampled = resampled_fixed

    return resampled

def load_atlas_labels(atlas_dir):
    """Load FreeSurferDKT atlas labels"""

    # Cortical labels
    cortical_file = os.path.join(atlas_dir, 'atlas', 'human', 'FreeSurferDKT_Cortical.txt')
    cortical_labels = {}
    with open(cortical_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    name = '_'.join(parts[1:])
                    cortical_labels[idx] = name

    # Subcortical labels
    subcortical_file = os.path.join(atlas_dir, 'atlas', 'human', 'FreeSurferDKT_Subcortical.txt')
    subcortical_labels = {}
    with open(subcortical_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    name = '_'.join(parts[1:])
                    subcortical_labels[idx] = name

    # Combine
    all_labels = {**cortical_labels, **subcortical_labels}
    return all_labels

def count_streamlines_in_roi(tdi_data, roi_mask):
    """
    Count streamlines passing through ROI using TDI
    TDI values represent tract density (number of streamlines per voxel)
    """
    tdi_in_roi = tdi_data[roi_mask]
    total_streamlines = np.sum(tdi_in_roi)
    return int(total_streamlines)

def calculate_roi_volume(roi_mask, voxel_dims):
    """Calculate ROI volume in mm³"""
    n_voxels = np.sum(roi_mask)
    voxel_volume = np.prod(voxel_dims)  # mm³ per voxel
    total_volume = n_voxels * voxel_volume
    return total_volume

def extract_metrics_for_subject(subj_id, results_dir, atlas_dir):
    """Extract all metrics for one subject"""

    print(f"\nProcessing {subj_id}...")

    # Paths
    subj_dir = os.path.join(results_dir, subj_id, 'ses-1', 'dwi')
    fa_file = os.path.join(subj_dir, f'{subj_id}_dwi.fib.gz.fa.nii.gz')
    tdi_file = os.path.join(subj_dir, f'{subj_id}_whole_brain.tt.gz.tdi.nii.gz')

    # Load FA map
    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()
    voxel_dims = fa_img.header.get_zooms()[:3]  # mm per voxel

    print(f"  FA shape: {fa_data.shape}, voxel size: {voxel_dims} mm")

    # Load TDI map
    tdi_img = nib.load(tdi_file)
    tdi_data = tdi_img.get_fdata()

    print(f"  TDI shape: {tdi_data.shape}")

    # Load atlases
    cortical_atlas_file = os.path.join(atlas_dir, 'atlas', 'human', 'FreeSurferDKT_Cortical.nii.gz')
    subcortical_atlas_file = os.path.join(atlas_dir, 'atlas', 'human', 'FreeSurferDKT_Subcortical.nii.gz')

    cortical_img = nib.load(cortical_atlas_file)
    subcortical_img = nib.load(subcortical_atlas_file)

    # Resample atlases to native space
    print("  Resampling atlases to native space...")
    cortical_resampled = resample_atlas_to_native(
        cortical_img.get_fdata(),
        cortical_img.affine,
        fa_data.shape,
        fa_img.affine
    )

    subcortical_resampled = resample_atlas_to_native(
        subcortical_img.get_fdata(),
        subcortical_img.affine,
        fa_data.shape,
        fa_img.affine
    )

    # Load labels
    all_labels = load_atlas_labels(atlas_dir)

    # Extract metrics for each ROI
    metrics = []

    for atlas_data, atlas_name in [(cortical_resampled, 'cortical'),
                                    (subcortical_resampled, 'subcortical')]:

        unique_rois = np.unique(atlas_data)
        unique_rois = unique_rois[unique_rois > 0]  # Exclude background

        for roi_idx in unique_rois:
            roi_idx = int(roi_idx)
            roi_name = all_labels.get(roi_idx, f'unknown_roi_{roi_idx}')

            # Create ROI mask
            roi_mask = (atlas_data == roi_idx)

            # Skip if ROI is empty
            if not np.any(roi_mask):
                continue

            # 1. Streamline count (from TDI)
            streamline_count = count_streamlines_in_roi(tdi_data, roi_mask)

            # 2. Mean FA value
            fa_in_roi = fa_data[roi_mask]
            fa_in_roi = fa_in_roi[fa_in_roi > 0]  # Exclude zero values
            mean_fa = np.mean(fa_in_roi) if len(fa_in_roi) > 0 else 0.0

            # 3. ROI volume
            roi_volume = calculate_roi_volume(roi_mask, voxel_dims)

            metrics.append({
                'subject_id': subj_id,
                'roi_index': roi_idx,
                'roi_name': roi_name,
                'atlas': atlas_name,
                'streamline_count': streamline_count,
                'mean_fa': mean_fa,
                'volume_mm3': roi_volume,
                'n_voxels': int(np.sum(roi_mask))
            })

    return metrics

def main():
    """Main processing function"""

    results_dir = 'C:/Users/Public/human_brain_lecture/results/251101'
    atlas_dir = 'C:/Users/Public/dsi_studio_win_cpu/dsi_studio_win'

    # Subject IDs
    subjects = [
        'sub-24053', 'sub-24630', 'sub-24663', 'sub-24683', 'sub-24684',
        'sub-24696', 'sub-24697', 'sub-24699', 'sub-24702', 'sub-60295'
    ]

    # Process all subjects
    all_metrics = []

    for subj_id in subjects:
        try:
            metrics = extract_metrics_for_subject(subj_id, results_dir, atlas_dir)
            all_metrics.extend(metrics)
            print(f"  OK - Extracted {len(metrics)} ROI metrics")
        except Exception as e:
            print(f"  ERROR processing {subj_id}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Save complete data
    output_file = os.path.join(results_dir, 'roi_tract_metrics_all_subjects.csv')
    df.to_csv(output_file, index=False)
    print(f"\nOK - Saved complete data: {output_file}")
    print(f"  Total rows: {len(df)}")

    # Save per-subject files
    print("\nOK - Saving individual subject files...")
    for subj_id in subjects:
        subj_data = df[df['subject_id'] == subj_id]
        if len(subj_data) > 0:
            subj_file = os.path.join(results_dir, f'{subj_id}_roi_tract_metrics.csv')
            subj_data.to_csv(subj_file, index=False)
            print(f"  {subj_id}: {len(subj_data)} ROIs")

    # Create summary statistics
    print("\nOK - Creating summary statistics...")
    summary_stats = df.groupby('roi_name').agg({
        'streamline_count': ['mean', 'std', 'min', 'max'],
        'mean_fa': ['mean', 'std', 'min', 'max'],
        'volume_mm3': ['mean', 'std', 'min', 'max']
    }).round(2)

    summary_file = os.path.join(results_dir, 'roi_tract_metrics_summary.csv')
    summary_stats.to_csv(summary_file)
    print(f"  Saved summary: {summary_file}")

    # Display sample
    print("\n" + "="*80)
    print("SAMPLE DATA (first 10 rows):")
    print("="*80)
    print(df.head(10).to_string())

    print("\n" + "="*80)
    print("SUMMARY STATISTICS (top 10 ROIs by streamline count):")
    print("="*80)
    top_rois = df.groupby('roi_name')['streamline_count'].mean().sort_values(ascending=False).head(10)
    print(top_rois)

    print("\n" + "="*80)
    print("DATA SHAPE:")
    print("="*80)
    print(f"Total measurements: {len(df)}")
    print(f"Number of subjects: {df['subject_id'].nunique()}")
    print(f"Number of ROIs: {df['roi_name'].nunique()}")
    print(f"Columns: {list(df.columns)}")

    return df

if __name__ == "__main__":
    df = main()
    print("\nOK - ROI tract metrics extraction complete!")
