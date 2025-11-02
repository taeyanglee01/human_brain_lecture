#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract ROI-based FA values using FreeSurfer DKT atlas
This script uses the FreeSurferDKT atlas from DSI Studio to extract
mean FA values for each ROI in each subject.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy import ndimage

def load_atlas_labels(txt_file):
    """Load atlas labels from txt file"""
    labels = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        labels[idx] = name
                    except ValueError:
                        continue
    return labels

def resample_atlas_to_fa(atlas_data, atlas_affine, fa_shape, fa_affine):
    """
    Resample atlas to match FA image dimensions using nearest neighbor interpolation
    """
    from scipy.ndimage import affine_transform

    # Calculate the transformation matrix
    atlas_to_fa = np.linalg.inv(fa_affine) @ atlas_affine

    # Extract scaling and translation
    scale = np.array(atlas_data.shape) / np.array(fa_shape)

    # Resample using zoom (simpler approach)
    resampled = ndimage.zoom(atlas_data,
                             1.0/scale,
                             order=0,  # nearest neighbor
                             mode='constant',
                             cval=0)

    # Crop or pad to match exact FA shape
    result = np.zeros(fa_shape, dtype=atlas_data.dtype)
    min_shape = np.minimum(resampled.shape, fa_shape)

    result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
        resampled[:min_shape[0], :min_shape[1], :min_shape[2]]

    return result

def extract_roi_fa_values(fa_file, atlas_file, atlas_labels):
    """
    Extract mean FA value for each ROI

    Parameters:
    -----------
    fa_file : str
        Path to FA NIfTI file
    atlas_file : str
        Path to atlas NIfTI file
    atlas_labels : dict
        Dictionary mapping ROI indices to names

    Returns:
    --------
    dict : ROI name -> mean FA value
    """
    # Load FA map
    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()

    # Load atlas
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Resample atlas to FA space if needed
    if atlas_data.shape != fa_data.shape:
        print(f"  Resampling atlas from {atlas_data.shape} to {fa_data.shape}")
        atlas_data = resample_atlas_to_fa(
            atlas_data,
            atlas_img.affine,
            fa_data.shape,
            fa_img.affine
        )

    # Extract mean FA for each ROI
    roi_values = {}
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois > 0]  # Exclude background

    for roi_idx in unique_rois:
        if roi_idx in atlas_labels:
            roi_mask = (atlas_data == roi_idx)
            fa_in_roi = fa_data[roi_mask]
            # Only include non-zero FA values
            fa_in_roi = fa_in_roi[fa_in_roi > 0]

            if len(fa_in_roi) > 0:
                mean_fa = np.mean(fa_in_roi)
                roi_name = atlas_labels[roi_idx]
                roi_values[roi_name] = mean_fa

    return roi_values

def main():
    # Define paths
    results_dir = Path(r"C:\Users\Public\human_brain_lecture\results\251101")
    atlas_dir = Path(r"C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\atlas\human")

    # Use FreeSurfer DKT Cortical atlas
    cortical_atlas = atlas_dir / "FreeSurferDKT_Cortical.nii.gz"
    cortical_labels_file = atlas_dir / "FreeSurferDKT_Cortical.txt"

    # Use FreeSurfer DKT Subcortical atlas
    subcortical_atlas = atlas_dir / "FreeSurferDKT_Subcortical.nii.gz"
    subcortical_labels_file = atlas_dir / "FreeSurferDKT_Subcortical.txt"

    # Load atlas labels
    print("Loading atlas labels...")
    cortical_labels = load_atlas_labels(str(cortical_labels_file))
    subcortical_labels = load_atlas_labels(str(subcortical_labels_file))

    print(f"Cortical ROIs: {len(cortical_labels)}")
    print(f"Subcortical ROIs: {len(subcortical_labels)}")

    # Find all FA map files
    fa_files = list(results_dir.glob("**/ses-1/dwi/*.fa.nii.gz"))
    fa_files.sort()

    print(f"\nFound {len(fa_files)} FA maps to process\n")

    # Process each subject
    all_results = []

    for fa_file in fa_files:
        subject_name = fa_file.parts[-4]
        print(f"Processing {subject_name}...")

        try:
            # Extract cortical ROI values
            print("  Extracting cortical ROI values...")
            cortical_values = extract_roi_fa_values(
                str(fa_file),
                str(cortical_atlas),
                cortical_labels
            )

            # Extract subcortical ROI values
            print("  Extracting subcortical ROI values...")
            subcortical_values = extract_roi_fa_values(
                str(fa_file),
                str(subcortical_atlas),
                subcortical_labels
            )

            # Combine values
            all_values = {**cortical_values, **subcortical_values}
            all_values['Subject'] = subject_name
            all_results.append(all_values)

            print(f"  Extracted {len(all_values)-1} ROI values")

        except Exception as e:
            print(f"  Error processing {subject_name}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorganize columns: Subject first, then all ROIs
    cols = ['Subject'] + [col for col in df.columns if col != 'Subject']
    df = df[cols]

    # Save to CSV
    output_file = results_dir / "roi_fa_values.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total subjects: {len(df)}")
    print(f"Total ROIs: {len(df.columns)-1}")

    # Also save summary statistics
    summary_file = results_dir / "roi_fa_summary.csv"
    summary = df.describe()
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")

if __name__ == "__main__":
    main()
