#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize FA maps with axial, coronal, and sagittal slices
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_fa_map(fa_file, output_file):
    """
    Visualize FA map with three orthogonal slices

    Parameters:
    -----------
    fa_file : str
        Path to FA NIfTI file
    output_file : str
        Path to save the visualization
    """
    # Load FA map
    print(f"Loading: {fa_file}")
    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()

    # Get middle slices
    x_mid = fa_data.shape[0] // 2
    y_mid = fa_data.shape[1] // 2
    z_mid = fa_data.shape[2] // 2

    # Extract slices
    sagittal = fa_data[x_mid, :, :]  # Sagittal (YZ plane)
    coronal = fa_data[:, y_mid, :]   # Coronal (XZ plane)
    axial = fa_data[:, :, z_mid]     # Axial (XY plane)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot sagittal
    im1 = axes[0].imshow(np.rot90(sagittal), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Sagittal', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot coronal
    im2 = axes[1].imshow(np.rot90(coronal), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Coronal', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Plot axial
    im3 = axes[2].imshow(np.rot90(axial), cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Axial', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add colorbar
    cbar = fig.colorbar(im3, ax=axes, orientation='horizontal',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Fractional Anisotropy (FA)', fontsize=12)

    # Add subject name as main title
    subject_name = os.path.basename(fa_file).split('_dwi')[0]
    fig.suptitle(f'FA Map - {subject_name}', fontsize=16, fontweight='bold')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def main():
    # Define paths
    results_dir = Path(r"C:\Users\Public\human_brain_lecture\results\251101")
    output_dir = results_dir / "fa_visualizations"
    output_dir.mkdir(exist_ok=True)

    # Find all FA map files
    fa_files = list(results_dir.glob("**/ses-1/dwi/*.fa.nii.gz"))
    fa_files.sort()

    print(f"Found {len(fa_files)} FA maps to visualize\n")

    # Visualize each FA map
    for fa_file in fa_files:
        subject_name = fa_file.parts[-4]  # Extract subject name
        output_file = output_dir / f"{subject_name}_fa_map.png"

        try:
            visualize_fa_map(str(fa_file), str(output_file))
        except Exception as e:
            print(f"Error processing {subject_name}: {e}")

    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
