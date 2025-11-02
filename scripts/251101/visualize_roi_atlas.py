#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize FreeSurferDKT atlas ROIs with labels
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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

def create_color_map(n_regions):
    """Create distinct colors for each ROI"""
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', n_regions)
    colors = [cmap(i) for i in range(n_regions)]
    return colors

def visualize_atlas_with_labels(atlas_file, labels_file, output_file, atlas_name):
    """
    Visualize atlas with ROI labels
    """
    # Load atlas
    print(f"Loading {atlas_name}...")
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Load labels
    labels = load_atlas_labels(labels_file)

    # Get unique ROI indices
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois > 0]  # Exclude background

    print(f"Found {len(unique_rois)} ROIs")

    # Get middle slices
    x_mid = atlas_data.shape[0] // 2
    y_mid = atlas_data.shape[1] // 2
    z_mid = atlas_data.shape[2] // 2

    # Extract slices
    sagittal = atlas_data[x_mid, :, :]
    coronal = atlas_data[:, y_mid, :]
    axial = atlas_data[:, :, z_mid]

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # Create grid for layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.3)

    # Plot slices
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Sagittal
    im1 = ax1.imshow(np.rot90(sagittal), cmap='tab20', vmin=0, vmax=len(unique_rois))
    ax1.set_title('Sagittal View', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Coronal
    im2 = ax2.imshow(np.rot90(coronal), cmap='tab20', vmin=0, vmax=len(unique_rois))
    ax2.set_title('Coronal View', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Axial
    im3 = ax3.imshow(np.rot90(axial), cmap='tab20', vmin=0, vmax=len(unique_rois))
    ax3.set_title('Axial View', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Add title
    fig.suptitle(f'{atlas_name} Atlas - {len(unique_rois)} ROIs',
                 fontsize=18, fontweight='bold', y=0.98)

    # Create legend with ROI labels in the bottom half
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # Sort labels by index
    sorted_labels = sorted([(idx, labels.get(idx, f'ROI {idx}'))
                           for idx in unique_rois])

    # Create legend entries - split into columns
    n_cols = 4
    n_per_col = (len(sorted_labels) + n_cols - 1) // n_cols

    legend_text = ""
    for col in range(n_cols):
        start_idx = col * n_per_col
        end_idx = min(start_idx + n_per_col, len(sorted_labels))

        for idx, name in sorted_labels[start_idx:end_idx]:
            legend_text += f"{idx:3d}. {name}\n"

        if col < n_cols - 1:
            legend_text += "\n"

    # Display legend text
    ax_legend.text(0.05, 0.95, legend_text,
                  transform=ax_legend.transAxes,
                  fontsize=8,
                  verticalalignment='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")

def create_combined_visualization():
    """Create visualization for both cortical and subcortical atlases"""
    atlas_dir = Path(r"C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\atlas\human")
    output_dir = Path(r"C:\Users\Public\human_brain_lecture\results\251101")

    # Visualize cortical atlas
    cortical_atlas = atlas_dir / "FreeSurferDKT_Cortical.nii.gz"
    cortical_labels = atlas_dir / "FreeSurferDKT_Cortical.txt"
    cortical_output = output_dir / "roi_atlas_cortical.png"

    visualize_atlas_with_labels(
        str(cortical_atlas),
        str(cortical_labels),
        str(cortical_output),
        "FreeSurferDKT Cortical"
    )

    # Visualize subcortical atlas
    subcortical_atlas = atlas_dir / "FreeSurferDKT_Subcortical.nii.gz"
    subcortical_labels = atlas_dir / "FreeSurferDKT_Subcortical.txt"
    subcortical_output = output_dir / "roi_atlas_subcortical.png"

    visualize_atlas_with_labels(
        str(subcortical_atlas),
        str(subcortical_labels),
        str(subcortical_output),
        "FreeSurferDKT Subcortical"
    )

    # Create combined summary
    create_combined_summary(atlas_dir, output_dir)

def create_combined_summary(atlas_dir, output_dir):
    """Create a combined summary figure showing both atlases"""
    # Load both atlases
    cortical_atlas = nib.load(str(atlas_dir / "FreeSurferDKT_Cortical.nii.gz"))
    subcortical_atlas = nib.load(str(atlas_dir / "FreeSurferDKT_Subcortical.nii.gz"))

    cortical_data = cortical_atlas.get_fdata().astype(int)
    subcortical_data = subcortical_atlas.get_fdata().astype(int)

    # Load labels
    cortical_labels = load_atlas_labels(str(atlas_dir / "FreeSurferDKT_Cortical.txt"))
    subcortical_labels = load_atlas_labels(str(atlas_dir / "FreeSurferDKT_Subcortical.txt"))

    # Combine atlases (subcortical ROIs will overlay on cortical)
    combined = cortical_data.copy()
    # Shift subcortical indices to avoid overlap
    subcortical_shifted = subcortical_data.copy()
    subcortical_shifted[subcortical_shifted > 0] += 100
    combined[subcortical_data > 0] = subcortical_shifted[subcortical_data > 0]

    # Get middle slices
    x_mid = combined.shape[0] // 2
    y_mid = combined.shape[1] // 2
    z_mid = combined.shape[2] // 2

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FreeSurferDKT Atlas - All ROIs Overview',
                 fontsize=16, fontweight='bold')

    # Plot cortical
    axes[0, 0].imshow(np.rot90(cortical_data[x_mid, :, :]), cmap='tab20')
    axes[0, 0].set_title('Cortical - Sagittal', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.rot90(cortical_data[:, y_mid, :]), cmap='tab20')
    axes[0, 1].set_title('Cortical - Coronal', fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.rot90(cortical_data[:, :, z_mid]), cmap='tab20')
    axes[0, 2].set_title('Cortical - Axial', fontweight='bold')
    axes[0, 2].axis('off')

    # Plot subcortical
    axes[1, 0].imshow(np.rot90(subcortical_data[x_mid, :, :]), cmap='tab20')
    axes[1, 0].set_title('Subcortical - Sagittal', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.rot90(subcortical_data[:, y_mid, :]), cmap='tab20')
    axes[1, 1].set_title('Subcortical - Coronal', fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.rot90(subcortical_data[:, :, z_mid]), cmap='tab20')
    axes[1, 2].set_title('Subcortical - Axial', fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()

    output_file = output_dir / "roi_atlas_overview.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved overview: {output_file}")

    # Create ROI list document
    create_roi_list_document(cortical_labels, subcortical_labels, output_dir)

def create_roi_list_document(cortical_labels, subcortical_labels, output_dir):
    """Create a text document with all ROI names"""
    output_file = output_dir / "roi_list.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FreeSurferDKT Atlas - Complete ROI List\n")
        f.write("="*80 + "\n\n")

        f.write(f"CORTICAL REGIONS ({len(cortical_labels)} ROIs)\n")
        f.write("-"*80 + "\n")
        for idx, name in sorted(cortical_labels.items()):
            f.write(f"{idx:3d}. {name}\n")

        f.write("\n" + "="*80 + "\n\n")

        f.write(f"SUBCORTICAL REGIONS ({len(subcortical_labels)} ROIs)\n")
        f.write("-"*80 + "\n")
        for idx, name in sorted(subcortical_labels.items()):
            f.write(f"{idx:3d}. {name}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"TOTAL: {len(cortical_labels) + len(subcortical_labels)} ROIs\n")
        f.write("="*80 + "\n")

    print(f"Saved ROI list: {output_file}")

if __name__ == "__main__":
    create_combined_visualization()
    print("\nAll visualizations complete!")
