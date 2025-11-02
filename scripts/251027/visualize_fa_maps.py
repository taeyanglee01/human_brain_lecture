"""
Visualize FA maps in three orthogonal orientations
Shows axial, coronal, and sagittal slices with corpus callosum clearly visible
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_fa_map(subject_id, results_dir):
    """
    Visualize FA map for a given subject in three orientations

    Parameters:
    - subject_id: subject identifier (e.g., 'sub-010002')
    - results_dir: path to results directory
    """

    # Load FA map
    fa_path = os.path.join(results_dir, subject_id, f'{subject_id}.fib.gz.fa.nii.gz')

    print(f"Loading FA map for {subject_id}...")
    print(f"  FA file: {fa_path}")

    # Load NIfTI file
    img = nib.load(fa_path)
    fa_data = img.get_fdata()

    print(f"  FA data shape: {fa_data.shape}")
    print(f"  FA range: {fa_data.min():.3f} - {fa_data.max():.3f}")
    print(f"  FA mean: {fa_data[fa_data > 0].mean():.3f}")

    # Select slices to show corpus callosum
    # Axial: middle-upper slice (corpus callosum is at the top-middle of brain)
    axial_slice = int(fa_data.shape[2] * 0.55)  # ~55% height

    # Coronal: middle slice (corpus callosum runs anterior-posterior in midline)
    coronal_slice = fa_data.shape[1] // 2

    # Sagittal: middle slice (midline sagittal shows corpus callosum best)
    sagittal_slice = fa_data.shape[0] // 2

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{subject_id} - FA Map (Fractional Anisotropy)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Row 1: Single slices showing corpus callosum
    # Axial view
    ax_axial = axes[0, 0]
    im1 = ax_axial.imshow(fa_data[:, :, axial_slice].T, cmap='hot', origin='lower',
                          vmin=0, vmax=1, aspect='auto')
    ax_axial.set_title(f'Axial (Z={axial_slice})', fontsize=14, fontweight='bold')
    ax_axial.set_xlabel('Left ← → Right', fontsize=11)
    ax_axial.set_ylabel('Posterior ← → Anterior', fontsize=11)
    ax_axial.axis('on')
    plt.colorbar(im1, ax=ax_axial, fraction=0.046, pad=0.04, label='FA')

    # Coronal view
    ax_coronal = axes[0, 1]
    im2 = ax_coronal.imshow(fa_data[:, coronal_slice, :].T, cmap='hot', origin='lower',
                            vmin=0, vmax=1, aspect='auto')
    ax_coronal.set_title(f'Coronal (Y={coronal_slice})', fontsize=14, fontweight='bold')
    ax_coronal.set_xlabel('Left ← → Right', fontsize=11)
    ax_coronal.set_ylabel('Inferior ← → Superior', fontsize=11)
    ax_coronal.axis('on')
    plt.colorbar(im2, ax=ax_coronal, fraction=0.046, pad=0.04, label='FA')

    # Sagittal view
    ax_sagittal = axes[0, 2]
    im3 = ax_sagittal.imshow(fa_data[sagittal_slice, :, :].T, cmap='hot', origin='lower',
                             vmin=0, vmax=1, aspect='auto')
    ax_sagittal.set_title(f'Sagittal (X={sagittal_slice}, Midline)', fontsize=14, fontweight='bold')
    ax_sagittal.set_xlabel('Posterior ← → Anterior', fontsize=11)
    ax_sagittal.set_ylabel('Inferior ← → Superior', fontsize=11)
    ax_sagittal.axis('on')
    plt.colorbar(im3, ax=ax_sagittal, fraction=0.046, pad=0.04, label='FA')

    # Row 2: Multiple slices
    # Axial slices
    ax_axial_multi = axes[1, 0]
    axial_slices = [int(fa_data.shape[2] * p) for p in [0.45, 0.50, 0.55, 0.60]]
    axial_montage = np.hstack([fa_data[:, :, s].T for s in axial_slices])
    im4 = ax_axial_multi.imshow(axial_montage, cmap='hot', origin='lower',
                                 vmin=0, vmax=1, aspect='auto')
    ax_axial_multi.set_title('Axial (Multiple Slices)', fontsize=14, fontweight='bold')
    ax_axial_multi.set_xlabel('← Inferior    |    Superior →', fontsize=11)
    ax_axial_multi.set_yticks([])

    # Coronal slices
    ax_coronal_multi = axes[1, 1]
    coronal_slices = [int(fa_data.shape[1] * p) for p in [0.40, 0.45, 0.50, 0.55]]
    coronal_montage = np.hstack([fa_data[:, s, :].T for s in coronal_slices])
    im5 = ax_coronal_multi.imshow(coronal_montage, cmap='hot', origin='lower',
                                   vmin=0, vmax=1, aspect='auto')
    ax_coronal_multi.set_title('Coronal (Multiple Slices)', fontsize=14, fontweight='bold')
    ax_coronal_multi.set_xlabel('← Posterior    |    Anterior →', fontsize=11)
    ax_coronal_multi.set_yticks([])

    # Sagittal slices
    ax_sagittal_multi = axes[1, 2]
    sagittal_slices = [int(fa_data.shape[0] * p) for p in [0.45, 0.48, 0.50, 0.52]]
    sagittal_montage = np.hstack([fa_data[s, :, :].T for s in sagittal_slices])
    im6 = ax_sagittal_multi.imshow(sagittal_montage, cmap='hot', origin='lower',
                                    vmin=0, vmax=1, aspect='auto')
    ax_sagittal_multi.set_title('Sagittal (Multiple Slices)', fontsize=14, fontweight='bold')
    ax_sagittal_multi.set_xlabel('← Left    |    Midline    |    Right →', fontsize=11)
    ax_sagittal_multi.set_yticks([])

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(results_dir, subject_id, f'{subject_id}_FA_map_3views.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()

    # Create a second figure focused on corpus callosum
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'{subject_id} - FA Map: Corpus Callosum Views',
                  fontsize=18, fontweight='bold')

    # Axial - showing corpus callosum from top
    ax1 = axes2[0]
    im1 = ax1.imshow(fa_data[:, :, axial_slice].T, cmap='hot', origin='lower',
                     vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Axial: Corpus Callosum (Top View)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Left ← → Right', fontsize=12)
    ax1.set_ylabel('Posterior ← → Anterior', fontsize=12)

    # Add annotation
    height, width = fa_data[:, :, axial_slice].T.shape
    ax1.text(width//2, height*0.95, 'Corpus Callosum',
             ha='center', va='top', color='cyan', fontsize=12,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax1.arrow(width//2, height*0.9, 0, -height*0.15,
              head_width=10, head_length=5, fc='cyan', ec='cyan', linewidth=2)

    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='FA')

    # Coronal - showing corpus callosum from front
    ax2 = axes2[1]
    im2 = ax2.imshow(fa_data[:, coronal_slice, :].T, cmap='hot', origin='lower',
                     vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Coronal: Corpus Callosum (Front View)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Left ← → Right', fontsize=12)
    ax2.set_ylabel('Inferior ← → Superior', fontsize=12)

    # Add annotation
    height, width = fa_data[:, coronal_slice, :].T.shape
    ax2.text(width//2, height*0.75, 'Corpus Callosum',
             ha='center', va='center', color='cyan', fontsize=12,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax2.arrow(width//2, height*0.7, 0, -height*0.1,
              head_width=10, head_length=5, fc='cyan', ec='cyan', linewidth=2)

    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='FA')

    # Sagittal - showing corpus callosum from side (midline)
    ax3 = axes2[2]
    im3 = ax3.imshow(fa_data[sagittal_slice, :, :].T, cmap='hot', origin='lower',
                     vmin=0, vmax=1, aspect='auto')
    ax3.set_title('Sagittal: Corpus Callosum (Midline)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Posterior ← → Anterior', fontsize=12)
    ax3.set_ylabel('Inferior ← → Superior', fontsize=12)

    # Add annotation for corpus callosum regions
    height, width = fa_data[sagittal_slice, :, :].T.shape
    ax3.text(width*0.5, height*0.75, 'Corpus Callosum',
             ha='center', va='center', color='cyan', fontsize=12,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='FA')

    plt.tight_layout()

    # Save second figure
    output_path2 = os.path.join(results_dir, subject_id, f'{subject_id}_FA_corpus_callosum.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path2}")
    plt.close()

    print(f"\nVisualization complete for {subject_id}!")
    print("="*60)

if __name__ == "__main__":
    # Configuration
    results_dir = r"C:\Users\Public\human_brain_lecture\results"
    subjects = ['sub-010002', 'sub-010015']

    print("="*60)
    print("FA Map Visualization")
    print("="*60)

    for subject_id in subjects:
        visualize_fa_map(subject_id, results_dir)

    print("\nAll FA visualizations complete!")
