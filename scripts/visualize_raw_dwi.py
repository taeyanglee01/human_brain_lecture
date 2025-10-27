"""
Visualize raw DWI data and save results as PNG
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_dwi(subject_id, data_dir, output_dir):
    """
    Visualize DWI data for a given subject

    Parameters:
    - subject_id: subject identifier (e.g., 'sub-010002')
    - data_dir: path to data directory
    - output_dir: path to output directory
    """

    # Load DWI data
    dwi_path = os.path.join(data_dir, subject_id, 'ses-01', 'dwi', f'{subject_id}_ses-01_dwi.nii.gz')
    bval_path = os.path.join(data_dir, subject_id, 'ses-01', 'dwi', f'{subject_id}_ses-01_dwi.bval')
    bvec_path = os.path.join(data_dir, subject_id, 'ses-01', 'dwi', f'{subject_id}_ses-01_dwi.bvec')

    print(f"Loading DWI data for {subject_id}...")
    print(f"  DWI file: {dwi_path}")

    # Load NIfTI file
    img = nib.load(dwi_path)
    data = img.get_fdata()

    # Load b-values
    with open(bval_path, 'r') as f:
        bvals = np.array([float(x) for x in f.read().split()])

    print(f"  Data shape: {data.shape}")
    print(f"  Number of volumes: {data.shape[-1]}")
    print(f"  B-values range: {bvals.min():.0f} - {bvals.max():.0f}")
    print(f"  Unique b-values: {np.unique(bvals)}")

    # Create output directory
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Figure 1: Show different b-value volumes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{subject_id} - Raw DWI Data (Different B-values)', fontsize=16, fontweight='bold')

    # Find indices for different b-values
    b0_idx = np.where(bvals == 0)[0]
    b_low_idx = np.where((bvals > 0) & (bvals < 500))[0]
    b_mid_idx = np.where((bvals >= 500) & (bvals < 1500))[0]
    b_high_idx = np.where(bvals >= 1500)[0]

    middle_slice = data.shape[2] // 2

    # Show b=0 image
    if len(b0_idx) > 0:
        axes[0, 0].imshow(data[:, :, middle_slice, b0_idx[0]].T, cmap='gray', origin='lower')
        axes[0, 0].set_title(f'B=0 (volume {b0_idx[0]})')
        axes[0, 0].axis('off')

    # Show low b-value
    if len(b_low_idx) > 0:
        axes[0, 1].imshow(data[:, :, middle_slice, b_low_idx[0]].T, cmap='gray', origin='lower')
        axes[0, 1].set_title(f'B={bvals[b_low_idx[0]]:.0f} (volume {b_low_idx[0]})')
        axes[0, 1].axis('off')

    # Show mid b-value
    if len(b_mid_idx) > 0:
        axes[0, 2].imshow(data[:, :, middle_slice, b_mid_idx[0]].T, cmap='gray', origin='lower')
        axes[0, 2].set_title(f'B={bvals[b_mid_idx[0]]:.0f} (volume {b_mid_idx[0]})')
        axes[0, 2].axis('off')

    # Show high b-value
    if len(b_high_idx) > 0:
        axes[1, 0].imshow(data[:, :, middle_slice, b_high_idx[0]].T, cmap='gray', origin='lower')
        axes[1, 0].set_title(f'B={bvals[b_high_idx[0]]:.0f} (volume {b_high_idx[0]})')
        axes[1, 0].axis('off')

    # Show mean b=0
    if len(b0_idx) > 0:
        mean_b0 = np.mean(data[:, :, :, b0_idx], axis=-1)
        axes[1, 1].imshow(mean_b0[:, :, middle_slice].T, cmap='gray', origin='lower')
        axes[1, 1].set_title(f'Mean B=0 (n={len(b0_idx)})')
        axes[1, 1].axis('off')

    # Show mean DWI
    mean_dwi = np.mean(data, axis=-1)
    axes[1, 2].imshow(mean_dwi[:, :, middle_slice].T, cmap='gray', origin='lower')
    axes[1, 2].set_title(f'Mean DWI (all volumes)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(subject_output_dir, f'{subject_id}_raw_dwi_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # Figure 2: Show different slices of b=0 image
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'{subject_id} - B=0 Image (Axial Slices)', fontsize=16, fontweight='bold')

    if len(b0_idx) > 0:
        b0_data = data[:, :, :, b0_idx[0]]
        slice_indices = np.linspace(10, data.shape[2]-10, 15, dtype=int)

        for idx, slice_num in enumerate(slice_indices):
            row = idx // 5
            col = idx % 5
            axes[row, col].imshow(b0_data[:, :, slice_num].T, cmap='gray', origin='lower')
            axes[row, col].set_title(f'Slice {slice_num}')
            axes[row, col].axis('off')

    plt.tight_layout()
    output_path = os.path.join(subject_output_dir, f'{subject_id}_raw_b0_slices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # Figure 3: Signal intensity profile across volumes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{subject_id} - DWI Signal Analysis', fontsize=16, fontweight='bold')

    # Plot 1: B-values
    axes[0, 0].plot(bvals, 'o-', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Volume Index')
    axes[0, 0].set_ylabel('B-value (s/mm²)')
    axes[0, 0].set_title('B-values across volumes')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean signal intensity per volume
    mean_intensity = np.mean(data, axis=(0, 1, 2))
    axes[0, 1].plot(mean_intensity, 'o-', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Volume Index')
    axes[0, 1].set_ylabel('Mean Signal Intensity')
    axes[0, 1].set_title('Mean signal intensity per volume')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Signal vs b-value
    axes[1, 0].scatter(bvals, mean_intensity, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('B-value (s/mm²)')
    axes[1, 0].set_ylabel('Mean Signal Intensity')
    axes[1, 0].set_title('Signal intensity vs B-value')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Histogram of b=0 intensities
    if len(b0_idx) > 0:
        b0_data = data[:, :, :, b0_idx[0]]
        axes[1, 1].hist(b0_data.flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Signal Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('B=0 intensity histogram')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(subject_output_dir, f'{subject_id}_signal_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    print(f"\nVisualization complete for {subject_id}!")
    print(f"Output directory: {subject_output_dir}")
    print("="*60)

if __name__ == "__main__":
    # Configuration
    data_dir = r"C:\Users\Public\human_brain_lecture\data"
    output_dir = r"C:\Users\Public\human_brain_lecture\results"

    subjects = ['sub-010002', 'sub-010015']

    print("="*60)
    print("DWI Raw Data Visualization")
    print("="*60)

    for subject_id in subjects:
        visualize_dwi(subject_id, data_dir, output_dir)

    print("\nAll visualizations complete!")
