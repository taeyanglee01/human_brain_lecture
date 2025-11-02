"""
Generate Tract Density Image (TDI) from DSI Studio tractography results
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import gzip
import struct
import os

def load_dsi_studio_tracts(tt_gz_path):
    """
    Load tract data from DSI Studio .tt.gz file
    """
    print(f"Loading tracts from {tt_gz_path}...")

    with gzip.open(tt_gz_path, 'rb') as f:
        # Read header
        header = f.read(8)
        id_string = header[:4].decode('ascii')

        if id_string != 'TRCT':
            print(f"Warning: File ID is '{id_string}', expected 'TRCT'")

        # Read dimension
        dim_data = f.read(12)
        dim_x, dim_y, dim_z = struct.unpack('III', dim_data)
        print(f"  Dimension: {dim_x} x {dim_y} x {dim_z}")

        # Read voxel size
        vs_data = f.read(12)
        voxel_size = struct.unpack('fff', vs_data)
        print(f"  Voxel size: {voxel_size}")

        # Read transformation matrix (4x4 = 16 floats)
        trans_data = f.read(64)
        trans_matrix = np.array(struct.unpack('f'*16, trans_data)).reshape(4, 4)

        # Read number of tracts
        n_tract_data = f.read(4)
        n_tracts = struct.unpack('I', n_tract_data)[0]
        print(f"  Number of tracts: {n_tracts}")

        # Read tract data
        tracts = []
        for i in range(n_tracts):
            # Read number of points in this tract
            n_points_data = f.read(4)
            if len(n_points_data) < 4:
                break

            n_points = struct.unpack('I', n_points_data)[0]

            # Read tract points
            points_data = f.read(n_points * 3 * 4)  # 3 floats per point
            if len(points_data) < n_points * 3 * 4:
                break

            points = np.array(struct.unpack('f'*(n_points*3), points_data))
            points = points.reshape(-1, 3)

            tracts.append(points)

            if (i + 1) % 50000 == 0:
                print(f"    Loaded {i+1}/{n_tracts} tracts...")

    print(f"  Successfully loaded {len(tracts)} tracts")

    return tracts, (dim_x, dim_y, dim_z), voxel_size, trans_matrix

def create_tdi(tracts, dimensions, voxel_size):
    """
    Create Tract Density Image from tract data
    """
    print("\nCreating TDI...")

    dim_x, dim_y, dim_z = dimensions
    tdi = np.zeros((dim_x, dim_y, dim_z), dtype=np.float32)

    # Count tract passages through each voxel
    for i, tract in enumerate(tracts):
        # Convert tract coordinates to voxel indices
        voxel_indices = tract / np.array(voxel_size)
        voxel_indices = voxel_indices.astype(int)

        # Clip to valid range
        voxel_indices[:, 0] = np.clip(voxel_indices[:, 0], 0, dim_x - 1)
        voxel_indices[:, 1] = np.clip(voxel_indices[:, 1], 0, dim_y - 1)
        voxel_indices[:, 2] = np.clip(voxel_indices[:, 2], 0, dim_z - 1)

        # Increment TDI for each voxel the tract passes through
        for idx in voxel_indices:
            tdi[idx[0], idx[1], idx[2]] += 1

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{len(tracts)} tracts...")

    print(f"  TDI range: {tdi.min():.0f} - {tdi.max():.0f}")
    print(f"  Non-zero voxels: {np.sum(tdi > 0)}")

    return tdi

def visualize_tdi(tdi, subject_id, output_dir):
    """
    Visualize TDI in three orientations
    """
    print(f"\nVisualizing TDI for {subject_id}...")

    # Use log scale for better visualization
    tdi_log = np.log10(tdi + 1)

    # Select slices
    axial_slice = int(tdi.shape[2] * 0.55)
    coronal_slice = int(tdi.shape[1] * 0.50)
    sagittal_slice = int(tdi.shape[0] * 0.50)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'{subject_id} - Tract Density Image (TDI)\nWhole Brain Tractography',
                 fontsize=18, fontweight='bold')

    # Row 1: Single slice views
    # Axial
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(tdi_log[:, :, axial_slice].T, cmap='hot', origin='lower', aspect='auto')
    ax1.set_title(f'Axial (Z={axial_slice})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Left ← → Right', fontsize=11)
    ax1.set_ylabel('Posterior ← → Anterior', fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='log10(Tract Density + 1)')

    # Coronal
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(tdi_log[:, coronal_slice, :].T, cmap='hot', origin='lower', aspect='auto')
    ax2.set_title(f'Coronal (Y={coronal_slice})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Left ← → Right', fontsize=11)
    ax2.set_ylabel('Inferior ← → Superior', fontsize=11)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='log10(Tract Density + 1)')

    # Sagittal
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(tdi_log[sagittal_slice, :, :].T, cmap='hot', origin='lower', aspect='auto')
    ax3.set_title(f'Sagittal (X={sagittal_slice}, Midline)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Posterior ← → Anterior', fontsize=11)
    ax3.set_ylabel('Inferior ← → Superior', fontsize=11)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='log10(Tract Density + 1)')

    # Row 2: Multiple slices - Axial
    ax4 = fig.add_subplot(gs[1, :])
    axial_slices = [int(tdi.shape[2] * p) for p in [0.35, 0.45, 0.55, 0.65]]
    axial_montage = np.hstack([tdi_log[:, :, s].T for s in axial_slices])
    im4 = ax4.imshow(axial_montage, cmap='hot', origin='lower', aspect='auto')
    ax4.set_title('Axial - Multiple Slices (Inferior → Superior)', fontsize=14, fontweight='bold')
    ax4.set_yticks([])
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='log10(Tract Density + 1)')

    # Row 3: Multiple slices - Coronal
    ax5 = fig.add_subplot(gs[2, :])
    coronal_slices = [int(tdi.shape[1] * p) for p in [0.35, 0.45, 0.55, 0.65]]
    coronal_montage = np.hstack([tdi_log[:, s, :].T for s in coronal_slices])
    im5 = ax5.imshow(coronal_montage, cmap='hot', origin='lower', aspect='auto')
    ax5.set_title('Coronal - Multiple Slices (Posterior → Anterior)', fontsize=14, fontweight='bold')
    ax5.set_yticks([])
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='log10(Tract Density + 1)')

    plt.tight_layout()

    output_path = os.path.join(output_dir, subject_id, f'{subject_id}_TDI_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()

def analyze_high_density_regions(tdi, subject_id):
    """
    Analyze and report high density regions
    """
    print(f"\n{'='*80}")
    print(f"HIGH-DENSITY WHITE MATTER REGIONS ANALYSIS - {subject_id}")
    print(f"{'='*80}")

    # Calculate statistics
    print("\nTDI Statistics:")
    print(f"  Maximum density: {tdi.max():.0f} tract passages")
    print(f"  Mean density (non-zero): {tdi[tdi > 0].mean():.2f}")
    print(f"  Median density (non-zero): {np.median(tdi[tdi > 0]):.2f}")

    # Find high-density regions (top 1%)
    threshold = np.percentile(tdi[tdi > 0], 99)
    print(f"\n  99th percentile threshold: {threshold:.0f}")

    high_density_mask = tdi > threshold
    n_high_density = np.sum(high_density_mask)
    print(f"  High-density voxels: {n_high_density}")

    # Analyze spatial distribution
    print("\nHigh-Density Region Distribution:")

    # Get center of mass for high-density regions
    coords = np.where(high_density_mask)
    if len(coords[0]) > 0:
        center_x = np.mean(coords[0])
        center_y = np.mean(coords[1])
        center_z = np.mean(coords[2])

        print(f"  Center of mass: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")

        # Analyze distribution across brain regions
        # Central regions (corpus callosum area)
        central_mask = (coords[0] > tdi.shape[0]*0.4) & (coords[0] < tdi.shape[0]*0.6)
        central_count = np.sum(central_mask)

        print(f"\n  Central region (likely corpus callosum): {central_count} voxels")
        print(f"    ({central_count/n_high_density*100:.1f}% of high-density voxels)")

    print("\nExpected High-Density Regions in Healthy Brain:")
    print("  1. Corpus Callosum - Major interhemispheric commissure")
    print("     - Connects left and right hemispheres")
    print("     - Highest density of crossing fibers")
    print("\n  2. Internal Capsule - Major projection pathway")
    print("     - Connects cortex to subcortical structures")
    print("     - Convergence point for motor/sensory tracts")
    print("\n  3. Corona Radiata - Fan-shaped projection fibers")
    print("     - Connects cortex to internal capsule")
    print("     - High density due to converging pathways")
    print("\n  4. Superior Longitudinal Fasciculus - Association fiber")
    print("     - Connects frontal, parietal, occipital lobes")
    print("     - Long-range intrahemispheric connections")
    print("\n  5. Brainstem - All descending/ascending tracts pass through")
    print("     - Extremely high density region")
    print("     - Critical bottleneck for connectivity")

    print(f"\n{'='*80}")

    return threshold

if __name__ == "__main__":
    results_dir = r"C:\Users\Public\human_brain_lecture\results"
    subjects = ['sub-010002', 'sub-010015']

    print("="*80)
    print("TDI GENERATION AND ANALYSIS")
    print("="*80)

    for subject_id in subjects:
        print(f"\n\nProcessing {subject_id}...")
        print("-"*80)

        tt_gz_path = os.path.join(results_dir, subject_id, f'{subject_id}_whole_brain.tt.gz')

        # Load tracts
        try:
            tracts, dimensions, voxel_size, trans_matrix = load_dsi_studio_tracts(tt_gz_path)

            # Create TDI
            tdi = create_tdi(tracts, dimensions, voxel_size)

            # Save TDI as NIfTI
            tdi_nifti = nib.Nifti1Image(tdi, affine=trans_matrix)
            tdi_path = os.path.join(results_dir, subject_id, f'{subject_id}_TDI.nii.gz')
            nib.save(tdi_nifti, tdi_path)
            print(f"\n  Saved TDI: {tdi_path}")

            # Visualize
            visualize_tdi(tdi, subject_id, results_dir)

            # Analyze
            threshold = analyze_high_density_regions(tdi, subject_id)

        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\nTDI generation and analysis complete!")
