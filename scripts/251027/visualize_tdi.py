"""
Visualize and analyze Tract Density Images (TDI)
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_and_analyze_tdi(tdi_path, subject_id, output_dir):
    """
    Visualize TDI and analyze high-density regions
    """
    print(f"\nProcessing {subject_id}...")
    print("-"*80)

    # Load TDI
    print(f"Loading TDI from {tdi_path}...")
    tdi_img = nib.load(tdi_path)
    tdi = tdi_img.get_fdata()

    print(f"  TDI shape: {tdi.shape}")
    print(f"  TDI range: {tdi.min():.0f} - {tdi.max():.0f}")
    print(f"  Mean density (non-zero): {tdi[tdi > 0].mean():.2f}")
    print(f"  Median density (non-zero): {np.median(tdi[tdi > 0]):.2f}")

    # Use log scale for better visualization
    tdi_log = np.log10(tdi + 1)

    # Select slices showing major white matter structures
    axial_slice = int(tdi.shape[2] * 0.55)  # Corpus callosum level
    coronal_slice = int(tdi.shape[1] * 0.50)  # Midline
    sagittal_slice = int(tdi.shape[0] * 0.50)  # Midline

    # Create comprehensive figure
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    fig.suptitle(f'{subject_id} - Tract Density Image (TDI)\n' +
                 f'Whole Brain Tractography (335k+ tracts)',
                 fontsize=18, fontweight='bold')

    # Row 1: Single slice views with annotations
    # Axial
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(tdi_log[:, :, axial_slice].T, cmap='hot', origin='lower', aspect='auto')
    ax1.set_title(f'Axial (Z={axial_slice})\nCorpus Callosum Level',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Left ← → Right', fontsize=11)
    ax1.set_ylabel('Posterior ← → Anterior', fontsize=11)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('log10(Tract Count + 1)', fontsize=10)

    # Annotate corpus callosum
    height, width = tdi_log[:, :, axial_slice].T.shape
    ax1.text(width//2, height*0.55, '← Corpus Callosum →',
             ha='center', va='center', color='cyan', fontsize=11,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    # Coronal
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(tdi_log[:, coronal_slice, :].T, cmap='hot', origin='lower', aspect='auto')
    ax2.set_title(f'Coronal (Y={coronal_slice})\nMidline View',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Left ← → Right', fontsize=11)
    ax2.set_ylabel('Inferior ← → Superior', fontsize=11)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('log10(Tract Count + 1)', fontsize=10)

    # Annotate internal capsule
    height, width = tdi_log[:, coronal_slice, :].T.shape
    ax2.text(width*0.35, height*0.45, 'IC', ha='center', va='center',
             color='yellow', fontsize=10, fontweight='bold')
    ax2.text(width*0.65, height*0.45, 'IC', ha='center', va='center',
             color='yellow', fontsize=10, fontweight='bold')

    # Sagittal
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(tdi_log[sagittal_slice, :, :].T, cmap='hot', origin='lower', aspect='auto')
    ax3.set_title(f'Sagittal (X={sagittal_slice})\nMidline - Corpus Callosum',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Posterior ← → Anterior', fontsize=11)
    ax3.set_ylabel('Inferior ← → Superior', fontsize=11)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label('log10(Tract Count + 1)', fontsize=10)

    # Row 2: Multiple axial slices
    ax4 = fig.add_subplot(gs[1, :])
    axial_slices = [int(tdi.shape[2] * p) for p in [0.35, 0.45, 0.55, 0.65]]
    axial_montage = np.hstack([tdi_log[:, :, s].T for s in axial_slices])
    im4 = ax4.imshow(axial_montage, cmap='hot', origin='lower', aspect='auto')
    ax4.set_title('Axial Slices: Inferior (left) → Superior (right)',
                  fontsize=14, fontweight='bold')
    ax4.set_yticks([])
    ax4.set_xticks([])
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.03, pad=0.02)
    cb4.set_label('log10(Tract Count + 1)', fontsize=10)

    # Row 3: Multiple coronal slices
    ax5 = fig.add_subplot(gs[2, :])
    coronal_slices = [int(tdi.shape[1] * p) for p in [0.35, 0.45, 0.55, 0.65]]
    coronal_montage = np.hstack([tdi_log[:, s, :].T for s in coronal_slices])
    im5 = ax5.imshow(coronal_montage, cmap='hot', origin='lower', aspect='auto')
    ax5.set_title('Coronal Slices: Posterior (left) → Anterior (right)',
                  fontsize=14, fontweight='bold')
    ax5.set_yticks([])
    ax5.set_xticks([])
    cb5 = plt.colorbar(im5, ax=ax5, fraction=0.03, pad=0.02)
    cb5.set_label('log10(Tract Count + 1)', fontsize=10)

    # Row 4: Statistics and high-density analysis
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.axis('off')

    # Calculate statistics
    stats_text = "TDI STATISTICS\\n" + "="*40 + "\\n\\n"
    stats_text += f"Total tracts: 335,224\\n"
    stats_text += f"Max density: {tdi.max():.0f} passages\\n"
    stats_text += f"Mean (non-zero): {tdi[tdi > 0].mean():.1f}\\n"
    stats_text += f"Median (non-zero): {np.median(tdi[tdi > 0]):.1f}\\n\\n"

    # Percentiles
    p99 = np.percentile(tdi[tdi > 0], 99)
    p95 = np.percentile(tdi[tdi > 0], 95)
    p90 = np.percentile(tdi[tdi > 0], 90)

    stats_text += f"99th percentile: {p99:.0f}\\n"
    stats_text += f"95th percentile: {p95:.0f}\\n"
    stats_text += f"90th percentile: {p90:.0f}\\n"

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Histogram
    ax7 = fig.add_subplot(gs[3, 1])
    tdi_nonzero = tdi[tdi > 0]
    ax7.hist(np.log10(tdi_nonzero + 1), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax7.set_xlabel('log10(Tract Density + 1)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('Tract Density Distribution', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # High-density regions explanation
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis('off')

    explanation_text = "HIGH-DENSITY REGIONS\\n" + "="*40 + "\\n\\n"
    explanation_text += "Expected hotspots:\\n\\n"
    explanation_text += "1. CORPUS CALLOSUM\\n"
    explanation_text += "   - Interhemispheric bridge\\n"
    explanation_text += "   - Highest density\\n\\n"
    explanation_text += "2. INTERNAL CAPSULE\\n"
    explanation_text += "   - Projection pathways\\n"
    explanation_text += "   - Motor/sensory tracts\\n\\n"
    explanation_text += "3. CORONA RADIATA\\n"
    explanation_text += "   - Fan-shaped fibers\\n"
    explanation_text += "   - Cortex ↔ subcortex\\n\\n"
    explanation_text += "4. BRAINSTEM\\n"
    explanation_text += "   - All tracts converge\\n"
    explanation_text += "   - Critical bottleneck"

    ax8.text(0.05, 0.95, explanation_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, subject_id, f'{subject_id}_TDI_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\\n  Saved: {output_path}")
    plt.close()

    return tdi, p99

def create_analysis_report(subject_id, tdi, p99, output_dir):
    """
    Create detailed analysis report
    """
    print(f"\\n{'='*80}")
    print(f"TRACT DENSITY ANALYSIS - {subject_id}")
    print(f"{'='*80}")

    print("\\n1. TRACTOGRAPHY SUMMARY:")
    print("-"*80)
    print(f"  - Total tracts generated: 335,224")
    print(f"  - FA threshold: 0.2 (white matter only)")
    print(f"  - Angle threshold: 45 degrees")
    print(f"  - Minimum fiber length: 20 mm")
    print(f"  - Algorithm: Streamline tractography")

    print("\\n2. TDI CHARACTERISTICS:")
    print("-"*80)
    print(f"  - Maximum density: {tdi.max():.0f} tract passages")
    print(f"  - Mean density (non-zero voxels): {tdi[tdi > 0].mean():.2f}")
    print(f"  - Median density: {np.median(tdi[tdi > 0]):.2f}")
    print(f"  - 99th percentile: {p99:.0f}")
    print(f"  - Non-zero voxels: {np.sum(tdi > 0):,}")

    # Identify high-density voxels
    high_density = tdi > p99
    n_high = np.sum(high_density)
    print(f"  - High-density voxels (>99th percentile): {n_high:,}")

    print("\\n3. HIGH-DENSITY WHITE MATTER REGIONS:")
    print("-"*80)
    print("\\n  A. CORPUS CALLOSUM (Interhemispheric Commissure)")
    print("     Location: Midline, superior aspect")
    print("     Function: Connects left and right hemispheres")
    print("     Characteristics:")
    print("       - Largest white matter structure")
    print("       - ~200 million axons")
    print("       - High FA and high tract density")
    print("       - Visible as bright band in axial/sagittal views")

    print("\\n  B. INTERNAL CAPSULE (Projection Fibers)")
    print("     Location: Between thalamus and basal ganglia")
    print("     Function: Major motor and sensory pathway")
    print("     Characteristics:")
    print("       - V-shaped in coronal view")
    print("       - Dense fiber convergence")
    print("       - Critical for motor control")
    print("       - Stroke here causes severe deficits")

    print("\\n  C. CORONA RADIATA (Projection Fibers)")
    print("     Location: Fans out from internal capsule to cortex")
    print("     Function: Connects cortex to subcortical structures")
    print("     Characteristics:")
    print("       - Fan or crown-like appearance")
    print("       - Connects to motor, sensory, association cortex")
    print("       - High density due to converging pathways")

    print("\\n  D. SUPERIOR LONGITUDINAL FASCICULUS (Association Fibers)")
    print("     Location: Lateral hemisphere, anterior-posterior")
    print("     Function: Connects frontal, parietal, occipital lobes")
    print("     Characteristics:")
    print("       - Long association pathway")
    print("       - Important for language (arcuate fasciculus)")
    print("       - Moderate to high density")

    print("\\n  E. BRAINSTEM (All Pathways Converge)")
    print("     Location: Inferior, connects brain to spinal cord")
    print("     Function: All ascending/descending tracts pass through")
    print("     Characteristics:")
    print("       - Extremely high density")
    print("       - Critical bottleneck")
    print("       - Damage causes widespread deficits")

    print("\\n4. INTERPRETATION:")
    print("-"*80)
    print("  The TDI reveals the brain's structural connectivity architecture.")
    print("  High-density regions represent:")
    print("    - Major white matter pathways")
    print("    - Convergence/divergence points")
    print("    - Anatomical bottlenecks")
    print("    - Critical communication hubs")

    print("\\n  Clinical significance:")
    print("    - Lesions in high-density areas cause severe deficits")
    print("    - TDI can help surgical planning")
    print("    - Age/disease affects density patterns")
    print("    - Useful for connectivity mapping")

    print(f"\\n{'='*80}")

    # Save report
    report_path = os.path.join(output_dir, subject_id, f'{subject_id}_TDI_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\\n")
        f.write(f"TRACT DENSITY ANALYSIS - {subject_id}\\n")
        f.write(f"{'='*80}\\n\\n")
        f.write(f"Maximum density: {tdi.max():.0f} tract passages\\n")
        f.write(f"Mean density (non-zero): {tdi[tdi > 0].mean():.2f}\\n")
        f.write(f"99th percentile: {p99:.0f}\\n")

    print(f"\\n  Saved report: {report_path}")

if __name__ == "__main__":
    results_dir = r"C:\\Users\\Public\\human_brain_lecture\\results"
    subjects = ['sub-010002', 'sub-010015']

    print("="*80)
    print("TDI VISUALIZATION AND ANALYSIS")
    print("="*80)

    for subject_id in subjects:
        tdi_path = os.path.join(results_dir, subject_id, f'{subject_id}_whole_brain.tt.gz.tdi.nii.gz')

        # Visualize and get statistics
        tdi, p99 = visualize_and_analyze_tdi(tdi_path, subject_id, results_dir)

        # Create analysis report
        create_analysis_report(subject_id, tdi, p99, results_dir)

    print("\\n\\nTDI analysis complete!")
