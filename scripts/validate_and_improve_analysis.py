"""
Validate FA analysis results and improve visualization with std
"""
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_simple_atlas(fa_shape):
    """
    Create a simple anatomical atlas with major white matter regions
    """
    atlas_data = np.zeros(fa_shape, dtype=np.int32)
    atlas_labels = {}

    nx, ny, nz = fa_shape

    # 1. Corpus Callosum (middle-upper region)
    cc_z_start = int(nz * 0.50)
    cc_z_end = int(nz * 0.60)
    cc_x_start = int(nx * 0.35)
    cc_x_end = int(nx * 0.65)
    cc_y_start = int(ny * 0.40)
    cc_y_end = int(ny * 0.60)
    atlas_data[cc_x_start:cc_x_end, cc_y_start:cc_y_end, cc_z_start:cc_z_end] = 1
    atlas_labels[1] = "Corpus_Callosum"

    # 2. Left Internal Capsule
    atlas_data[int(nx*0.35):int(nx*0.45), int(ny*0.45):int(ny*0.60), int(nz*0.35):int(nz*0.50)] = 2
    atlas_labels[2] = "Internal_Capsule_L"

    # 3. Right Internal Capsule
    atlas_data[int(nx*0.55):int(nx*0.65), int(ny*0.45):int(ny*0.60), int(nz*0.35):int(nz*0.50)] = 3
    atlas_labels[3] = "Internal_Capsule_R"

    # 4. Left Corona Radiata
    atlas_data[int(nx*0.25):int(nx*0.40), int(ny*0.40):int(ny*0.60), int(nz*0.55):int(nz*0.70)] = 4
    atlas_labels[4] = "Corona_Radiata_L"

    # 5. Right Corona Radiata
    atlas_data[int(nx*0.60):int(nx*0.75), int(ny*0.40):int(ny*0.60), int(nz*0.55):int(nz*0.70)] = 5
    atlas_labels[5] = "Corona_Radiata_R"

    # 6. Left Superior Longitudinal Fasciculus
    atlas_data[int(nx*0.20):int(nx*0.35), int(ny*0.30):int(ny*0.70), int(nz*0.50):int(nz*0.65)] = 6
    atlas_labels[6] = "Superior_Longitudinal_Fasciculus_L"

    # 7. Right Superior Longitudinal Fasciculus
    atlas_data[int(nx*0.65):int(nx*0.80), int(ny*0.30):int(ny*0.70), int(nz*0.50):int(nz*0.65)] = 7
    atlas_labels[7] = "Superior_Longitudinal_Fasciculus_R"

    # 8. Anterior Commissure
    atlas_data[int(nx*0.45):int(nx*0.55), int(ny*0.52):int(ny*0.58), int(nz*0.42):int(nz*0.48)] = 8
    atlas_labels[8] = "Anterior_Commissure"

    # 9. Left Cingulum
    atlas_data[int(nx*0.42):int(nx*0.48), int(ny*0.35):int(ny*0.65), int(nz*0.50):int(nz*0.65)] = 9
    atlas_labels[9] = "Cingulum_L"

    # 10. Right Cingulum
    atlas_data[int(nx*0.52):int(nx*0.58), int(ny*0.35):int(ny*0.65), int(nz*0.50):int(nz*0.65)] = 10
    atlas_labels[10] = "Cingulum_R"

    # 11. Fornix
    atlas_data[int(nx*0.47):int(nx*0.53), int(ny*0.48):int(ny*0.55), int(nz*0.48):int(nz*0.58)] = 11
    atlas_labels[11] = "Fornix"

    # 12. Left Corticospinal Tract
    atlas_data[int(nx*0.38):int(nx*0.45), int(ny*0.48):int(ny*0.62), int(nz*0.30):int(nz*0.50)] = 12
    atlas_labels[12] = "Corticospinal_Tract_L"

    # 13. Right Corticospinal Tract
    atlas_data[int(nx*0.55):int(nx*0.62), int(ny*0.48):int(ny*0.62), int(nz*0.30):int(nz*0.50)] = 13
    atlas_labels[13] = "Corticospinal_Tract_R"

    # 14. Left Uncinate Fasciculus
    atlas_data[int(nx*0.30):int(nx*0.42), int(ny*0.60):int(ny*0.75), int(nz*0.35):int(nz*0.45)] = 14
    atlas_labels[14] = "Uncinate_Fasciculus_L"

    # 15. Right Uncinate Fasciculus
    atlas_data[int(nx*0.58):int(nx*0.70), int(ny*0.60):int(ny*0.75), int(nz*0.35):int(nz*0.45)] = 15
    atlas_labels[15] = "Uncinate_Fasciculus_R"

    return atlas_data, atlas_labels

def extract_roi_fa_statistics(fa_path, atlas_data, atlas_labels):
    """
    Extract mean and std FA values for each ROI
    """
    fa_img = nib.load(fa_path)
    fa_data = fa_img.get_fdata()

    roi_stats = {}

    for roi_idx, roi_name in atlas_labels.items():
        roi_mask = (atlas_data == roi_idx)
        fa_roi = fa_data[roi_mask]

        if len(fa_roi) > 0 and np.sum(fa_roi > 0) > 0:
            fa_valid = fa_roi[fa_roi > 0]
            roi_stats[roi_name] = {
                'mean': np.mean(fa_valid),
                'std': np.std(fa_valid),
                'median': np.median(fa_valid),
                'n_voxels': len(fa_valid)
            }
        else:
            roi_stats[roi_name] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'n_voxels': 0
            }

    return roi_stats

def visualize_segmentation(fa_path, atlas_data, atlas_labels, subject_id, output_dir):
    """
    Visualize ROI segmentation overlaid on FA map
    """
    print(f"\nVisualizing segmentation for {subject_id}...")

    fa_img = nib.load(fa_path)
    fa_data = fa_img.get_fdata()

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'{subject_id} - ROI Segmentation Quality Check',
                 fontsize=18, fontweight='bold')

    # Select slices
    slices = {
        'axial': int(fa_data.shape[2] * 0.55),
        'coronal': int(fa_data.shape[1] * 0.50),
        'sagittal': int(fa_data.shape[0] * 0.50)
    }

    # Row 1: FA maps
    ax1 = axes[0, 0]
    ax1.imshow(fa_data[:, :, slices['axial']].T, cmap='gray', origin='lower')
    ax1.set_title('FA Map - Axial', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = axes[0, 1]
    ax2.imshow(fa_data[:, slices['coronal'], :].T, cmap='gray', origin='lower')
    ax2.set_title('FA Map - Coronal', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = axes[0, 2]
    ax3.imshow(fa_data[slices['sagittal'], :, :].T, cmap='gray', origin='lower')
    ax3.set_title('FA Map - Sagittal', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Row 2: ROI overlays
    ax4 = axes[1, 0]
    ax4.imshow(fa_data[:, :, slices['axial']].T, cmap='gray', origin='lower', alpha=0.7)
    atlas_slice = atlas_data[:, :, slices['axial']].T
    masked_atlas = np.ma.masked_where(atlas_slice == 0, atlas_slice)
    im1 = ax4.imshow(masked_atlas, cmap='tab20', origin='lower', alpha=0.5, vmin=0, vmax=15)
    ax4.set_title('ROI Overlay - Axial', fontsize=14, fontweight='bold')
    ax4.axis('off')

    ax5 = axes[1, 1]
    ax5.imshow(fa_data[:, slices['coronal'], :].T, cmap='gray', origin='lower', alpha=0.7)
    atlas_slice = atlas_data[:, slices['coronal'], :].T
    masked_atlas = np.ma.masked_where(atlas_slice == 0, atlas_slice)
    ax5.imshow(masked_atlas, cmap='tab20', origin='lower', alpha=0.5, vmin=0, vmax=15)
    ax5.set_title('ROI Overlay - Coronal', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = axes[1, 2]
    ax6.imshow(fa_data[slices['sagittal'], :, :].T, cmap='gray', origin='lower', alpha=0.7)
    atlas_slice = atlas_data[slices['sagittal'], :, :].T
    masked_atlas = np.ma.masked_where(atlas_slice == 0, atlas_slice)
    ax6.imshow(masked_atlas, cmap='tab20', origin='lower', alpha=0.5, vmin=0, vmax=15)
    ax6.set_title('ROI Overlay - Sagittal', fontsize=14, fontweight='bold')
    ax6.axis('off')

    # Row 3: Individual ROI highlighting (Corpus Callosum, Fornix)
    # Corpus Callosum
    ax7 = axes[2, 0]
    cc_mask = (atlas_data == 1)
    ax7.imshow(fa_data[:, :, slices['axial']].T, cmap='gray', origin='lower', alpha=0.7)
    cc_slice = cc_mask[:, :, slices['axial']].T
    masked_cc = np.ma.masked_where(~cc_slice, np.ones_like(cc_slice))
    ax7.imshow(masked_cc, cmap='Reds', origin='lower', alpha=0.6)
    ax7.set_title('Corpus Callosum (Axial)', fontsize=14, fontweight='bold')
    ax7.axis('off')

    # Fornix
    ax8 = axes[2, 1]
    fornix_mask = (atlas_data == 11)
    ax8.imshow(fa_data[:, slices['coronal'], :].T, cmap='gray', origin='lower', alpha=0.7)
    fornix_slice = fornix_mask[:, slices['coronal'], :].T
    masked_fornix = np.ma.masked_where(~fornix_slice, np.ones_like(fornix_slice))
    ax8.imshow(masked_fornix, cmap='Blues', origin='lower', alpha=0.6)
    ax8.set_title('Fornix (Coronal)', fontsize=14, fontweight='bold')
    ax8.axis('off')

    # Legend
    ax9 = axes[2, 2]
    ax9.axis('off')
    legend_text = "ROI Labels:\n\n"
    for idx, name in atlas_labels.items():
        legend_text += f"{idx:2d}. {name}\n"

    ax9.text(0.1, 0.9, legend_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = os.path.join(output_dir, subject_id, f'{subject_id}_segmentation_quality.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()

def create_improved_comparison_plot(df_comparison, sub1, sub2, roi_stats_sub1, roi_stats_sub2, results_dir):
    """
    Create improved comparison plot with std error bars
    """
    print("\n\nCreating improved visualization with std...")

    # Get Corpus Callosum and top 10 ROIs by difference
    corpus_callosum_row = df_comparison[df_comparison.index == 'Corpus_Callosum']
    other_rois = df_comparison[df_comparison.index != 'Corpus_Callosum'].head(10)
    plot_data = pd.concat([corpus_callosum_row, other_rois])

    # Extract std values
    std_sub1 = [roi_stats_sub1[roi]['std'] for roi in plot_data.index]
    std_sub2 = [roi_stats_sub2[roi]['std'] for roi in plot_data.index]

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    fig.suptitle('FA Value Comparison Between Subjects (with Standard Deviation)',
                 fontsize=18, fontweight='bold')

    # Plot 1: Bar plot with error bars
    ax1 = fig.add_subplot(gs[0, :])

    x = np.arange(len(plot_data))
    width = 0.35

    bars1 = ax1.bar(x - width/2, plot_data[f'{sub1}_FA'], width,
                    yerr=std_sub1, capsize=5,
                    label=f'{sub1} (Elderly)', alpha=0.8, color='steelblue',
                    error_kw={'linewidth': 2, 'ecolor': 'darkblue'})
    bars2 = ax1.bar(x + width/2, plot_data[f'{sub2}_FA'], width,
                    yerr=std_sub2, capsize=5,
                    label=f'{sub2} (Young)', alpha=0.8, color='coral',
                    error_kw={'linewidth': 2, 'ecolor': 'darkred'})

    ax1.set_xlabel('ROI', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean FA Value ± SD', fontsize=14, fontweight='bold')
    ax1.set_title('FA Values Comparison: Corpus Callosum + Top 10 Different ROIs',
                  fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=13, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(plot_data[[f'{sub1}_FA', f'{sub2}_FA']].max()) * 1.3)

    # Highlight Corpus Callosum
    bars1[0].set_color('darkblue')
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(3)
    bars2[0].set_color('darkred')
    bars2[0].set_edgecolor('black')
    bars2[0].set_linewidth(3)

    # Plot 2: Difference with std propagation
    ax2 = fig.add_subplot(gs[1, 0])

    # Calculate std of difference (propagation of uncertainty)
    std_diff = np.sqrt(np.array(std_sub1)**2 + np.array(std_sub2)**2)

    colors = ['darkred' if roi == 'Corpus_Callosum' else 'steelblue' for roi in plot_data.index]
    bars = ax2.barh(plot_data.index, plot_data['Difference'], xerr=std_diff,
                    color=colors, alpha=0.7, capsize=5,
                    error_kw={'linewidth': 2})

    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(3)

    ax2.set_xlabel('FA Difference (Young - Elderly) ± SD', fontsize=12, fontweight='bold')
    ax2.set_title('FA Differences with Uncertainty', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Mean FA comparison
    ax3 = fig.add_subplot(gs[1, 1])

    mean_sub1 = plot_data[f'{sub1}_FA'].mean()
    mean_sub2 = plot_data[f'{sub2}_FA'].mean()
    std_mean_sub1 = np.mean(std_sub1)
    std_mean_sub2 = np.mean(std_sub2)

    means = [mean_sub1, mean_sub2]
    stds = [std_mean_sub1, std_mean_sub2]
    labels_mean = [f'{sub1}\n(Elderly)', f'{sub2}\n(Young)']
    colors_mean = ['steelblue', 'coral']

    bars_mean = ax3.bar(labels_mean, means, yerr=stds, capsize=10,
                        color=colors_mean, alpha=0.8,
                        edgecolor='black', linewidth=2,
                        error_kw={'linewidth': 3, 'ecolor': 'black'})

    for bar, mean_val in zip(bars_mean, means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=15, fontweight='bold')

    ax3.set_ylabel('Mean FA Value ± SD', fontsize=13, fontweight='bold')
    ax3.set_title(f'Overall Mean FA Comparison\n(Difference: {mean_sub2-mean_sub1:+.4f})',
                  fontsize=14, fontweight='bold')
    ax3.set_ylim(0, max(means) * 1.25)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Coefficient of Variation comparison
    ax4 = fig.add_subplot(gs[2, :])

    cv_sub1 = [roi_stats_sub1[roi]['std'] / roi_stats_sub1[roi]['mean'] * 100
               for roi in plot_data.index]
    cv_sub2 = [roi_stats_sub2[roi]['std'] / roi_stats_sub2[roi]['mean'] * 100
               for roi in plot_data.index]

    x = np.arange(len(plot_data))
    width = 0.35

    ax4.bar(x - width/2, cv_sub1, width, label=f'{sub1} (Elderly)',
            alpha=0.8, color='steelblue')
    ax4.bar(x + width/2, cv_sub2, width, label=f'{sub2} (Young)',
            alpha=0.8, color='coral')

    ax4.set_xlabel('ROI', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Coefficient of Variation (%)', fontsize=14, fontweight='bold')
    ax4.set_title('Variability Within ROIs (CV = SD/Mean × 100%)',
                  fontsize=15, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=11)
    ax4.legend(fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(results_dir, 'FA_ROI_comparison_plot_with_std.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved improved comparison plot: {output_path}")
    plt.close()

def generate_validation_report(df_comparison, roi_stats_sub1, roi_stats_sub2, results_dir):
    """
    Generate validation report comparing with literature
    """
    print("\n" + "="*80)
    print("VALIDATION REPORT: Age-Related FA Changes")
    print("="*80)

    report = []

    report.append("\n1. LITERATURE COMPARISON:")
    report.append("-" * 80)
    report.append("\nExpected age-related FA changes (from literature):")
    report.append("  - Corpus Callosum: Elderly typically show 5-15% LOWER FA than young adults")
    report.append("  - Fornix: Highly vulnerable to aging, 15-30% LOWER FA in elderly")
    report.append("  - Cingulum: 10-20% LOWER FA in elderly")
    report.append("  - Corona Radiata: 8-15% LOWER FA in elderly")
    report.append("\nKey References:")
    report.append("  * Sullivan et al. (2010) - Diffusion tensor imaging and aging")
    report.append("  * Bennett & Madden (2014) - Disconnected aging")
    report.append("  * Burzynska et al. (2010) - Age-related differences in white matter")

    report.append("\n\n2. OUR RESULTS:")
    report.append("-" * 80)

    # Corpus Callosum
    cc_diff = df_comparison.loc['Corpus_Callosum', 'Percent_Change']
    report.append(f"\n  Corpus Callosum:")
    report.append(f"    - Elderly (sub-010002): {roi_stats_sub1['Corpus_Callosum']['mean']:.4f} ± {roi_stats_sub1['Corpus_Callosum']['std']:.4f}")
    report.append(f"    - Young (sub-010015):   {roi_stats_sub2['Corpus_Callosum']['mean']:.4f} ± {roi_stats_sub2['Corpus_Callosum']['std']:.4f}")
    report.append(f"    - Change: {cc_diff:.2f}%")
    if cc_diff > -5:
        report.append(f"    [!] WARNING: Expected decrease not observed!")
        report.append(f"    -> Possible reasons: Individual variation, segmentation quality")
    else:
        report.append(f"    [OK] Consistent with literature (decreased in elderly)")

    # Fornix
    fornix_diff = df_comparison.loc['Fornix', 'Percent_Change']
    report.append(f"\n  Fornix:")
    report.append(f"    - Elderly (sub-010002): {roi_stats_sub1['Fornix']['mean']:.4f} ± {roi_stats_sub1['Fornix']['std']:.4f}")
    report.append(f"    - Young (sub-010015):   {roi_stats_sub2['Fornix']['mean']:.4f} ± {roi_stats_sub2['Fornix']['std']:.4f}")
    report.append(f"    - Change: {fornix_diff:.2f}%")
    if fornix_diff > 0:
        report.append(f"    [!] WARNING: Unexpected increase in young subject!")
        report.append(f"    -> Likely explanation: Segmentation issue (fornix is small and difficult to segment)")
    else:
        report.append(f"    [OK] Consistent with literature (decreased in elderly)")

    report.append("\n\n3. SEGMENTATION QUALITY ASSESSMENT:")
    report.append("-" * 80)
    report.append("\nSegmentation Method: Anatomically-defined box ROIs")
    report.append("Limitations:")
    report.append("  1. Simple geometric boxes - not using true anatomical atlas")
    report.append("  2. No registration to template space")
    report.append("  3. Fornix is very small - likely poor spatial specificity")
    report.append("  4. Individual anatomical variation not accounted for")

    report.append("\n\nROI Quality Metrics:")
    for roi in ['Corpus_Callosum', 'Fornix', 'Cingulum_L', 'Cingulum_R']:
        n_vox_1 = roi_stats_sub1[roi]['n_voxels']
        n_vox_2 = roi_stats_sub2[roi]['n_voxels']
        cv_1 = roi_stats_sub1[roi]['std'] / roi_stats_sub1[roi]['mean'] * 100
        cv_2 = roi_stats_sub2[roi]['std'] / roi_stats_sub2[roi]['mean'] * 100

        report.append(f"\n  {roi}:")
        report.append(f"    - Voxels: {n_vox_1} (elderly), {n_vox_2} (young)")
        report.append(f"    - CV: {cv_1:.1f}% (elderly), {cv_2:.1f}% (young)")

        if n_vox_1 < 100 or n_vox_2 < 100:
            report.append(f"    [!] Small ROI - may have poor reliability")
        if cv_1 > 40 or cv_2 > 40:
            report.append(f"    [!] High variability - check segmentation")

    report.append("\n\n4. FINAL VERDICT:")
    report.append("="*80)
    report.append("\nReliability Assessment:")

    # Calculate score
    score = 0
    issues = []

    if abs(cc_diff) > 5:
        issues.append("- Corpus callosum shows minimal age effect (expected stronger decrease)")
    else:
        score += 1

    if fornix_diff > 0:
        issues.append("- Fornix shows INCREASE in young (opposite of expected)")
        issues.append("  → Segmentation likely unreliable for small structures")
    else:
        score += 1

    # Check if most ROIs follow expected pattern
    expected_decrease_rois = ['Corona_Radiata_L', 'Corona_Radiata_R', 'Cingulum_L', 'Cingulum_R']
    decreased = sum(1 for roi in expected_decrease_rois
                   if roi in df_comparison.index and df_comparison.loc[roi, 'Percent_Change'] < 0)

    if decreased >= 2:
        score += 1
        report.append("\n  [OK] Majority of expected ROIs show age-related decrease")
    else:
        issues.append("- Many ROIs show unexpected patterns")

    report.append(f"\n\nConfidence Score: {score}/3")

    if score >= 2:
        report.append("\n[OK] MODERATELY RELIABLE")
        report.append("  - Results partially consistent with literature")
        report.append("  - Some discrepancies likely due to:")
        report.append("    1. Simplified segmentation method")
        report.append("    2. Small sample size (n=2)")
        report.append("    3. Individual variation")
    else:
        report.append("\n[!] LOW RELIABILITY")
        report.append("  - Several inconsistencies with expected age effects")
        report.append("  - Major limitations:")

    if issues:
        report.append("\n\nIssues Found:")
        for issue in issues:
            report.append(f"  {issue}")

    report.append("\n\nRECOMMENDATIONS:")
    report.append("-" * 80)
    report.append("For more reliable results:")
    report.append("  1. Use validated atlas (e.g., JHU DTI atlas, FreeSurfer wmparc)")
    report.append("  2. Register FA maps to standard template")
    report.append("  3. Use automated tractography-based segmentation (e.g., in DSI Studio)")
    report.append("  4. Increase sample size")
    report.append("  5. Control for head motion and data quality")

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_path = os.path.join(results_dir, 'validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n\nSaved validation report: {report_path}")

    return report_text

if __name__ == "__main__":
    results_dir = r"C:\Users\Public\human_brain_lecture\results"
    subjects = ['sub-010002', 'sub-010015']

    print("="*80)
    print("VALIDATION AND IMPROVED ANALYSIS")
    print("="*80)

    # Load FA data
    fa_paths = {}
    for subject_id in subjects:
        fa_paths[subject_id] = os.path.join(results_dir, subject_id,
                                            f'{subject_id}.fib.gz.fa.nii.gz')

    # Create atlas
    first_fa = nib.load(fa_paths[subjects[0]])
    fa_shape = first_fa.shape
    atlas_data, atlas_labels = create_simple_atlas(fa_shape)

    # Visualize segmentation for both subjects
    for subject_id in subjects:
        visualize_segmentation(fa_paths[subject_id], atlas_data, atlas_labels,
                              subject_id, results_dir)

    # Extract statistics
    roi_stats = {}
    for subject_id in subjects:
        print(f"\nExtracting statistics for {subject_id}...")
        roi_stats[subject_id] = extract_roi_fa_statistics(fa_paths[subject_id],
                                                          atlas_data, atlas_labels)

    # Load comparison table
    df_comparison = pd.read_csv(os.path.join(results_dir, 'FA_ROI_comparison_table.csv'),
                               index_col=0)

    # Create improved plot
    create_improved_comparison_plot(df_comparison, subjects[0], subjects[1],
                                   roi_stats[subjects[0]], roi_stats[subjects[1]],
                                   results_dir)

    # Generate validation report
    generate_validation_report(df_comparison, roi_stats[subjects[0]],
                              roi_stats[subjects[1]], results_dir)

    print("\n\nValidation and improvement complete!")
