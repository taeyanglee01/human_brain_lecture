"""
Compare FA values across ROIs for two subjects using atlas-based segmentation
"""
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets
from nilearn.image import resample_to_img
import os

def extract_roi_fa_values(fa_path, atlas_data, atlas_labels):
    """
    Extract mean FA values for each ROI

    Parameters:
    - fa_path: path to FA NIfTI file
    - atlas_data: atlas volume data
    - atlas_labels: dictionary mapping ROI index to name

    Returns:
    - Dictionary of ROI names to mean FA values
    """
    # Load FA data
    fa_img = nib.load(fa_path)
    fa_data = fa_img.get_fdata()

    roi_values = {}

    for roi_idx, roi_name in atlas_labels.items():
        # Get mask for this ROI
        roi_mask = (atlas_data == roi_idx)

        # Extract FA values within this ROI
        fa_roi = fa_data[roi_mask]

        # Calculate mean FA (excluding zeros)
        if len(fa_roi) > 0 and np.sum(fa_roi > 0) > 0:
            mean_fa = np.mean(fa_roi[fa_roi > 0])
            roi_values[roi_name] = mean_fa
        else:
            roi_values[roi_name] = 0.0

    return roi_values

def create_simple_atlas(fa_shape):
    """
    Create a simple anatomical atlas with major white matter regions
    This is a simplified version for demonstration
    """
    atlas_data = np.zeros(fa_shape, dtype=np.int32)

    # Define ROI labels
    atlas_labels = {}

    # Get dimensions
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

def compare_fa_roi(results_dir, subjects):
    """
    Compare FA values across ROIs for multiple subjects
    """
    print("="*60)
    print("FA ROI-based Comparison")
    print("="*60)

    # Load FA data for both subjects
    fa_data_dict = {}

    for subject_id in subjects:
        fa_path = os.path.join(results_dir, subject_id, f'{subject_id}.fib.gz.fa.nii.gz')
        print(f"\nLoading FA data for {subject_id}...")

        fa_img = nib.load(fa_path)
        fa_data_dict[subject_id] = fa_img

    # Create atlas based on first subject's dimensions
    first_subject = subjects[0]
    fa_shape = fa_data_dict[first_subject].shape
    print(f"\nCreating anatomical atlas (shape: {fa_shape})...")

    atlas_data, atlas_labels = create_simple_atlas(fa_shape)
    print(f"Number of ROIs: {len(atlas_labels)}")

    # Extract ROI FA values for each subject
    all_roi_values = {}

    for subject_id in subjects:
        fa_path = os.path.join(results_dir, subject_id, f'{subject_id}.fib.gz.fa.nii.gz')
        print(f"\nExtracting ROI values for {subject_id}...")

        roi_values = extract_roi_fa_values(fa_path, atlas_data, atlas_labels)
        all_roi_values[subject_id] = roi_values

        # Print summary
        print(f"  ROIs with valid FA values: {sum(1 for v in roi_values.values() if v > 0)}")

    # Create comparison DataFrame
    df = pd.DataFrame(all_roi_values).T
    df.index.name = 'Subject'

    # Calculate differences
    sub1, sub2 = subjects[0], subjects[1]
    df_comparison = pd.DataFrame({
        f'{sub1}_FA': df.loc[sub1],
        f'{sub2}_FA': df.loc[sub2],
        'Difference': df.loc[sub2] - df.loc[sub1],
        'Abs_Difference': np.abs(df.loc[sub2] - df.loc[sub1]),
        'Percent_Change': ((df.loc[sub2] - df.loc[sub1]) / df.loc[sub1] * 100).round(2)
    })

    # Filter out ROIs with zero values
    df_comparison = df_comparison[df_comparison[f'{sub1}_FA'] > 0]

    # Sort by absolute difference
    df_comparison = df_comparison.sort_values('Abs_Difference', ascending=False)

    # Save full comparison table
    output_csv = os.path.join(results_dir, 'FA_ROI_comparison_table.csv')
    df_comparison.to_csv(output_csv)
    print(f"\n\nSaved comparison table: {output_csv}")

    # Print table
    print("\n" + "="*80)
    print("FA Comparison Table (Top ROIs by difference)")
    print("="*80)
    print(df_comparison.head(15).to_string())

    return df_comparison, sub1, sub2

def visualize_fa_comparison(df_comparison, sub1, sub2, results_dir):
    """
    Visualize FA comparison focusing on Corpus Callosum and top 10 different ROIs
    """
    print("\n\nCreating visualization...")

    # Get Corpus Callosum and top 10 ROIs by difference
    corpus_callosum_row = df_comparison[df_comparison.index == 'Corpus_Callosum']

    # Get top 10 ROIs excluding Corpus Callosum
    other_rois = df_comparison[df_comparison.index != 'Corpus_Callosum'].head(10)

    # Combine
    plot_data = pd.concat([corpus_callosum_row, other_rois])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('FA Value Comparison Between Subjects', fontsize=18, fontweight='bold')

    # Plot 1: Bar plot of FA values for both subjects
    ax1 = fig.add_subplot(gs[0, :])

    x = np.arange(len(plot_data))
    width = 0.35

    bars1 = ax1.bar(x - width/2, plot_data[f'{sub1}_FA'], width,
                    label=sub1, alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, plot_data[f'{sub2}_FA'], width,
                    label=sub2, alpha=0.8, color='coral')

    ax1.set_xlabel('ROI', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean FA Value', fontsize=13, fontweight='bold')
    ax1.set_title('FA Values Comparison: Corpus Callosum + Top 10 Different ROIs',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(plot_data[[f'{sub1}_FA', f'{sub2}_FA']].max()) * 1.1)

    # Highlight Corpus Callosum
    bars1[0].set_color('darkblue')
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(2)
    bars2[0].set_color('darkred')
    bars2[0].set_edgecolor('black')
    bars2[0].set_linewidth(2)

    # Plot 2: Difference plot
    ax2 = fig.add_subplot(gs[1, 0])

    colors = ['darkred' if roi == 'Corpus_Callosum' else 'steelblue' for roi in plot_data.index]
    bars = ax2.barh(plot_data.index, plot_data['Difference'], color=colors, alpha=0.7)

    # Highlight Corpus Callosum
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)

    ax2.set_xlabel('FA Difference (Sub-010015 - Sub-010002)', fontsize=11, fontweight='bold')
    ax2.set_title('Absolute FA Differences', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Mean FA comparison with overall difference
    ax3 = fig.add_subplot(gs[1, 1])

    # Calculate overall means
    mean_sub1 = plot_data[f'{sub1}_FA'].mean()
    mean_sub2 = plot_data[f'{sub2}_FA'].mean()
    mean_diff = mean_sub2 - mean_sub1

    # Bar plot for means
    means = [mean_sub1, mean_sub2]
    labels_mean = [sub1, sub2]
    colors_mean = ['steelblue', 'coral']

    bars_mean = ax3.bar(labels_mean, means, color=colors_mean, alpha=0.8,
                        edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, mean_val in zip(bars_mean, means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax3.set_ylabel('Mean FA Value', fontsize=13, fontweight='bold')
    ax3.set_title(f'Overall Mean FA\n(Difference: {mean_diff:+.4f})',
                  fontsize=13, fontweight='bold')
    ax3.set_ylim(0, max(means) * 1.15)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(results_dir, 'FA_ROI_comparison_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison plot: {output_path}")
    plt.close()

    # Create a detailed table figure
    fig2, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = plot_data[[f'{sub1}_FA', f'{sub2}_FA', 'Difference', 'Percent_Change']].round(4)
    table_data.columns = [f'{sub1}\nFA', f'{sub2}\nFA', 'Difference\n(Sub2-Sub1)', 'Change\n(%)']

    # Create table
    table = ax.table(cellText=table_data.values,
                    rowLabels=table_data.index,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    rowLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style row labels
    for i in range(1, len(table_data) + 1):
        table[(i, -1)].set_facecolor('#E7E6E6')
        table[(i, -1)].set_text_props(weight='bold')

        # Highlight Corpus Callosum row
        if table_data.index[i-1] == 'Corpus_Callosum':
            for j in range(-1, len(table_data.columns)):
                table[(i, j)].set_facecolor('#FFD966')
                table[(i, j)].set_text_props(weight='bold')

    plt.title('FA ROI Comparison Table\nCorpus Callosum + Top 10 Different ROIs',
              fontsize=16, fontweight='bold', pad=20)

    output_path_table = os.path.join(results_dir, 'FA_ROI_comparison_table.png')
    plt.savefig(output_path_table, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison table image: {output_path_table}")
    plt.close()

if __name__ == "__main__":
    # Configuration
    results_dir = r"C:\Users\Public\human_brain_lecture\results"
    subjects = ['sub-010002', 'sub-010015']

    # Compare FA values
    df_comparison, sub1, sub2 = compare_fa_roi(results_dir, subjects)

    # Visualize comparison
    visualize_fa_comparison(df_comparison, sub1, sub2, results_dir)

    print("\n\nFA ROI comparison complete!")
