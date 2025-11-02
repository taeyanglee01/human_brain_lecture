#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze correlation between behavioral scores and FA values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def load_behavioral_data(metadata_dir):
    """
    Load and process behavioral data

    Returns:
    --------
    DataFrame with columns: Subject, ARI_total, DERS_total, ERQ_reappraisal, ERQ_suppression, ERQ_total
    """
    # Load TSV files
    ari_df = pd.read_csv(metadata_dir / 'ari.tsv', sep='\t')
    ders_df = pd.read_csv(metadata_dir / 'ders.tsv', sep='\t')
    erq_df = pd.read_csv(metadata_dir / 'erq_s.tsv', sep='\t')

    # Initialize results
    behavioral_scores = []

    # Get subjects (exclude PILOT subjects)
    subjects = ari_df['participant_id'].values
    subjects = [s for s in subjects if isinstance(s, str) and not s.startswith('sub-PILOT')]

    for subject in subjects:
        # Get ARI total (sum of ari_01 to ari_07)
        ari_row = ari_df[ari_df['participant_id'] == subject]
        if len(ari_row) > 0:
            ari_cols = [c for c in ari_df.columns if c.startswith('ari_') and c != 'ari_comments']
            ari_values = ari_row[ari_cols].values[0]
            # Convert to numeric, replacing 'n/a' with NaN
            ari_values = pd.to_numeric(ari_values, errors='coerce')
            ari_total = np.nansum(ari_values)
        else:
            ari_total = np.nan

        # Get DERS total (sum of ders_01 to ders_05)
        ders_row = ders_df[ders_df['participant_id'] == subject]
        if len(ders_row) > 0:
            ders_cols = [c for c in ders_df.columns if c.startswith('ders_')]
            ders_values = ders_row[ders_cols].values[0]
            ders_values = pd.to_numeric(ders_values, errors='coerce')
            ders_total = np.nansum(ders_values)
        else:
            ders_total = np.nan

        # Get ERQ scores
        erq_row = erq_df[erq_df['participant_id'] == subject]
        if len(erq_row) > 0:
            erq_values = erq_row[['erq_s_01', 'erq_s_02', 'erq_s_03',
                                  'erq_s_04', 'erq_s_05', 'erq_s_06']].values[0]
            erq_values = pd.to_numeric(erq_values, errors='coerce')

            # ERQ Reappraisal: items 1, 3, 5
            erq_reappraisal = np.nansum(erq_values[[0, 2, 4]])

            # ERQ Suppression: items 2, 4, 6
            erq_suppression = np.nansum(erq_values[[1, 3, 5]])

            # ERQ Total
            erq_total = np.nansum(erq_values)
        else:
            erq_reappraisal = np.nan
            erq_suppression = np.nan
            erq_total = np.nan

        behavioral_scores.append({
            'Subject': subject,
            'ARI_total': ari_total,
            'DERS_total': ders_total,
            'ERQ_reappraisal': erq_reappraisal,
            'ERQ_suppression': erq_suppression,
            'ERQ_total': erq_total
        })

    return pd.DataFrame(behavioral_scores)

def perform_correlation_analysis(behavioral_df, fa_df):
    """
    Perform correlation analysis between behavioral scores and FA values

    Returns:
    --------
    Dictionary with correlation results for each behavioral measure
    """
    # Merge dataframes
    merged_df = behavioral_df.merge(fa_df, on='Subject')

    # Get behavioral score columns
    behavioral_cols = ['ARI_total', 'DERS_total', 'ERQ_reappraisal',
                       'ERQ_suppression', 'ERQ_total']

    # Get FA columns (all columns except Subject and behavioral scores)
    fa_cols = [c for c in merged_df.columns
               if c not in ['Subject'] + behavioral_cols]

    # Store results
    results = {}

    for behavior in behavioral_cols:
        correlations = []

        for roi in fa_cols:
            # Remove NaN values
            mask = merged_df[[behavior, roi]].notna().all(axis=1)
            if mask.sum() >= 3:  # Need at least 3 data points
                x = merged_df.loc[mask, behavior].values
                y = merged_df.loc[mask, roi].values

                # Pearson correlation
                r, p = stats.pearsonr(x, y)

                correlations.append({
                    'ROI': roi,
                    'r': r,
                    'p': p,
                    'n': mask.sum()
                })

        results[behavior] = pd.DataFrame(correlations)

    return results, merged_df

def create_correlation_heatmap(results, output_dir):
    """
    Create heatmap of correlations for each behavioral measure
    """
    behavioral_measures = list(results.keys())

    fig, axes = plt.subplots(len(behavioral_measures), 1,
                            figsize=(20, 5*len(behavioral_measures)))

    if len(behavioral_measures) == 1:
        axes = [axes]

    for idx, (behavior, ax) in enumerate(zip(behavioral_measures, axes)):
        # Get correlation data
        corr_df = results[behavior]

        # Sort by correlation coefficient
        corr_df = corr_df.sort_values('r', ascending=False)

        # Create data for heatmap
        corr_values = corr_df['r'].values.reshape(1, -1)
        roi_names = corr_df['ROI'].values

        # Plot heatmap
        im = ax.imshow(corr_values, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1)

        # Set ticks
        ax.set_yticks([0])
        ax.set_yticklabels([behavior])
        ax.set_xticks(range(len(roi_names)))
        ax.set_xticklabels(roi_names, rotation=90, ha='right', fontsize=6)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.1, aspect=30)
        cbar.set_label('Pearson r', fontsize=10)

        # Add title
        ax.set_title(f'{behavior} - FA Correlations',
                    fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    output_file = output_dir / 'correlation_heatmap_all.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def create_significant_correlations_plot(results, output_dir, p_threshold=0.05):
    """
    Plot only significant correlations
    """
    for behavior, corr_df in results.items():
        # Filter significant correlations
        sig_df = corr_df[corr_df['p'] < p_threshold].copy()

        if len(sig_df) > 0:
            # Sort by absolute correlation
            sig_df['abs_r'] = sig_df['r'].abs()
            sig_df = sig_df.sort_values('abs_r', ascending=True)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(sig_df)*0.3)))

            # Color by positive/negative correlation
            colors = ['red' if r < 0 else 'blue' for r in sig_df['r']]

            # Horizontal bar plot
            bars = ax.barh(range(len(sig_df)), sig_df['r'], color=colors, alpha=0.7)

            # Add ROI names
            ax.set_yticks(range(len(sig_df)))
            ax.set_yticklabels(sig_df['ROI'], fontsize=9)

            # Add p-values as text
            for i, (r, p) in enumerate(zip(sig_df['r'], sig_df['p'])):
                ax.text(r + 0.02*np.sign(r), i, f'p={p:.3f}',
                       va='center', fontsize=7)

            ax.set_xlabel('Pearson Correlation (r)', fontsize=11, fontweight='bold')
            ax.set_title(f'{behavior} - Significant Correlations (p < {p_threshold})',
                        fontsize=13, fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            output_file = output_dir / f'significant_correlations_{behavior}.png'
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")
        else:
            print(f"No significant correlations found for {behavior}")

def create_scatter_plots(results, merged_df, output_dir, top_n=5):
    """
    Create scatter plots for top correlations
    """
    behavioral_cols = ['ARI_total', 'DERS_total', 'ERQ_reappraisal',
                       'ERQ_suppression', 'ERQ_total']

    for behavior in behavioral_cols:
        corr_df = results[behavior]

        # Get top positive and negative correlations
        corr_df['abs_r'] = corr_df['r'].abs()
        top_corr = corr_df.nlargest(top_n, 'abs_r')

        if len(top_corr) > 0:
            # Create subplots
            n_plots = len(top_corr)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols,
                                    figsize=(6*n_cols, 5*n_rows))

            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, (_, row) in enumerate(top_corr.iterrows()):
                if idx < len(axes):
                    ax = axes[idx]
                    roi = row['ROI']
                    r = row['r']
                    p = row['p']

                    # Get data
                    mask = merged_df[[behavior, roi]].notna().all(axis=1)
                    x = merged_df.loc[mask, behavior].values
                    y = merged_df.loc[mask, roi].values

                    # Scatter plot
                    ax.scatter(x, y, alpha=0.6, s=100)

                    # Add regression line
                    z = np.polyfit(x, y, 1)
                    p_fit = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2)

                    # Labels
                    ax.set_xlabel(behavior, fontsize=10, fontweight='bold')
                    ax.set_ylabel(f'{roi} FA', fontsize=10, fontweight='bold')
                    ax.set_title(f'r = {r:.3f}, p = {p:.3f}',
                               fontsize=11, fontweight='bold')
                    ax.grid(alpha=0.3)

            # Hide unused subplots
            for idx in range(len(top_corr), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle(f'{behavior} - Top {top_n} Correlations with ROI FA',
                        fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()

            output_file = output_dir / f'scatter_plots_{behavior}.png'
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")

def create_summary_report(behavioral_df, results, output_dir):
    """
    Create a summary report of the analysis
    """
    output_file = output_dir / 'analysis_summary.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BEHAVIORAL-FA CORRELATION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Behavioral scores summary
        f.write("BEHAVIORAL SCORES SUMMARY\n")
        f.write("-"*80 + "\n")
        behavioral_cols = ['ARI_total', 'DERS_total', 'ERQ_reappraisal',
                          'ERQ_suppression', 'ERQ_total']
        summary_stats = behavioral_df[behavioral_cols].describe()
        f.write(summary_stats.to_string())
        f.write("\n\n")

        # Correlation results
        for behavior, corr_df in results.items():
            f.write("="*80 + "\n")
            f.write(f"{behavior} - CORRELATION RESULTS\n")
            f.write("="*80 + "\n\n")

            # Overall statistics
            f.write(f"Total ROIs analyzed: {len(corr_df)}\n")
            f.write(f"Mean |r|: {corr_df['r'].abs().mean():.3f}\n")
            f.write(f"Max |r|: {corr_df['r'].abs().max():.3f}\n\n")

            # Significant correlations
            sig_05 = corr_df[corr_df['p'] < 0.05]
            sig_01 = corr_df[corr_df['p'] < 0.01]

            f.write(f"Significant correlations (p < 0.05): {len(sig_05)}\n")
            f.write(f"Highly significant correlations (p < 0.01): {len(sig_01)}\n\n")

            if len(sig_05) > 0:
                f.write("TOP 10 SIGNIFICANT CORRELATIONS:\n")
                f.write("-"*80 + "\n")
                top_sig = sig_05.nlargest(10, 'r', keep='all')
                for _, row in top_sig.iterrows():
                    f.write(f"{row['ROI']:50s} r={row['r']:7.3f}  p={row['p']:.4f}  n={row['n']}\n")
                f.write("\n")

            f.write("\n")

    print(f"Saved summary: {output_file}")

def main():
    # Define paths
    results_dir = Path(r"C:\Users\Public\human_brain_lecture\results\251101")
    metadata_dir = results_dir / "metadata"

    print("Loading behavioral data...")
    behavioral_df = load_behavioral_data(metadata_dir)
    print(f"Loaded data for {len(behavioral_df)} subjects")
    print("\nBehavioral scores:")
    print(behavioral_df)

    print("\nLoading FA data...")
    fa_df = pd.read_csv(results_dir / "roi_fa_values.csv")
    print(f"Loaded FA data for {len(fa_df)} subjects, {len(fa_df.columns)-1} ROIs")

    print("\nPerforming correlation analysis...")
    results, merged_df = perform_correlation_analysis(behavioral_df, fa_df)

    print("\nCreating visualizations...")

    # Create correlation heatmap
    create_correlation_heatmap(results, results_dir)

    # Create significant correlations plot
    create_significant_correlations_plot(results, results_dir, p_threshold=0.05)

    # Create scatter plots for top correlations
    create_scatter_plots(results, merged_df, results_dir, top_n=5)

    # Create summary report
    create_summary_report(behavioral_df, results, results_dir)

    # Save detailed correlation results
    for behavior, corr_df in results.items():
        output_file = results_dir / f'correlations_{behavior}.csv'
        corr_df.to_csv(output_file, index=False)
        print(f"Saved detailed results: {output_file}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
