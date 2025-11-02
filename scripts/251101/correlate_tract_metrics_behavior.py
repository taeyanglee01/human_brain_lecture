#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correlate ROI tract metrics with behavioral data
각 ROI의 백질 특성(streamline count, FA, volume)과 행동 데이터 상관분석
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_behavioral_data(metadata_dir):
    """Load and process behavioral data"""

    subjects = [
        'sub-24053', 'sub-24630', 'sub-24663', 'sub-24683', 'sub-24684',
        'sub-24696', 'sub-24697', 'sub-24699', 'sub-24702', 'sub-60295'
    ]

    behavioral_data = {}

    for subj_id in subjects:
        # ARI
        ari_file = os.path.join(metadata_dir, 'ari.tsv')
        ari_df = pd.read_csv(ari_file, sep='\t')
        ari_row = ari_df[ari_df['participant_id'] == subj_id]
        if len(ari_row) > 0:
            ari_cols = [f'ari_{i:02d}' for i in range(1, 8)]
            ari_values = ari_row[ari_cols].values[0]
            ari_total = np.nansum(ari_values)
        else:
            ari_total = np.nan

        # DERS
        ders_file = os.path.join(metadata_dir, 'ders.tsv')
        ders_df = pd.read_csv(ders_file, sep='\t')
        ders_row = ders_df[ders_df['participant_id'] == subj_id]
        if len(ders_row) > 0:
            ders_cols = [f'ders_{i:02d}' for i in range(1, 6)]
            ders_values = ders_row[ders_cols].values[0]
            ders_total = np.nansum(ders_values)
        else:
            ders_total = np.nan

        # ERQ
        erq_file = os.path.join(metadata_dir, 'erq_s.tsv')
        erq_df = pd.read_csv(erq_file, sep='\t')
        erq_row = erq_df[erq_df['participant_id'] == subj_id]
        if len(erq_row) > 0:
            erq_cols = [f'erq_s_{i:02d}' for i in range(1, 7)]
            erq_values = erq_row[erq_cols].values[0]

            # Reappraisal: items 1, 3, 5
            erq_reappraisal = np.nansum(erq_values[[0, 2, 4]])

            # Suppression: items 2, 4, 6
            erq_suppression = np.nansum(erq_values[[1, 3, 5]])

            # Total
            erq_total = np.nansum(erq_values)
        else:
            erq_reappraisal = np.nan
            erq_suppression = np.nan
            erq_total = np.nan

        behavioral_data[subj_id] = {
            'ARI_total': ari_total,
            'DERS_total': ders_total,
            'ERQ_reappraisal': erq_reappraisal,
            'ERQ_suppression': erq_suppression,
            'ERQ_total': erq_total
        }

    return pd.DataFrame(behavioral_data).T

def perform_correlation_analysis(tract_df, behavioral_df):
    """Perform correlation analysis between tract metrics and behavioral scores"""

    # Get unique ROIs
    rois = tract_df['roi_name'].unique()

    # Metrics to analyze
    tract_metrics = ['streamline_count', 'mean_fa', 'volume_mm3']
    behavioral_measures = ['ARI_total', 'DERS_total', 'ERQ_reappraisal', 'ERQ_suppression', 'ERQ_total']

    results = []

    for roi in rois:
        roi_data = tract_df[tract_df['roi_name'] == roi].set_index('subject_id')

        for tract_metric in tract_metrics:
            tract_values = roi_data[tract_metric]

            for behavior in behavioral_measures:
                behavior_values = behavioral_df[behavior]

                # Align subjects
                common_subjects = tract_values.index.intersection(behavior_values.index)
                if len(common_subjects) < 3:
                    continue

                x = tract_values.loc[common_subjects]
                y = behavior_values.loc[common_subjects]

                # Convert to numpy arrays with same order
                x = x.reindex(common_subjects).values
                y = y.reindex(common_subjects).values

                # Remove NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(mask) < 3:
                    continue

                x_clean = x[mask]
                y_clean = y[mask]

                # Pearson correlation
                try:
                    r, p = stats.pearsonr(x_clean, y_clean)

                    results.append({
                        'roi': roi,
                        'tract_metric': tract_metric,
                        'behavioral_measure': behavior,
                        'r': r,
                        'p': p,
                        'n': len(x_clean),
                        'abs_r': abs(r)
                    })
                except:
                    continue

    return pd.DataFrame(results)

def create_visualizations(correlation_df, output_dir):
    """Create visualization of correlation results"""

    # For each behavioral measure and tract metric, create heatmap
    behavioral_measures = correlation_df['behavioral_measure'].unique()
    tract_metrics = correlation_df['tract_metric'].unique()

    for behavior in behavioral_measures:
        for metric in tract_metrics:
            subset = correlation_df[
                (correlation_df['behavioral_measure'] == behavior) &
                (correlation_df['tract_metric'] == metric)
            ]

            if len(subset) == 0:
                continue

            # Get significant correlations (p < 0.05)
            sig_subset = subset[subset['p'] < 0.05].sort_values('abs_r', ascending=False)

            if len(sig_subset) == 0:
                continue

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Top positive correlations
            top_pos = sig_subset[sig_subset['r'] > 0].head(10)
            if len(top_pos) > 0:
                ax = axes[0]
                bars = ax.barh(range(len(top_pos)), top_pos['r'].values,
                              color='darkgreen', alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(top_pos)))
                ax.set_yticklabels([roi[:30] for roi in top_pos['roi'].values],
                                  fontsize=9)
                ax.set_xlabel('Correlation (r)', fontsize=11, fontweight='bold')
                ax.set_title(f'Top Positive Correlations\n{behavior} ~ {metric}',
                           fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                # Add p-values
                for i, (bar, p_val) in enumerate(zip(bars, top_pos['p'].values)):
                    sig = '**' if p_val < 0.01 else '*'
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{sig}', va='center', fontsize=10, fontweight='bold')

            # Top negative correlations
            top_neg = sig_subset[sig_subset['r'] < 0].head(10)
            if len(top_neg) > 0:
                ax = axes[1]
                bars = ax.barh(range(len(top_neg)), top_neg['r'].values,
                              color='darkred', alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(top_neg)))
                ax.set_yticklabels([roi[:30] for roi in top_neg['roi'].values],
                                  fontsize=9)
                ax.set_xlabel('Correlation (r)', fontsize=11, fontweight='bold')
                ax.set_title(f'Top Negative Correlations\n{behavior} ~ {metric}',
                           fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                # Add p-values
                for i, (bar, p_val) in enumerate(zip(bars, top_neg['p'].values)):
                    sig = '**' if p_val < 0.01 else '*'
                    ax.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/2,
                           f'{sig}', va='center', ha='right', fontsize=10, fontweight='bold')

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'correlation_{behavior}_{metric}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved: {output_file}")

def create_summary_report(correlation_df, output_dir):
    """Create summary report of significant correlations"""

    # Get significant correlations
    sig_df = correlation_df[correlation_df['p'] < 0.05].copy()
    sig_df = sig_df.sort_values('abs_r', ascending=False)

    # Save to CSV
    output_file = os.path.join(output_dir, 'tract_behavior_correlations_significant.csv')
    sig_df.to_csv(output_file, index=False)
    print(f"\nSaved significant correlations: {output_file}")
    print(f"Total significant correlations: {len(sig_df)}")

    # Create summary by behavioral measure
    print("\n" + "="*80)
    print("SIGNIFICANT CORRELATIONS SUMMARY")
    print("="*80)

    for behavior in sig_df['behavioral_measure'].unique():
        behavior_subset = sig_df[sig_df['behavioral_measure'] == behavior]
        print(f"\n{behavior}: {len(behavior_subset)} significant correlations")

        for metric in behavior_subset['tract_metric'].unique():
            metric_subset = behavior_subset[behavior_subset['tract_metric'] == metric]
            print(f"  {metric}: {len(metric_subset)} correlations")

            # Show top 3
            top3 = metric_subset.head(3)
            for idx, row in top3.iterrows():
                sig = '**' if row['p'] < 0.01 else '*'
                print(f"    {row['roi'][:40]:40s}  r={row['r']:6.3f} {sig}  p={row['p']:.4f}")

    return sig_df

def main():
    """Main analysis function"""

    results_dir = 'C:/Users/Public/human_brain_lecture/results/251101'
    metadata_dir = os.path.join(results_dir, 'metadata')

    print("Loading data...")

    # Load tract metrics
    tract_file = os.path.join(results_dir, 'roi_tract_metrics_all_subjects.csv')
    tract_df = pd.read_csv(tract_file)
    print(f"  Tract metrics: {len(tract_df)} rows, {tract_df['subject_id'].nunique()} subjects")

    # Load behavioral data
    behavioral_df = load_behavioral_data(metadata_dir)
    print(f"  Behavioral data: {len(behavioral_df)} subjects")
    print(f"  Measures: {list(behavioral_df.columns)}")

    # Perform correlation analysis
    print("\nPerforming correlation analysis...")
    correlation_df = perform_correlation_analysis(tract_df, behavioral_df)
    print(f"  Total correlations computed: {len(correlation_df)}")

    # Save all correlations
    all_corr_file = os.path.join(results_dir, 'tract_behavior_correlations_all.csv')
    correlation_df.to_csv(all_corr_file, index=False)
    print(f"  Saved all correlations: {all_corr_file}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(correlation_df, results_dir)

    # Create summary report
    sig_df = create_summary_report(correlation_df, results_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
