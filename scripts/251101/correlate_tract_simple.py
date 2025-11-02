import os, pandas as pd, numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load tract metrics
tract_df = pd.read_csv('roi_tract_metrics_all_subjects.csv')
print(f'Tract metrics: {len(tract_df)} rows')

# Load behavioral data (simplified)
metadata_dir = 'metadata'

# ARI
ari_df = pd.read_csv(os.path.join(metadata_dir, 'ari.tsv'), sep='\t')
ari_df['ARI_total'] = ari_df[[f'ari_{i:02d}' for i in range(1,8)]].sum(axis=1)

# DERS  
ders_df = pd.read_csv(os.path.join(metadata_dir, 'ders.tsv'), sep='\t')
ders_df['DERS_total'] = ders_df[[f'ders_{i:02d}' for i in range(1,6)]].sum(axis=1)

# ERQ
erq_df = pd.read_csv(os.path.join(metadata_dir, 'erq_s.tsv'), sep='\t')
erq_cols = [f'erq_s_{i:02d}' for i in range(1,7)]
erq_df['ERQ_reappraisal'] = erq_df[[erq_cols[i] for i in [0,2,4]]].sum(axis=1)
erq_df['ERQ_suppression'] = erq_df[[erq_cols[i] for i in [1,3,5]]].sum(axis=1)
erq_df['ERQ_total'] = erq_df[erq_cols].sum(axis=1)

# Merge all behavioral
behavioral = pd.merge(ari_df[['participant_id', 'ARI_total']], 
                     ders_df[['participant_id', 'DERS_total']], 
                     on='participant_id')
behavioral = pd.merge(behavioral,
                     erq_df[['participant_id', 'ERQ_reappraisal', 'ERQ_suppression', 'ERQ_total']],
                     on='participant_id')
behavioral = behavioral.rename(columns={'participant_id': 'subject_id'})
print(f'Behavioral: {len(behavioral)} subjects')

# Correlations
results = []
rois = tract_df['roi_name'].unique()
metrics = ['streamline_count', 'mean_fa', 'volume_mm3']
behaviors = ['ARI_total', 'DERS_total', 'ERQ_reappraisal', 'ERQ_suppression', 'ERQ_total']

for roi in rois:
    roi_data = tract_df[tract_df['roi_name'] == roi]
    
    for metric in metrics:
        for behavior in behaviors:
            # Merge
            merged = pd.merge(roi_data[['subject_id', metric]],
                            behavioral[['subject_id', behavior]],
                            on='subject_id')
            
            if len(merged) < 3:
                continue
            
            x = merged[metric].values
            y = merged[behavior].values
            
            # Remove NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                continue
            
            try:
                r, p = stats.pearsonr(x[mask], y[mask])
                results.append({
                    'roi': roi,
                    'metric': metric,
                    'behavior': behavior,
                    'r': r,
                    'p': p,
                    'n': mask.sum()
                })
            except:
                continue

results_df = pd.DataFrame(results)
print(f'Total correlations: {len(results_df)}')

# Save
results_df.to_csv('tract_behavior_correlations_all.csv', index=False)
print('Saved: tract_behavior_correlations_all.csv')

# Significant only
sig = results_df[results_df['p'] < 0.05].copy()
sig['abs_r'] = sig['r'].abs()
sig = sig.sort_values('abs_r', ascending=False)
sig.to_csv('tract_behavior_correlations_significant.csv', index=False)
print(f'Significant (p<0.05): {len(sig)}')
print('\nTop 10:')
print(sig[['roi', 'metric', 'behavior', 'r', 'p']].head(10))
