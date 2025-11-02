import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load significant correlations
sig = pd.read_csv('tract_behavior_correlations_significant.csv')
print(f'Significant correlations: {len(sig)}')

# Create visualizations for each behavioral measure
behaviors = sig['behavior'].unique()

for behavior in behaviors:
    behavior_sig = sig[sig['behavior'] == behavior].sort_values('abs_r', ascending=False)
    
    if len(behavior_sig) == 0:
        continue
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Significant Correlations with {behavior}', fontsize=16, fontweight='bold')
    
    # By metric
    for idx, metric in enumerate(['streamline_count', 'mean_fa', 'volume_mm3']):
        metric_data = behavior_sig[behavior_sig['metric'] == metric]
        
        if len(metric_data) == 0:
            axes[idx].text(0.5, 0.5, 'No significant\ncorrelations',
                         ha='center', va='center', fontsize=14)
            axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            continue
        
        # Split positive and negative
        pos = metric_data[metric_data['r'] > 0]
        neg = metric_data[metric_data['r'] < 0]
        
        ax = axes[idx]
        y_pos = 0
        colors = []
        labels = []
        values = []
        
        # Plot positive correlations
        for _, row in pos.iterrows():
            values.append(row['r'])
            labels.append(row['roi'][:25])
            colors.append('darkgreen' if row['p'] < 0.01 else 'green')
            y_pos += 1
        
        # Plot negative correlations
        for _, row in neg.iterrows():
            values.append(row['r'])
            labels.append(row['roi'][:25])
            colors.append('darkred' if row['p'] < 0.01 else 'red')
            y_pos += 1
        
        if len(values) > 0:
            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Correlation (r)', fontsize=10, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.axvline(0, color='black', linewidth=1)
            ax.grid(axis='x', alpha=0.3)
            
            # Add significance markers
            for i, (bar, row_data) in enumerate(zip(bars, list(pos.iterrows()) + list(neg.iterrows()))):
                _, row = row_data
                sig_marker = '**' if row['p'] < 0.01 else '*'
                x_pos = bar.get_width()
                if x_pos > 0:
                    ax.text(x_pos + 0.02, bar.get_y() + bar.get_height()/2,
                           sig_marker, va='center', fontsize=10, fontweight='bold')
                else:
                    ax.text(x_pos - 0.02, bar.get_y() + bar.get_height()/2,
                           sig_marker, va='center', ha='right', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No significant\ncorrelations',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'correlations_{behavior}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: correlations_{behavior}.png')

# Summary plot - all significant correlations
fig, ax = plt.subplots(figsize=(14, 10))

# Get top 30 by abs(r)
top30 = sig.head(30).copy()
top30['label'] = top30['roi'].str[:20] + ' - ' + top30['metric'].str[:10] + ' ~ ' + top30['behavior']

colors = ['darkgreen' if (r > 0 and p < 0.01) else 
          'green' if r > 0 else
          'darkred' if p < 0.01 else 'red' 
          for r, p in zip(top30['r'], top30['p'])]

bars = ax.barh(range(len(top30)), top30['r'].values, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['label'].values, fontsize=8)
ax.set_xlabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Significant Correlations\nTract Metrics vs Behavioral Measures',
            fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=1.5)
ax.grid(axis='x', alpha=0.3)

# Add significance markers
for i, (bar, p_val, r_val) in enumerate(zip(bars, top30['p'], top30['r'])):
    sig_marker = '**' if p_val < 0.01 else '*'
    x_pos = bar.get_width()
    if x_pos > 0:
        ax.text(x_pos + 0.01, bar.get_y() + bar.get_height()/2,
               sig_marker, va='center', fontsize=9, fontweight='bold')
    else:
        ax.text(x_pos - 0.01, bar.get_y() + bar.get_height()/2,
               sig_marker, va='center', ha='right', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('correlations_summary_top30.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: correlations_summary_top30.png')

print('\nVisualization complete!')
