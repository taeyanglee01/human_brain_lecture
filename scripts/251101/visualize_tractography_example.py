import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load TDI for sub-24053
tdi_file = 'sub-24053/ses-1/dwi/sub-24053_whole_brain.tt.gz.tdi.nii.gz'
tdi_img = nib.load(tdi_file)
tdi_data = tdi_img.get_fdata()

print(f'TDI shape: {tdi_data.shape}')
print(f'TDI range: {tdi_data.min():.0f} - {tdi_data.max():.0f}')
print(f'Non-zero voxels: {np.count_nonzero(tdi_data)}')

# Create figure with multiple views
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Whole Brain Fiber Tractography Visualization\nSub-24053 Tract Density Image (TDI)', 
             fontsize=18, fontweight='bold', y=0.98)

# Custom colormap (black -> blue -> cyan -> yellow -> red)
colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('tract_density', colors, N=n_bins)

# Get middle slices
x_mid = tdi_data.shape[0] // 2
y_mid = tdi_data.shape[1] // 2
z_mid = tdi_data.shape[2] // 2

# Sagittal views (left and right hemispheres)
ax1 = plt.subplot(3, 3, 1)
sagittal_left = tdi_data[x_mid - 15, :, :]
im1 = ax1.imshow(np.rot90(sagittal_left), cmap=cmap, aspect='auto', 
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax1.set_title('Sagittal (Left Hemisphere)', fontsize=12, fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Tract Density')

ax2 = plt.subplot(3, 3, 2)
sagittal_mid = tdi_data[x_mid, :, :]
im2 = ax2.imshow(np.rot90(sagittal_mid), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax2.set_title('Sagittal (Midline)', fontsize=12, fontweight='bold')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Tract Density')

ax3 = plt.subplot(3, 3, 3)
sagittal_right = tdi_data[x_mid + 15, :, :]
im3 = ax3.imshow(np.rot90(sagittal_right), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax3.set_title('Sagittal (Right Hemisphere)', fontsize=12, fontweight='bold')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Tract Density')

# Coronal views
ax4 = plt.subplot(3, 3, 4)
coronal_front = tdi_data[:, y_mid - 20, :]
im4 = ax4.imshow(np.rot90(coronal_front), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax4.set_title('Coronal (Anterior)', fontsize=12, fontweight='bold')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Tract Density')

ax5 = plt.subplot(3, 3, 5)
coronal_mid = tdi_data[:, y_mid, :]
im5 = ax5.imshow(np.rot90(coronal_mid), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax5.set_title('Coronal (Central)', fontsize=12, fontweight='bold')
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Tract Density')

ax6 = plt.subplot(3, 3, 6)
coronal_back = tdi_data[:, y_mid + 20, :]
im6 = ax6.imshow(np.rot90(coronal_back), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax6.set_title('Coronal (Posterior)', fontsize=12, fontweight='bold')
ax6.axis('off')
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Tract Density')

# Axial views
ax7 = plt.subplot(3, 3, 7)
axial_bottom = tdi_data[:, :, z_mid - 15]
im7 = ax7.imshow(np.rot90(axial_bottom), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax7.set_title('Axial (Inferior)', fontsize=12, fontweight='bold')
ax7.axis('off')
plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='Tract Density')

ax8 = plt.subplot(3, 3, 8)
axial_mid = tdi_data[:, :, z_mid]
im8 = ax8.imshow(np.rot90(axial_mid), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax8.set_title('Axial (Central)', fontsize=12, fontweight='bold')
ax8.axis('off')
plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, label='Tract Density')

ax9 = plt.subplot(3, 3, 9)
axial_top = tdi_data[:, :, z_mid + 15]
im9 = ax9.imshow(np.rot90(axial_top), cmap=cmap, aspect='auto',
                 vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax9.set_title('Axial (Superior)', fontsize=12, fontweight='bold')
ax9.axis('off')
plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04, label='Tract Density')

# Add information box
info_text = f"""
Tractography Information:
━━━━━━━━━━━━━━━━━━━━━━━
Subject: sub-24053
Total Tracts: 462,414 streamlines
Method: Deterministic (Streamline)
FA Threshold: 0.2
Turning Angle: 45°
Min Length: 20 mm

TDI Resolution: {tdi_data.shape[0]}×{tdi_data.shape[1]}×{tdi_data.shape[2]}
Voxel Size: ~1.69×1.69×1.70 mm³
Max Density: {tdi_data.max():.0f} streamlines/voxel
"""

fig.text(0.02, 0.02, info_text, fontsize=10, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='bottom')

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
output_file = 'sub-24053_tractography_TDI_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f'\nSaved: {output_file}')
print('Visualization complete!')

# Create a simpler single-view version for report
fig2, ax = plt.subplots(figsize=(12, 10))
axial_mid = tdi_data[:, :, z_mid]
im = ax.imshow(np.rot90(axial_mid), cmap=cmap, aspect='auto',
               vmin=0, vmax=np.percentile(tdi_data[tdi_data>0], 98))
ax.set_title(f'Whole Brain Tractography - Tract Density Image (Axial View)\nSub-24053: {462414:,} Streamlines', 
            fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label('Number of Streamlines per Voxel', fontsize=12, fontweight='bold')

# Add scale bar
from matplotlib.patches import Rectangle
voxel_size_mm = 1.69
scale_length_mm = 50  # 50mm scale bar
scale_length_voxels = scale_length_mm / voxel_size_mm
x_start = axial_mid.shape[1] * 0.75
y_start = axial_mid.shape[0] * 0.92

rect = Rectangle((x_start, y_start), scale_length_voxels, 3, 
                 facecolor='white', edgecolor='black', linewidth=2,
                 transform=ax.transData)
ax.add_patch(rect)
ax.text(x_start + scale_length_voxels/2, y_start - 5, '50 mm',
        ha='center', va='top', fontsize=10, fontweight='bold',
        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
output_simple = 'sub-24053_tractography_simple.png'
plt.savefig(output_simple, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f'Saved: {output_simple}')
