"""
Download specific subjects from OpenNeuro dataset ds000221 using openneuro-py
"""
from openneuro import download

# Dataset and subjects to download
dataset_id = 'ds000221'
target_dir = r'C:\Users\Public\human_brain_lecture\data'

# Download specific subjects
# Include patterns for sub-01 and sub-15
include_patterns = [
    'sub-01/**',
    'sub-15/**',
    'dataset_description.json',
    'participants.tsv',
    'participants.json',
    'README',
    'CHANGES',
    'task-*.json',
]

print(f"Downloading dataset {dataset_id}...")
print(f"Target directory: {target_dir}")
print(f"Including subjects: sub-01, sub-15")
print("="*60)

try:
    download(
        dataset=dataset_id,
        target_dir=target_dir,
        include=include_patterns,
    )
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
except Exception as e:
    print(f"\nError during download: {e}")
    print("\nTrying alternative method...")
    # If include doesn't work, download specific subjects
    for subject in ['01', '15']:
        try:
            download(
                dataset=dataset_id,
                target_dir=target_dir,
                include=[f'sub-{subject}/**'],
            )
            print(f"Successfully downloaded sub-{subject}")
        except Exception as e2:
            print(f"Error downloading sub-{subject}: {e2}")
