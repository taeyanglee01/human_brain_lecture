"""
Download specific subjects from OpenNeuro dataset ds000221
"""
import urllib.request
import os
import json

def download_file(url, filepath):
    """Download file from URL to filepath"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  -> Saved to {filepath}")
        return True
    except Exception as e:
        print(f"  -> Error: {e}")
        return False

def download_subject(dataset_id, subject_id, output_dir):
    """Download a subject from OpenNeuro"""

    # OpenNeuro S3 base URL
    base_url = f"https://s3.amazonaws.com/openneuro.org/{dataset_id}"

    # Common file patterns for this dataset
    files_to_download = [
        f"sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz",
        f"sub-{subject_id}/func/sub-{subject_id}_task-rest_bold.nii.gz",
        f"sub-{subject_id}/dwi/sub-{subject_id}_dwi.nii.gz",
        f"sub-{subject_id}/dwi/sub-{subject_id}_dwi.bval",
        f"sub-{subject_id}/dwi/sub-{subject_id}_dwi.bvec",
    ]

    # Dataset files
    dataset_files = [
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
    ]

    # Download dataset description files (only once)
    for file in dataset_files:
        url = f"{base_url}/{file}"
        filepath = os.path.join(output_dir, file)
        if not os.path.exists(filepath):
            download_file(url, filepath)

    # Download subject files
    for file in files_to_download:
        url = f"{base_url}/{file}"
        filepath = os.path.join(output_dir, file)
        download_file(url, filepath)

if __name__ == "__main__":
    dataset_id = "ds000221"
    output_dir = r"C:\Users\Public\human_brain_lecture\data"

    # Download subjects
    subjects = ["01", "15"]

    for subject_id in subjects:
        print(f"\n{'='*60}")
        print(f"Downloading subject {subject_id}")
        print(f"{'='*60}")
        download_subject(dataset_id, subject_id, output_dir)

    print("\n\nDownload complete!")
