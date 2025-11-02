#!/usr/bin/env python3
"""
DWI Preprocessing Script using FSL eddy
This script preprocesses diffusion MRI data with eddy current and motion correction
Date: 2025-11-01
Dataset: OpenNeuro ds006131

Requirements:
- FSL (FMRIB Software Library) installed
- Python 3.x
- nibabel library (pip install nibabel)
"""

import os
import subprocess
import json
import glob
from pathlib import Path

# Configuration
DATA_DIR = r"C:\Users\Public\human_brain_lecture\data\251101"
RESULTS_DIR = r"C:\Users\Public\human_brain_lecture\results\251101"

# List of subjects (excluding PILOT)
SUBJECTS = [
    "sub-24053",
    "sub-24630",
    "sub-24663",
    "sub-24683",
    "sub-24684",
    "sub-24696",
    "sub-24697",
    "sub-24699",
    "sub-24702",
    "sub-60295"
]

def run_command(cmd, description=""):
    """Execute shell command and handle errors"""
    if description:
        print(f"  {description}")

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDERR: {e.stderr}")
        return False

def get_total_readout_time(json_file):
    """Extract TotalReadoutTime from JSON metadata"""
    try:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            return metadata.get('TotalReadoutTime', 0.0526491)
    except Exception as e:
        print(f"Warning: Could not read JSON file {json_file}, using default value")
        return 0.0526491

def preprocess_subject(subject):
    """Preprocess single subject DWI data"""
    print("=" * 50)
    print(f"Processing: {subject}")
    print("=" * 50)

    # Define subject-specific paths
    subj_dwi_dir = Path(DATA_DIR) / subject / "ses-1" / "dwi"
    subj_results_dir = Path(RESULTS_DIR) / subject / "ses-1" / "dwi"

    # Create output directory
    subj_results_dir.mkdir(parents=True, exist_ok=True)

    # Define input files (using magnitude data)
    dwi_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz"
    bval_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.bval"
    bvec_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.bvec"
    json_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.json"

    # Check if files exist
    if not dwi_file.exists():
        print(f"ERROR: DWI file not found for {subject}")
        return False

    print("\nStep 1: Creating brain mask from b0 image...")
    # Extract first b0 volume
    b0_vol = subj_results_dir / "b0_vol"
    if not run_command(
        ["fslroi", str(dwi_file), str(b0_vol), "0", "1"],
        "Extracting b0 volume"
    ):
        return False

    # Create brain mask using BET
    b0_brain = subj_results_dir / "b0_brain"
    if not run_command(
        ["bet", str(b0_vol), str(b0_brain), "-m", "-f", "0.3"],
        "Creating brain mask with BET"
    ):
        return False

    print("\nStep 2: Creating index file for eddy...")
    # Get number of volumes
    try:
        result = subprocess.run(
            ["fslnvols", str(dwi_file)],
            capture_output=True,
            text=True,
            check=True
        )
        nvols = int(result.stdout.strip())
        print(f"  Number of volumes: {nvols}")
    except Exception as e:
        print(f"ERROR: Could not determine number of volumes: {e}")
        return False

    # Create index file (all volumes use the same acquisition parameters)
    index_file = subj_results_dir / "index.txt"
    with open(index_file, 'w') as f:
        f.write(' '.join(['1'] * nvols))
    print(f"  Created index file: {index_file}")

    print("\nStep 3: Creating acquisition parameters file...")
    # Get TotalReadoutTime from JSON
    total_readout_time = get_total_readout_time(json_file)

    # acqparams.txt format: phase-encoding direction and readout time
    # PhaseEncodingDirection is "j-" (A>>P)
    acqparams_file = subj_results_dir / "acqparams.txt"
    with open(acqparams_file, 'w') as f:
        f.write(f"0 -1 0 {total_readout_time}\n")
    print(f"  Created acqparams file with TotalReadoutTime={total_readout_time}")

    print("\nStep 4: Running eddy correction...")
    print("  This may take 10-30 minutes per subject...")

    # Prepare eddy command
    dwi_eddy_prefix = subj_results_dir / "dwi_eddy"
    b0_brain_mask = subj_results_dir / "b0_brain_mask"

    eddy_cmd = [
        "eddy_openmp",  # or "eddy_cuda" if GPU available
        f"--imain={dwi_file}",
        f"--mask={b0_brain_mask}",
        f"--acqp={acqparams_file}",
        f"--index={index_file}",
        f"--bvecs={bvec_file}",
        f"--bvals={bval_file}",
        f"--out={dwi_eddy_prefix}",
        "--repol",  # Replace outliers
        "--data_is_shelled",  # Data has distinct shells
        "--verbose"
    ]

    if not run_command(eddy_cmd, "Running eddy correction"):
        return False

    print("\nStep 5: Quality control - generating reports...")
    # Create QC directory
    qc_dir = subj_results_dir / "qc"
    qc_dir.mkdir(exist_ok=True)

    # Generate eddy QC reports if eddy_quad is available
    eddy_quad_cmd = [
        "eddy_quad",
        str(dwi_eddy_prefix),
        "-idx", str(index_file),
        "-par", str(acqparams_file),
        "-m", str(b0_brain_mask),
        "-b", str(bval_file),
        "-o", str(qc_dir)
    ]

    # Run eddy_quad (don't fail if not available)
    try:
        subprocess.run(eddy_quad_cmd, check=True, capture_output=True)
        print(f"  QC report generated: {qc_dir}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Note: eddy_quad not available, skipping QC report generation")

    # Copy corrected bvals and rotated bvecs
    import shutil
    shutil.copy(bval_file, subj_results_dir / "dwi_eddy.bval")

    rotated_bvecs = Path(str(dwi_eddy_prefix) + ".eddy_rotated_bvecs")
    if rotated_bvecs.exists():
        shutil.copy(rotated_bvecs, subj_results_dir / "dwi_eddy.bvec")

    print(f"\n✓ Preprocessing completed for {subject}")
    print(f"  Output: {dwi_eddy_prefix}.nii.gz")
    print()

    return True

def main():
    """Main processing pipeline"""
    print("=" * 50)
    print("DWI Preprocessing Pipeline")
    print("Dataset: OpenNeuro ds006131")
    print(f"Number of subjects: {len(SUBJECTS)}")
    print("=" * 50)
    print()

    # Check if FSL is available
    try:
        subprocess.run(["fslinfo"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: FSL not found. Please install FSL and add it to PATH")
        print("Visit: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation")
        return

    # Process all subjects
    success_count = 0
    failed_subjects = []

    for subject in SUBJECTS:
        if preprocess_subject(subject):
            success_count += 1
        else:
            failed_subjects.append(subject)

    # Summary
    print("=" * 50)
    print("Preprocessing Summary")
    print("=" * 50)
    print(f"Successfully processed: {success_count}/{len(SUBJECTS)} subjects")

    if failed_subjects:
        print(f"Failed subjects: {', '.join(failed_subjects)}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
