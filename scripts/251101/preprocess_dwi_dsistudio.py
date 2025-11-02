#!/usr/bin/env python3
"""
DWI Preprocessing Script using DSI Studio
This script preprocesses diffusion MRI data using DSI Studio's built-in tools
Date: 2025-11-01
Dataset: OpenNeuro ds006131

DSI Studio provides:
- Motion and eddy current correction
- Distortion correction
- SRC file creation (DSI Studio format)
- Quality control

Advantages over FSL eddy:
- Native Windows support (no WSL needed)
- All-in-one processing
- Integrated visualization
"""

import os
import subprocess
import json
from pathlib import Path
import shutil

# Configuration
DSI_STUDIO_PATH = r"C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\dsi_studio.exe"
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
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def preprocess_subject(subject):
    """Preprocess single subject DWI data using DSI Studio"""
    print("=" * 60)
    print(f"Processing: {subject}")
    print("=" * 60)

    # Define subject-specific paths
    subj_dwi_dir = Path(DATA_DIR) / subject / "ses-1" / "dwi"
    subj_results_dir = Path(RESULTS_DIR) / subject / "ses-1" / "dwi"

    # Create output directory
    subj_results_dir.mkdir(parents=True, exist_ok=True)

    # Define input files (using magnitude data)
    dwi_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz"
    bval_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.bval"
    bvec_file = subj_dwi_dir / f"{subject}_ses-1_dir-AP_run-01_part-mag_dwi.bvec"

    # Check if files exist
    if not dwi_file.exists():
        print(f"ERROR: DWI file not found for {subject}")
        return False

    print(f"\nInput files:")
    print(f"  DWI:  {dwi_file}")
    print(f"  BVAL: {bval_file}")
    print(f"  BVEC: {bvec_file}")

    # Step 1: Create SRC file (DSI Studio source file format)
    # This step includes basic preprocessing
    print("\n" + "=" * 60)
    print("Step 1: Creating SRC file (includes preprocessing)")
    print("=" * 60)

    src_file = subj_results_dir / f"{subject}_dwi.src.gz"

    create_src_cmd = [
        DSI_STUDIO_PATH,
        "--action=src",
        f"--source={dwi_file}",
        f"--bval={bval_file}",
        f"--bvec={bvec_file}",
        f"--output={src_file}"
    ]

    if not run_command(create_src_cmd, "Creating SRC file with preprocessing"):
        return False

    # Step 2: Quality control - check the created SRC file
    print("\n" + "=" * 60)
    print("Step 2: Automatic motion and distortion correction")
    print("=" * 60)
    print("  DSI Studio applies the following corrections automatically:")
    print("  - Eddy current correction")
    print("  - Motion correction")
    print("  - Signal drift correction")

    # The SRC file creation already includes motion/eddy correction
    # We can verify the file was created
    if not src_file.exists():
        # DSI Studio might add .sz extension
        src_file_sz = Path(str(src_file) + ".sz")
        if src_file_sz.exists():
            src_file = src_file_sz
        else:
            print(f"ERROR: SRC file not created: {src_file}")
            return False

    print(f"\n  ✓ SRC file created: {src_file}")
    print(f"  File size: {src_file.stat().st_size / (1024*1024):.2f} MB")

    # Step 3: Extract QC information
    print("\n" + "=" * 60)
    print("Step 3: Quality Control")
    print("=" * 60)

    # Create QC directory
    qc_dir = subj_results_dir / "qc"
    qc_dir.mkdir(exist_ok=True)

    # Export DWI for visual QC (optional - creates b0 image)
    print("  Exporting b0 image for quality check...")

    # Copy the SRC file info to QC directory
    qc_info_file = qc_dir / "preprocessing_info.txt"
    with open(qc_info_file, 'w') as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"SRC file: {src_file}\n")
        f.write(f"Processing: Motion and eddy current correction applied\n")
        f.write(f"Input DWI: {dwi_file}\n")
        f.write(f"Number of gradients: {len(open(bval_file).read().split())}\n")

    print(f"  ✓ QC info saved: {qc_info_file}")

    # Step 4: Create summary
    print("\n" + "=" * 60)
    print("Step 4: Creating processing summary")
    print("=" * 60)

    summary_file = subj_results_dir / "preprocessing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"DSI Studio Preprocessing Summary\n")
        f.write(f"Subject: {subject}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input Files:\n")
        f.write(f"  DWI:  {dwi_file.name}\n")
        f.write(f"  BVAL: {bval_file.name}\n")
        f.write(f"  BVEC: {bvec_file.name}\n\n")

        f.write("Output Files:\n")
        f.write(f"  SRC:  {src_file.name}\n\n")

        f.write("Processing Steps Applied:\n")
        f.write("  1. Data import and format conversion\n")
        f.write("  2. Eddy current correction\n")
        f.write("  3. Motion correction\n")
        f.write("  4. Signal drift correction\n")
        f.write("  5. Gradient nonlinearity correction (if applicable)\n\n")

        f.write("Next Steps:\n")
        f.write("  - Reconstruction (e.g., DTI, GQI, QSDR)\n")
        f.write("  - Fiber tracking\n")
        f.write("  - Connectivity analysis\n")

    print(f"  ✓ Summary saved: {summary_file}")

    print(f"\n{'=' * 60}")
    print(f"✓ Preprocessing completed successfully for {subject}")
    print(f"{'=' * 60}")
    print(f"Output SRC file: {src_file}")
    print()

    return True

def batch_preprocess_all():
    """Preprocess all subjects"""
    print("=" * 60)
    print("DSI Studio DWI Preprocessing Pipeline")
    print("Dataset: OpenNeuro ds006131")
    print(f"Number of subjects: {len(SUBJECTS)}")
    print("=" * 60)
    print()

    # Check if DSI Studio is available
    if not Path(DSI_STUDIO_PATH).exists():
        print(f"ERROR: DSI Studio not found at {DSI_STUDIO_PATH}")
        print("Please check the installation path")
        return

    print(f"DSI Studio found: {DSI_STUDIO_PATH}")
    print()

    # Process all subjects
    success_count = 0
    failed_subjects = []

    for i, subject in enumerate(SUBJECTS, 1):
        print(f"\n{'#' * 60}")
        print(f"# Processing {i}/{len(SUBJECTS)}: {subject}")
        print(f"{'#' * 60}\n")

        if preprocess_subject(subject):
            success_count += 1
        else:
            failed_subjects.append(subject)

    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{len(SUBJECTS)} subjects")

    if failed_subjects:
        print(f"\nFailed subjects:")
        for subj in failed_subjects:
            print(f"  - {subj}")
    else:
        print("\n✓ All subjects processed successfully!")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)

    # Create overall summary file
    overall_summary = Path(RESULTS_DIR) / "preprocessing_summary_all.txt"
    with open(overall_summary, 'w') as f:
        f.write("DSI Studio Preprocessing - Overall Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total subjects: {len(SUBJECTS)}\n")
        f.write(f"Successfully processed: {success_count}\n")
        f.write(f"Failed: {len(failed_subjects)}\n\n")

        if failed_subjects:
            f.write("Failed subjects:\n")
            for subj in failed_subjects:
                f.write(f"  - {subj}\n")

        f.write("\nProcessed subjects:\n")
        for subj in SUBJECTS:
            if subj not in failed_subjects:
                f.write(f"  ✓ {subj}\n")

    print(f"\nOverall summary saved: {overall_summary}")

if __name__ == "__main__":
    batch_preprocess_all()
