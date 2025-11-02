#!/bin/bash

# DWI Preprocessing Script using FSL eddy
# This script preprocesses diffusion MRI data with eddy current and motion correction
# Date: 2025-11-01
# Dataset: OpenNeuro ds006131

# Set paths
DATA_DIR="C:/Users/Public/human_brain_lecture/data/251101"
RESULTS_DIR="C:/Users/Public/human_brain_lecture/results/251101"

# List of subjects (excluding PILOT)
SUBJECTS=(
    "sub-24053"
    "sub-24630"
    "sub-24663"
    "sub-24683"
    "sub-24684"
    "sub-24696"
    "sub-24697"
    "sub-24699"
    "sub-24702"
    "sub-60295"
)

# FSL configuration
export FSLDIR=/usr/local/fsl
export PATH=${FSLDIR}/bin:${PATH}
. ${FSLDIR}/etc/fslconf/fsl.sh

# Function to preprocess single subject
preprocess_subject() {
    local SUBJECT=$1
    echo "=========================================="
    echo "Processing: ${SUBJECT}"
    echo "=========================================="

    # Define subject-specific paths
    SUBJ_DWI_DIR="${DATA_DIR}/${SUBJECT}/ses-1/dwi"
    SUBJ_RESULTS_DIR="${RESULTS_DIR}/${SUBJECT}/ses-1/dwi"

    # Create output directory
    mkdir -p "${SUBJ_RESULTS_DIR}"

    # Define input files (using magnitude data)
    DWI_FILE="${SUBJ_DWI_DIR}/${SUBJECT}_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz"
    BVAL_FILE="${SUBJ_DWI_DIR}/${SUBJECT}_ses-1_dir-AP_run-01_part-mag_dwi.bval"
    BVEC_FILE="${SUBJ_DWI_DIR}/${SUBJECT}_ses-1_dir-AP_run-01_part-mag_dwi.bvec"

    # Check if files exist
    if [[ ! -f "${DWI_FILE}" ]]; then
        echo "ERROR: DWI file not found for ${SUBJECT}"
        return 1
    fi

    echo "Step 1: Creating brain mask from b0 image..."
    # Extract b0 volumes (b-value < 50)
    fslroi "${DWI_FILE}" "${SUBJ_RESULTS_DIR}/b0_vol" 0 1

    # Create brain mask using BET
    bet "${SUBJ_RESULTS_DIR}/b0_vol" "${SUBJ_RESULTS_DIR}/b0_brain" -m -f 0.3

    echo "Step 2: Creating index file for eddy..."
    # Create index file (all volumes use the same acquisition parameters)
    NVOLS=$(fslnvols "${DWI_FILE}")
    INDEX_FILE="${SUBJ_RESULTS_DIR}/index.txt"
    indx=""
    for ((i=1; i<=NVOLS; i++)); do
        indx="$indx 1"
    done
    echo $indx > "${INDEX_FILE}"

    echo "Step 3: Creating acquisition parameters file..."
    # acqparams.txt format: phase-encoding direction and readout time
    # PhaseEncodingDirection is "j-" (A>>P), TotalReadoutTime from JSON
    ACQPARAMS_FILE="${SUBJ_RESULTS_DIR}/acqparams.txt"
    echo "0 -1 0 0.0526491" > "${ACQPARAMS_FILE}"

    echo "Step 4: Running eddy correction..."
    # Run eddy for motion and eddy current correction
    # --imain: input DWI data
    # --mask: brain mask
    # --acqp: acquisition parameters
    # --index: volume indices
    # --bvecs: gradient directions
    # --bvals: b-values
    # --out: output prefix
    # --repol: replace outliers
    eddy_openmp \
        --imain="${DWI_FILE}" \
        --mask="${SUBJ_RESULTS_DIR}/b0_brain_mask" \
        --acqp="${ACQPARAMS_FILE}" \
        --index="${INDEX_FILE}" \
        --bvecs="${BVEC_FILE}" \
        --bvals="${BVAL_FILE}" \
        --out="${SUBJ_RESULTS_DIR}/dwi_eddy" \
        --repol \
        --data_is_shelled \
        --verbose

    echo "Step 5: Quality control - generating reports..."
    # Create QC directory
    QC_DIR="${SUBJ_RESULTS_DIR}/qc"
    mkdir -p "${QC_DIR}"

    # Generate eddy QC reports if eddy_quad is available
    if command -v eddy_quad &> /dev/null; then
        eddy_quad "${SUBJ_RESULTS_DIR}/dwi_eddy" \
            -idx "${INDEX_FILE}" \
            -par "${ACQPARAMS_FILE}" \
            -m "${SUBJ_RESULTS_DIR}/b0_brain_mask" \
            -b "${BVAL_FILE}" \
            -o "${QC_DIR}"
    fi

    # Copy corrected bvals and bvecs to output
    cp "${BVAL_FILE}" "${SUBJ_RESULTS_DIR}/dwi_eddy.bval"
    cp "${SUBJ_RESULTS_DIR}/dwi_eddy.eddy_rotated_bvecs" "${SUBJ_RESULTS_DIR}/dwi_eddy.bvec"

    echo "✓ Preprocessing completed for ${SUBJECT}"
    echo "  Output: ${SUBJ_RESULTS_DIR}/dwi_eddy.nii.gz"
    echo ""
}

# Main processing loop
echo "=========================================="
echo "DWI Preprocessing Pipeline"
echo "Dataset: OpenNeuro ds006131"
echo "Number of subjects: ${#SUBJECTS[@]}"
echo "=========================================="
echo ""

# Process all subjects
for SUBJECT in "${SUBJECTS[@]}"; do
    preprocess_subject "${SUBJECT}"
done

echo "=========================================="
echo "All preprocessing completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="
