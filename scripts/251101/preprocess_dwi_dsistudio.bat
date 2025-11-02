@echo off
REM DWI Preprocessing using DSI Studio
REM Windows Batch Script
REM Date: 2025-11-01

echo ============================================================
echo DSI Studio DWI Preprocessing
echo Dataset: OpenNeuro ds006131
echo ============================================================
echo.

set DSI_STUDIO="C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\dsi_studio.exe"
set DATA_DIR="C:\Users\Public\human_brain_lecture\data\251101"
set RESULTS_DIR="C:\Users\Public\human_brain_lecture\results\251101"

REM Check if DSI Studio exists
if not exist %DSI_STUDIO% (
    echo ERROR: DSI Studio not found at %DSI_STUDIO%
    pause
    exit /b 1
)

echo DSI Studio found: %DSI_STUDIO%
echo.

REM Process each subject
for %%S in (sub-24053 sub-24630 sub-24663 sub-24683 sub-24684 sub-24696 sub-24697 sub-24699 sub-24702 sub-60295) do (
    echo.
    echo ============================================================
    echo Processing: %%S
    echo ============================================================

    set SUBJ_DWI_DIR=%DATA_DIR%\%%S\ses-1\dwi
    set SUBJ_RESULTS_DIR=%RESULTS_DIR%\%%S\ses-1\dwi

    REM Create output directory
    if not exist %SUBJ_RESULTS_DIR% mkdir %SUBJ_RESULTS_DIR%

    REM Define files
    set DWI_FILE=%DATA_DIR%\%%S\ses-1\dwi\%%S_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz
    set BVAL_FILE=%DATA_DIR%\%%S\ses-1\dwi\%%S_ses-1_dir-AP_run-01_part-mag_dwi.bval
    set BVEC_FILE=%DATA_DIR%\%%S\ses-1\dwi\%%S_ses-1_dir-AP_run-01_part-mag_dwi.bvec
    set SRC_FILE=%RESULTS_DIR%\%%S\ses-1\dwi\%%S_dwi.src.gz

    echo Creating SRC file with preprocessing...
    %DSI_STUDIO% --action=src --source=!DWI_FILE! --bval=!BVAL_FILE! --bvec=!BVEC_FILE! --output=!SRC_FILE!

    if errorlevel 1 (
        echo ERROR: Failed to process %%S
    ) else (
        echo SUCCESS: %%S processed
    )

    echo.
)

echo.
echo ============================================================
echo All subjects processed!
echo Results saved to: %RESULTS_DIR%
echo ============================================================
pause
