# DWI Preprocessing with FSL eddy

## Overview
이 스크립트는 OpenNeuro dataset ds006131의 Diffusion Weighted Imaging (DWI) 데이터를 FSL의 eddy 도구를 사용하여 전처리합니다.

## 전처리 단계

### 1. Brain Mask 생성
- **도구**: FSL BET (Brain Extraction Tool)
- **과정**: 첫 번째 b0 볼륨(b-value ≈ 0)을 추출하여 뇌 마스크 생성
- **목적**: eddy correction 시 뇌 영역만 처리하여 정확도 향상

### 2. Index File 생성
- **파일**: `index.txt`
- **내용**: 각 볼륨이 어떤 acquisition parameter를 사용했는지 지정
- **형식**: 모든 볼륨에 대해 "1"로 설정 (단일 acquisition protocol 사용)

### 3. Acquisition Parameters 설정
- **파일**: `acqparams.txt`
- **내용**: Phase encoding 방향과 total readout time
- **형식**: `0 -1 0 0.0526491`
  - `0 -1 0`: PhaseEncodingDirection "j-" (Anterior >> Posterior)
  - `0.0526491`: TotalReadoutTime (seconds)

### 4. Eddy Current 및 Motion Correction
- **도구**: FSL eddy_openmp
- **보정 내용**:
  - **Eddy current distortion**: 경사자장(gradient)에 의한 와류 왜곡 보정
  - **Head motion**: 스캔 중 머리 움직임 보정
  - **Outlier replacement**: 이상치 데이터 대체 (`--repol` 옵션)
- **출력**:
  - `dwi_eddy.nii.gz`: 보정된 DWI 데이터
  - `dwi_eddy.eddy_rotated_bvecs`: 회전 보정된 gradient 방향
  - `dwi_eddy.eddy_movement_rms`: 움직임 정보
  - `dwi_eddy.eddy_parameters`: 보정 파라미터

### 5. Quality Control
- **도구**: eddy_quad (선택사항)
- **내용**: 전처리 품질 보고서 생성
- **출력 위치**: `qc/` 디렉토리

## 사용법

### 필수 요구사항
1. **FSL 설치**
   - 다운로드: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
   - Windows: WSL (Windows Subsystem for Linux) 환경에서 실행 권장

2. **Python** (Python 스크립트 사용 시)
   ```bash
   pip install nibabel
   ```

### 실행 방법

#### Option 1: Bash 스크립트 (Linux/WSL)
```bash
# 실행 권한 부여
chmod +x preprocess_dwi_eddy.sh

# 실행
./preprocess_dwi_eddy.sh
```

#### Option 2: Python 스크립트 (Windows/Linux)
```bash
python preprocess_dwi_eddy.py
```

## 입력 데이터
- **경로**: `C:\Users\Public\human_brain_lecture\data\251101\`
- **구조**:
  ```
  sub-{ID}/
    └── ses-1/
        └── dwi/
            ├── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz
            ├── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.bval
            ├── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.bvec
            └── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.json
  ```

## 출력 데이터
- **경로**: `C:\Users\Public\human_brain_lecture\results\251101\`
- **주요 파일**:
  ```
  sub-{ID}/
    └── ses-1/
        └── dwi/
            ├── dwi_eddy.nii.gz          # 보정된 DWI 데이터
            ├── dwi_eddy.bval            # b-values
            ├── dwi_eddy.bvec            # 회전 보정된 gradient vectors
            ├── b0_brain_mask.nii.gz     # 뇌 마스크
            ├── index.txt                # Index 파일
            ├── acqparams.txt            # Acquisition parameters
            └── qc/                      # QC 리포트
  ```

## 처리 시간
- 주체당 약 **10-30분** 소요 (CPU 사양에 따라 다름)
- 전체 10명 처리: 약 **2-5시간**

## 주의사항
1. **충분한 디스크 공간 확보** (각 subject당 ~1-2GB 추가 필요)
2. **FSL 환경 변수 설정** 확인
3. **GPU 사용 가능 시**: `eddy_openmp` 대신 `eddy_cuda` 사용 (훨씬 빠름)

## 다음 단계
전처리 완료 후 수행 가능한 분석:
1. **DTI fitting**: `dtifit`으로 FA, MD 맵 생성
2. **Tractography**: `probtrackx`로 백질 섬유 추적
3. **Connectivity analysis**: 뇌 영역 간 연결성 분석

## References
- FSL eddy: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy
- BIDS format: https://bids.neuroimaging.io/
- Dataset: OpenNeuro ds006131
