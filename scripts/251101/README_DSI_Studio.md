# DWI Preprocessing with DSI Studio

## Overview
DSI Studio를 사용한 Diffusion Weighted Imaging (DWI) 데이터 전처리 스크립트입니다.

## DSI Studio란?
- **공식 사이트**: https://dsi-studio.labsolver.org/
- **개발자**: Fang-Cheng (Frank) Yeh, Carnegie Mellon University
- **특징**:
  - Windows 네이티브 지원 (WSL 불필요)
  - 올인원 DWI 분석 도구
  - GUI와 CLI 모두 지원
  - 자동 전처리 기능 내장

## DSI Studio vs FSL eddy

| 특징 | DSI Studio | FSL eddy |
|------|------------|----------|
| **Windows 지원** | ✓ 네이티브 | WSL 필요 |
| **설치 난이도** | 쉬움 (압축 해제) | 복잡 (컴파일 필요) |
| **전처리 속도** | 빠름 | 느림~빠름 (GPU) |
| **자동화** | 높음 | 수동 설정 필요 |
| **GUI** | ✓ | ✗ |
| **출력 형식** | SRC (자체), NII | NII |
| **추가 분석** | 내장 (DTI, GQI 등) | 별도 도구 필요 |

## 전처리 과정

### DSI Studio가 자동으로 수행하는 전처리:

1. **Eddy Current Correction (와류 보정)**
   - 경사자장으로 인한 이미지 왜곡 제거
   - 각 diffusion gradient 방향마다 다른 왜곡 패턴 보정

2. **Motion Correction (움직임 보정)**
   - 스캔 중 피험자 머리 움직임 보정
   - 각 볼륨을 b0 이미지에 정렬
   - Gradient 방향도 함께 회전

3. **Signal Drift Correction (신호 드리프트 보정)**
   - 시간에 따른 신호 감쇠 보정
   - 스캐너 불안정성 보상

4. **Gradient Nonlinearity Correction (경사자장 비선형성 보정)**
   - 스캐너 경사자장의 비선형성으로 인한 왜곡 보정

5. **데이터 품질 평가**
   - 각 볼륨의 품질 자동 평가
   - 낮은 품질의 슬라이스 자동 탐지

## 사용법

### 필수 요구사항
- **DSI Studio**: `C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\dsi_studio.exe`
- **Python 3.x** (Python 스크립트 사용 시)
- **충분한 디스크 공간** (subject당 ~500MB)

### 실행 방법

#### Option 1: Python 스크립트 (권장)
```bash
python C:\Users\Public\human_brain_lecture\scripts\251101\preprocess_dwi_dsistudio.py
```

**장점**:
- 진행 상황 상세 출력
- 에러 핸들링
- QC 파일 자동 생성
- 전체 요약 리포트

#### Option 2: 배치 스크립트 (간단)
```cmd
C:\Users\Public\human_brain_lecture\scripts\251101\preprocess_dwi_dsistudio.bat
```

**장점**:
- Python 불필요
- 더블클릭으로 실행 가능

#### Option 3: 수동 실행 (단일 subject)
```cmd
"C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\dsi_studio.exe" ^
  --action=src ^
  --source="C:\Users\Public\human_brain_lecture\data\251101\sub-24053\ses-1\dwi\sub-24053_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz" ^
  --bval="C:\Users\Public\human_brain_lecture\data\251101\sub-24053\ses-1\dwi\sub-24053_ses-1_dir-AP_run-01_part-mag_dwi.bval" ^
  --bvec="C:\Users\Public\human_brain_lecture\data\251101\sub-24053\ses-1\dwi\sub-24053_ses-1_dir-AP_run-01_part-mag_dwi.bvec" ^
  --output="C:\Users\Public\human_brain_lecture\results\251101\sub-24053\ses-1\dwi\sub-24053_dwi.src.gz"
```

## 입력 데이터
- **경로**: `C:\Users\Public\human_brain_lecture\data\251101\`
- **파일 구조**:
  ```
  sub-{ID}/ses-1/dwi/
  ├── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.nii.gz  (DWI 데이터)
  ├── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.bval   (b-values)
  └── sub-{ID}_ses-1_dir-AP_run-01_part-mag_dwi.bvec   (gradient vectors)
  ```

## 출력 데이터
- **경로**: `C:\Users\Public\human_brain_lecture\results\251101\`
- **주요 파일**:
  ```
  sub-{ID}/ses-1/dwi/
  ├── sub-{ID}_dwi.src.gz.sz          # DSI Studio SRC 파일 (주요 출력)
  ├── preprocessing_summary.txt        # 전처리 요약
  └── qc/
      └── preprocessing_info.txt       # QC 정보
  ```

### SRC 파일 (.src.gz)
DSI Studio의 전용 포맷으로, 다음 정보를 포함:
- 전처리된 DWI 데이터
- b-values와 b-vectors
- 전처리 파라미터
- 데이터 품질 정보

## 처리 시간
- **Subject당**: 약 **2-5분** (CPU 성능에 따라)
- **전체 10명**: 약 **20-50분**
- **FSL eddy 대비**: 약 5-10배 빠름

## 다음 단계: 재구성 (Reconstruction)

SRC 파일 생성 후에는 다양한 diffusion model로 재구성 가능:

### 1. DTI (Diffusion Tensor Imaging)
```cmd
dsi_studio.exe --action=rec --source=sub-24053_dwi.src.gz.sz --method=1 --output=sub-24053_dti.fib.gz
```
- **Method 1**: DTI
- **출력**: FA, MD, AD, RD 맵

### 2. GQI (Generalized Q-Sampling Imaging)
```cmd
dsi_studio.exe --action=rec --source=sub-24053_dwi.src.gz.sz --method=4 --param0=1.25 --output=sub-24053_gqi.fib.gz
```
- **Method 4**: GQI
- **Param0**: Sampling length ratio (기본값 1.25)
- **장점**: Multi-shell 데이터에 최적화, crossing fiber 해결

### 3. Fiber Tracking (섬유 추적)
```cmd
dsi_studio.exe --action=trk --source=sub-24053_gqi.fib.gz --method=0 --fiber_count=100000 --output=sub-24053_track.tt.gz
```

## GUI에서 결과 확인

DSI Studio GUI에서 결과를 시각화할 수 있습니다:

1. **DSI Studio 실행**:
   ```
   C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\dsi_studio.exe
   ```

2. **SRC 파일 열기**: `Step T1: Open Source Images`
   - SRC 파일 (`.src.gz.sz`) 선택
   - 전처리된 데이터와 품질 지표 확인

3. **재구성**: `Step T2: Reconstruction`
   - DTI 또는 GQI 선택
   - 파라미터 설정 후 실행

4. **섬유 추적**: `Step T3: Fiber Tracking`
   - FIB 파일 열기
   - ROI 그리기 및 tractography 실행

## 품질 관리 (QC)

전처리 품질을 확인하는 방법:

1. **GUI에서 시각적 검사**:
   - SRC 파일을 DSI Studio에서 열기
   - 각 볼륨 확인 (슬라이더로 이동)
   - 움직임 아티팩트 확인

2. **자동 생성된 리포트**:
   - `preprocessing_summary.txt`: 전처리 정보
   - `preprocessing_info.txt`: 기본 데이터 정보

3. **수동 검사 항목**:
   - [ ] 모든 subject의 SRC 파일 생성 확인
   - [ ] 파일 크기 정상 범위 (200-500MB)
   - [ ] 에러 메시지 확인

## 문제 해결 (Troubleshooting)

### DSI Studio 실행 오류
```
ERROR: The application was unable to start correctly (0xc000007b)
```
**해결책**: Visual C++ Redistributable 설치
```cmd
C:\Users\Public\dsi_studio_win_cpu\dsi_studio_win\vc_redist.x64.exe
```

### 메모리 부족
**증상**: Processing이 중간에 멈춤
**해결책**: 한 번에 처리하는 subject 수 줄이기

### SRC 파일 생성 실패
**확인 사항**:
1. DWI 파일 존재 여부
2. BVAL/BVEC 파일 매칭
3. 디스크 공간 충분한지

## 추가 자료

- **DSI Studio 매뉴얼**: https://dsi-studio.labsolver.org/doc/
- **CLI 사용법**: https://dsi-studio.labsolver.org/doc/cli_t1.html
- **YouTube 튜토리얼**: https://www.youtube.com/c/DSIStudio

## 비교: DSI Studio 명령어

### 전체 파이프라인 (한 subject)
```cmd
# 1. 전처리 + SRC 생성
dsi_studio.exe --action=src --source=dwi.nii.gz --bval=dwi.bval --bvec=dwi.bvec --output=dwi.src.gz

# 2. GQI 재구성
dsi_studio.exe --action=rec --source=dwi.src.gz.sz --method=4 --param0=1.25 --output=dwi.fib.gz

# 3. 전뇌 tractography
dsi_studio.exe --action=trk --source=dwi.fib.gz --method=0 --fiber_count=100000 --output=whole_brain.tt.gz

# 4. FA 맵 추출
dsi_studio.exe --action=exp --source=dwi.fib.gz --export=fa
```

## Summary

**DSI Studio 전처리의 핵심 장점**:
1. ✓ Windows에서 바로 실행 (WSL 불필요)
2. ✓ 빠른 처리 속도 (FSL 대비 5-10배)
3. ✓ 자동 전처리 (설정 최소화)
4. ✓ 통합 분석 환경 (전처리→재구성→tracking→분석)
5. ✓ 시각화 도구 내장

**권장 워크플로우**:
```
다운로드 → DSI Studio 전처리 (SRC) → GQI 재구성 (FIB) → Tractography → 분석
```
