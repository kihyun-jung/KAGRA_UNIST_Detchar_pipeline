# End-to-End MLOps Pipeline for Time-Series Anomaly Detection
**(시계열 데이터 이상 탐지 및 분류를 위한 End-to-End MLOps 파이프라인)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Framework-orange?logo=tensorflow)
![HTCondor](https://img.shields.io/badge/Orchestration-HTCondor-green)
![Docker](https://img.shields.io/badge/Environment-Conda%2FCVMFS-blue)

## 📋 Project Overview

본 프로젝트는 대규모 시계열 데이터(KAGRA Gravitational Wave Data)에서 발생하는 **비정상 신호(Anomaly/Glitch)를 실시간으로 탐지, 분석하고 CNN 모델을 통해 유형을 분류**하는 자동화 파이프라인입니다.

Raw Data 수집부터 전처리(ETL), 통계적 상관관계 분석, 스펙트로그램 이미지 변환, 그리고 Deep Learning 기반 분류까지의 전 과정을 **하나의 오케스트레이션 시스템**으로 구축하였습니다.

> **Key Point:** 본 레포지토리는 KISTI 슈퍼컴퓨팅 환경의 대규모 병렬 처리를 로컬 환경에서도 검증할 수 있도록 **Mock Data Generator**와 **Simulation Logic**을 포함하여 포트폴리오용으로 재구성되었습니다.

---

## 🏗️ System Architecture

전체 시스템은 **결합도가 낮은 4개의 모듈(Decoupled Modules)**로 구성되어 있으며, 각 단계는 독립적으로 실행되거나 전체 파이프라인으로 통합 운영될 수 있습니다.

### 1. ETL & Trigger Generation (Data Ingestion)
* **Role:** 대용량 Raw Binary Data(.gwf)를 로드하여 신호 대 잡음비(SNR)가 높은 구간을 필터링(Triggering)합니다.
* **Tech:** Python, GwPy, Dynamic Configuration.
* **Scalability:** 파일 리스트(FFL) 자동 생성 및 배치 처리를 통한 대용량 데이터 대응.

### 2. Statistical Correlation Analysis (Root Cause Analysis)
* **Role:** 탐지된 이상 신호가 메인 채널의 실제 신호인지, 보조 센서(환경 노이즈)에 의한 잡음인지 통계적으로 검증(Veto)합니다.
* **Tech:** Hierarchical Veto (Hveto), Data Mining.
* **Process:** 수백 개의 보조 채널(Aux Channels)과의 상관관계를 분석하여 노이즈 원인을 규명합니다.

### 3. Feature Engineering & Visualization (Signal Processing)
* **Role:** 분석된 이벤트의 주파수 특성을 시각화하여 머신러닝 모델의 입력 데이터(Feature)를 생성합니다.
* **Tech:** FFT, Q-transform (Time-Frequency Analysis), SNR-weighted Coherence.
* **Output:** 고해상도 Q-spectrogram 이미지 및 Coherence 그래프.

### 4. Deep Learning Classification (Model Serving)
* **Role:** 생성된 스펙트로그램을 CNN 모델에 주입하여 이상 신호의 유형(Class)을 분류합니다.
* **Tech:** **PyTorch** & **TensorFlow** (Hybrid Support), CNN, Transfer Learning ready.
* **Feature:** 학습(Train)과 추론(Inference) 파이프라인의 자동화, 다중 프레임워크 지원.

---

## 🛠️ Tech Stack & Skills

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas, SciPy, GWpy (Time-series analysis) |
| **Deep Learning** | **PyTorch**, **TensorFlow/Keras**, Torchvision |
| **Orchestration** | **HTCondor** (Batch Job Scheduling), Shell Scripting |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Conda, CVMFS (Cluster File System) |

---

## 🚀 Key Engineering Competencies

### 1. Distributed Computing & Orchestration
* 슈퍼컴퓨터(KISTI) 환경에서의 작업 스케줄링을 위해 **HTCondor** 제출 파일(.sub)을 동적으로 생성하는 스크립트 구현.
* 로컬 개발 환경과 클러스터 배포 환경을 구분하는 **Hybrid Execution Strategy** 적용 (환경 감지 후 로컬 시뮬레이션 또는 배치 작업 제출).

### 2. Automated Pipeline Design
* 하드코딩을 지양하고, 설정 파일(Config/Templates) 기반으로 경로와 파라미터를 제어하여 유지보수성 향상.
* 각 단계의 입출력(I/O)이 명확히 정의되어 있어, 데이터 파이프라인의 **멱등성(Idempotency)** 보장.

### 3. Testing & Reproducibility (Mocking)
* 보안상 공개 불가능한 민감 데이터와 서버 환경을 대체하기 위해 **Mock Data Generator** 개발.
* 외부 의존성 없이 로컬 머신에서도 전체 로직(ETL -> ML)을 검증할 수 있는 **Smoke Test** 환경 구축.

### 4. Multi-Framework Support
* 단일 프레임워크 종속성을 탈피하기 위해 **PyTorch**와 **TensorFlow** 두 가지 버전의 모델링 코드를 모두 구현 및 모듈화.
* `--framework` 인자를 통해 실행 시점에 런타임 결정 가능.

---

## 📂 Directory Structure

```text
KAGRA_UNIST_Detchar_pipeline/
├── config/             # Configuration Templates (IaC approach)
├── data/               # Data Storage (Raw, Mock, Training Set)
├── jobs/               # Generated Job Submission Files (HTCondor)
├── results/            # Analysis Outputs (Logs, Images, CSVs)
├── src/                # Source Code
│   ├── etl/            # Data Extraction & Transformation Modules
│   ├── analysis/       # Statistical Analysis & Signal Processing
│   ├── ml/             # Machine Learning Models (PT/TF)
│   └── utils/          # Utility Scripts (Mock Generators, Loggers)
├── run_*.py            # Pipeline Orchestrator Scripts
└── requirements.txt    # Dependency Management
```
---

---

## 💻 How to Run (Demo)

본 프로젝트는 의존성 설치 후 4단계의 파이프라인 스크립트를 순차적으로 실행하여 전체 공정을 시뮬레이션할 수 있습니다.

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Data Ingestion & ETL
Raw Data 생성 및 전처리 작업을 수행합니다.
```bash
python run_omicron_pipeline.py -y 2023 -m 6 -d 18
```

### 3. Statistical Analysis (Root Cause)
상관관계 분석 환경을 구성하고 결과를 시뮬레이션합니다.
```bash
python run_hveto_pipeline.py -s 2023-06-18 -e 2023-06-18
```

### 4. Feature Extraction (Spectrogram)
노이즈 원인 채널을 추출하고 시각화 데이터를 생성합니다.
(Local 환경에서는 Smoke Test 모드로 자동 전환되어 단일 채널만 처리합니다.)
```bash
python run_coherence_pipeline.py
```

### 5. AI Classification (Training & Inference)
CNN 모델을 학습하고 생성된 이미지를 분류합니다. (Framework 선택 가능)
```bash
# Option A: PyTorch
python run_ml_pipeline.py --framework pytorch

# Option B: TensorFlow
python run_ml_pipeline.py --framework tensorflow
```

---

## 📊 Pipeline Execution Results (Preview)

본 파이프라인은 Raw Data에서 시작하여 최종 AI 분류까지 데이터가 변환되는 전 과정을 시각화합니다.

| Step 1. Signal Detection | Step 2. Root Cause Analysis | Step 3. Visualization & Classification |
| :---: | :---: | :---: |
| **Omicron ETL** | **Hveto Analysis** | **Final Output (Q-scan & AI)** |
| 📄 **XML/CSV Logs Generated**<br>*(SNR > 8 Triggers Extracted)* | ⚙️ **Veto Segments Identified**<br>*(Noise Correlation Logs)* | **Example of Q-spectrogram**<br><img width="350" height="400" alt="Main-DAC-STRAIN_C20-1270316463 693-0 5" src="https://github.com/user-attachments/assets/297d2f5a-a140-4ff9-bef1-b003048fb7ce" /> |
| *Data Ingestion & Filtering* | *Statistical Verification* | *Generated Spectrogram & Predicted Label* |

> **Note:** 최종 단계에서 생성된 Q-scan 이미지는 `results/qscan/` 폴더에서 확인할 수 있으며, AI 모델의 예측 결과는 CSV 파일로 저장됩니다.

---

**Author:** Kihyun Jung
**Contact:** khjung@unist.ac.kr or wjk9364@gmail.com
**Institution:** UNIST (Ulsan National Institute of Science and Technology)
