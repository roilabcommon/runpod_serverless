# Docker Setup Guide

이 가이드는 RunPod Serverless 프로젝트를 Docker로 빌드하고 실행하는 방법을 설명합니다.

## 사전 요구사항

- Docker 설치
- NVIDIA Docker Runtime (GPU 사용 시)
- Docker Compose 설치

## 프로젝트 구조

```
runpod_serverless/
├── Dockerfile                 # Docker 이미지 빌드 파일
├── docker-compose.yml         # Docker Compose 설정
├── build.sh                   # 빌드 헬퍼 스크립트
├── requirements.txt           # Python 의존성
├── .dockerignore             # Docker 빌드 제외 파일
├── TTS/
│   ├── SparkModel.py
│   ├── VibeVoiceModel.py
│   ├── setup_tts.sh          # TTS 통합 설치 스크립트
│   ├── clone_sparktts.sh     # SparkTTS 전용 클론 스크립트
│   ├── clone_vibevoice.sh    # VibeVoice 전용 클론 스크립트
│   ├── requirements.txt
│   ├── SparkTTS/             # SparkTTS 레포지토리 (빌드 중 클론됨)
│   ├── vibevoice/            # VibeVoice 모듈만 추출됨 (공간 절약)
│   └── pretrained_models/    # 사전 학습 모델 저장 경로
└── RVC/
    ├── RVC.py
    ├── setup_rvc.sh          # RVC 설치 스크립트
    ├── requirements.txt
    ├── rmvpe.pt              # 자동 다운로드됨
    ├── hubert_base.pt        # 자동 다운로드됨
    └── weights/              # RVC 모델 가중치 저장 경로
```

## 빌드 방법

### 1. Docker 이미지 빌드

```bash
# 기본 빌드
docker build -t runpod-serverless:latest .

# 또는 Docker Compose 사용
docker-compose build
```

빌드 과정에서 자동으로 수행되는 작업:
- PyTorch 및 CUDA 환경 설정
- TTS 의존성 설치 (SparkTTS, VibeVoice)
  - **SparkTTS**: 전체 레포지토리 클론
  - **VibeVoice**: `vibevoice` 모듈만 추출 (공간 절약)
- RVC 의존성 설치
- fairseq를 GitHub에서 소스로 빌드 (PyPI 버전 문제 해결)
- 필요한 모델 파일 다운로드 (rmvpe.pt, hubert_base.pt)

### 빌드 최적화 기능

Docker 빌드는 다음과 같이 최적화되었습니다:

1. **선택적 복사**: VibeVoice는 필요한 `vibevoice` 모듈만 복사하여 이미지 크기 감소
2. **Shallow Clone**: `--depth 1`로 git 히스토리 없이 최신 버전만 클론
3. **에러 내구성**: 개별 패키지 설치 실패 시에도 빌드 계속 진행

### 2. 모델 파일 준비

#### SparkTTS 모델
```bash
# SparkTTS 모델을 다운로드하여 아래 경로에 저장
./TTS/pretrained_models/Spark-TTS-0.5B/
```

#### VibeVoice 모델
VibeVoice 모델은 첫 실행 시 Hugging Face에서 자동으로 다운로드됩니다.
- 모델 이름: `vibevoice/VibeVoice-7B`

#### RVC 모델
```bash
# RVC 캐릭터 모델 파일(.pth)과 인덱스 파일(.index)을 저장
./RVC/weights/<language>/<character_name>.pth
./RVC/weights/<language>/<character_name>.index

# 예시:
./RVC/weights/ko/Poli.pth
./RVC/weights/ko/Poli.index
```

## 독립 스크립트 사용 (선택사항)

Docker를 사용하지 않고 직접 설치하려면 다음 스크립트를 사용할 수 있습니다:

### SparkTTS 설치

```bash
cd TTS
chmod +x clone_sparktts.sh
./clone_sparktts.sh
```

이 스크립트는:
- SparkTTS 레포지토리를 임시 디렉토리에 클론
- 의존성 설치
- `TTS/SparkTTS` 디렉토리에 복사
- 임시 파일 정리

### VibeVoice 설치

```bash
cd TTS
chmod +x clone_vibevoice.sh
./clone_vibevoice.sh
```

이 스크립트는:
- VibeVoice 레포지토리를 임시 디렉토리에 클론
- 의존성 설치
- **`vibevoice` 모듈만** `TTS/vibevoice`에 복사 (공간 절약)
- 전체 레포지토리 정리 (불필요한 파일 제거)

### 전체 TTS 설치

```bash
cd TTS
chmod +x setup_tts.sh
./setup_tts.sh
```

이 스크립트는 SparkTTS와 VibeVoice를 모두 설치합니다.

### RVC 설치

```bash
cd RVC
chmod +x setup_rvc.sh
./setup_rvc.sh
```

## 실행 방법

### Docker Compose 사용 (권장)

```bash
# 컨테이너 시작
docker-compose up -d

# 컨테이너 접속
docker-compose exec runpod-serverless bash

# 로그 확인
docker-compose logs -f

# 컨테이너 중지
docker-compose down
```

### Docker 직접 사용

```bash
# 컨테이너 실행
docker run -it --gpus all \
  -v $(pwd)/TTS/pretrained_models:/app/TTS/pretrained_models \
  -v $(pwd)/RVC/weights:/app/RVC/weights \
  -v $(pwd)/example/results:/app/example/results \
  runpod-serverless:latest bash

# 백그라운드 실행
docker run -d --gpus all \
  --name runpod-serverless \
  -v $(pwd)/TTS/pretrained_models:/app/TTS/pretrained_models \
  -v $(pwd)/RVC/weights:/app/RVC/weights \
  -v $(pwd)/example/results:/app/example/results \
  runpod-serverless:latest
```

## 모델 테스트

컨테이너 내부에서 다음과 같이 테스트할 수 있습니다:

```python
# Python 인터프리터 실행
python

# SparkTTS 테스트
from TTS.SparkModel import SparkModel
model = SparkModel()

# VibeVoice 테스트
from TTS.VibeVoiceModel import VibeVoiceModel
model = VibeVoiceModel()

# RVC 테스트
from RVC.RVC import RVC
rvc = RVC()
```

## 볼륨 마운트

Docker Compose는 다음 디렉토리를 자동으로 마운트합니다:

- `./TTS/pretrained_models` → `/app/TTS/pretrained_models` (TTS 모델)
- `./RVC/weights` → `/app/RVC/weights` (RVC 모델)
- `./example/results` → `/app/example/results` (생성된 오디오)

이를 통해 컨테이너를 삭제해도 모델과 결과물이 보존됩니다.

## 문제 해결

### GPU 인식 안 됨
```bash
# NVIDIA Docker Runtime 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# nvidia-container-toolkit 설치 (Ubuntu)
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 메모리 부족 오류
```bash
# Docker 메모리 제한 증가 또는 코드에서 배치 크기 감소
# VibeVoiceModel.py에서 4-bit 양자화가 활성화되어 있는지 확인
```

### 모델 다운로드 실패
```bash
# 컨테이너 내부에서 수동 다운로드
cd /app/RVC
wget https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/download/20230428/rmvpe.pt
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
```

## 개발 모드

소스 코드를 수정하면서 개발하려면 `docker-compose.yml`에서 주석 처리된 볼륨 마운트를 활성화하세요:

```yaml
volumes:
  - ./TTS:/app/TTS
  - ./RVC:/app/RVC
```

이후 컨테이너를 재시작하면 변경 사항이 즉시 반영됩니다.

## 추가 정보

- SparkTTS GitHub: https://github.com/laksjdjf/SparkTTS
- VibeVoice GitHub: https://github.com/microsoft/VibeVoice
- CUDA 버전: 12.1.0
- PyTorch 버전: 2.1.0
