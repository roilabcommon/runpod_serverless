# RunPod Serverless TTS Handler

RunPod serverless 환경에서 VibeVoice와 Spark TTS 모델을 사용하여 음성 합성을 수행하는 핸들러입니다.

## 기능

- VibeVoice-7B 모델 지원
- Spark-TTS-0.5B 모델 지원
- Base64 인코딩 또는 URL로 음성 샘플 입력
- Base64로 인코딩된 오디오 출력
- GPU 메모리 최적화 (4-bit quantization)

## 입력 형식

```json
{
  "input": {
    "text": "합성할 텍스트",
    "prompt_speech": "base64_encoded_audio 또는 http://url/to/audio.wav",
    "model_type": "vibevoice",  // "vibevoice" 또는 "spark" (기본값: "vibevoice")
    "cfg_scale": 2.0,  // 선택사항, VibeVoice 전용 (기본값: 2.0)
    "return_format": "base64"  // 선택사항 (기본값: "base64")
  }
}
```

### 필수 파라미터

- `text`: 음성으로 합성할 텍스트
- `prompt_speech`: 음성 샘플 (base64 인코딩 또는 URL)

### 선택 파라미터

- `model_type`: 사용할 모델 (`"vibevoice"` 또는 `"spark"`)
- `cfg_scale`: VibeVoice 모델의 Classifier-Free Guidance 스케일 (기본값: 2.0)
- `return_format`: 출력 형식 (현재는 `"base64"`만 지원)

## 출력 형식

```json
{
  "audio": "base64_encoded_audio_data",
  "sample_rate": 24000,
  "model_used": "vibevoice",
  "text_length": 150
}
```

## 로컬 테스트

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드

```bash
# VibeVoice 모델
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vibevoice/VibeVoice-7B', local_dir='TTS/vibevoice/VibeVoice-7B')"

# Spark TTS 모델
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SparkAudio/Spark-TTS-0.5B', local_dir='TTS/pretrained_models/Spark-TTS-0.5B')"
```

### 3. 로컬 테스트 실행

```python
import runpod
import base64

# 음성 파일을 base64로 인코딩
with open("sample_voice.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# 테스트 입력
test_input = {
    "input": {
        "text": "안녕하세요. 테스트 음성 합성입니다.",
        "prompt_speech": audio_base64,
        "model_type": "vibevoice",
        "cfg_scale": 2.0
    }
}

# handler.py를 import하여 테스트
from handler import handler, initialize_models

# 모델 초기화
initialize_models()

# TTS 생성
result = handler(test_input)

# 결과 저장
if "audio" in result:
    output_audio = base64.b64decode(result["audio"])
    with open("output.wav", "wb") as f:
        f.write(output_audio)
    print(f"✅ 음성 생성 완료: {result['sample_rate']}Hz, {result['text_length']} characters")
else:
    print(f"❌ 에러: {result.get('error')}")
```

## Docker 빌드 및 배포

### 1. Docker 이미지 빌드

```bash
docker build -t runpod-tts-serverless .
```

### 2. 로컬 Docker 테스트

```bash
docker run --gpus all -it runpod-tts-serverless
```

### 3. RunPod에 배포

1. Docker 이미지를 Docker Hub 또는 다른 레지스트리에 푸시

```bash
docker tag runpod-tts-serverless:latest yourusername/runpod-tts-serverless:latest
docker push yourusername/runpod-tts-serverless:latest
```

2. RunPod 콘솔에서 Serverless 엔드포인트 생성
   - Image: `yourusername/runpod-tts-serverless:latest`
   - GPU: RTX 4090 또는 A100 권장 (VibeVoice는 최소 11GB VRAM 필요)

3. 엔드포인트 테스트

```python
import runpod
import base64

runpod.api_key = "your_runpod_api_key"

# 음성 파일 준비
with open("sample_voice.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# RunPod 엔드포인트 호출
endpoint = runpod.Endpoint("ENDPOINT_ID")

run_request = endpoint.run(
    {
        "text": "안녕하세요. RunPod 서버리스 테스트입니다.",
        "prompt_speech": audio_base64,
        "model_type": "vibevoice"
    }
)

# 결과 확인
result = run_request.output()
print(result)
```

## 모델별 특징

### VibeVoice-7B
- 샘플레이트: 24kHz
- VRAM 요구사항: ~4-5GB (4-bit quantization)
- 고품질 음성 합성
- 다양한 화자 지원

### Spark-TTS-0.5B
- 샘플레이트: 16kHz
- VRAM 요구사항: ~2GB
- 빠른 추론 속도
- 경량 모델

## 에러 처리

handler는 다음과 같은 에러를 반환할 수 있습니다:

- `"Missing required parameter: text"`: 텍스트가 제공되지 않음
- `"Missing required parameter: prompt_speech"`: 음성 샘플이 제공되지 않음
- `"Invalid model_type"`: 잘못된 모델 타입
- `"VibeVoice model is not available"`: VibeVoice 모델 로드 실패
- `"Spark model is not available"`: Spark 모델 로드 실패
- `"Failed to download prompt speech from URL"`: URL에서 음성 다운로드 실패
- `"Failed to decode prompt speech from base64"`: Base64 디코딩 실패
- `"TTS generation failed"`: 음성 생성 실패

## 성능 최적화

1. **GPU 메모리 최적화**: VibeVoice는 4-bit quantization을 사용하여 11GB VRAM에서 실행 가능
2. **모델 캐싱**: 모델은 시작 시 한 번만 로드되어 메모리에 유지
3. **임시 파일 관리**: `tempfile.TemporaryDirectory()`를 사용하여 자동으로 정리

## 문제 해결

### CUDA Out of Memory
- VibeVoice 대신 Spark 모델 사용
- 더 큰 GPU 사용 (RTX 4090, A100 권장)
- `inference_steps`를 줄여서 메모리 사용량 감소

### 모델 로드 실패
- 로그 확인: Docker 컨테이너의 모델 다운로드 상태 확인
- 수동 다운로드: Hugging Face에서 모델을 수동으로 다운로드하여 적절한 경로에 배치

### 음성 품질 문제
- `cfg_scale` 조정 (1.5 ~ 3.0 범위에서 실험)
- 더 긴 프롬프트 음성 사용 (최소 5초 권장)
- 고품질 음성 샘플 사용 (노이즈 없는 깨끗한 오디오)
