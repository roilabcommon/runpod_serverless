#!/bin/bash
# RunPod Serverless 엔트리포인트 스크립트
# 1. Network Volume에 pretrained 모델 다운로드
# 2. handler.py 실행

set -e

VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
MODELS_DIR="$VOLUME_PATH/models"

echo "=========================================="
echo "RunPod TTS Service - Startup"
echo "=========================================="

# --- Network Volume 확인 ---
if [ -d "$VOLUME_PATH" ]; then
    echo "[OK] Network Volume detected: $VOLUME_PATH"
    mkdir -p "$MODELS_DIR"

    # --- Spark-TTS-0.5B 다운로드 ---
    SPARK_DIR="$MODELS_DIR/Spark-TTS-0.5B"
    if [ -f "$SPARK_DIR/config.json" ] && [ -d "$SPARK_DIR/BiCodec" ]; then
        echo "[OK] Spark-TTS-0.5B already exists on volume"
    else
        echo "[DOWNLOAD] Downloading Spark-TTS-0.5B to Network Volume..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='SparkAudio/Spark-TTS-0.5B',
    local_dir='$SPARK_DIR',
    local_dir_use_symlinks=False
)
print('[OK] Spark-TTS-0.5B download complete')
"
    fi

    # --- VibeVoice-7B 다운로드 ---
    VIBE_DIR="$MODELS_DIR/VibeVoice-7B"
    if [ -f "$VIBE_DIR/config.json" ]; then
        # config.json 존재 + safetensors 또는 bin 파일 확인
        HAS_WEIGHTS=$(find "$VIBE_DIR" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" \) | head -1)
        if [ -n "$HAS_WEIGHTS" ]; then
            echo "[OK] VibeVoice-7B already exists on volume"
        else
            echo "[DOWNLOAD] VibeVoice-7B incomplete. Re-downloading..."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vibevoice/VibeVoice-7B',
    local_dir='$VIBE_DIR',
    local_dir_use_symlinks=False
)
print('[OK] VibeVoice-7B download complete')
"
        fi
    else
        echo "[DOWNLOAD] Downloading VibeVoice-7B to Network Volume..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vibevoice/VibeVoice-7B',
    local_dir='$VIBE_DIR',
    local_dir_use_symlinks=False
)
print('[OK] VibeVoice-7B download complete')
"
    fi

    # --- 다운로드 결과 요약 ---
    echo ""
    echo "=========================================="
    echo "Models on Network Volume:"
    du -sh "$MODELS_DIR"/* 2>/dev/null || echo "  (empty)"
    echo "=========================================="

else
    echo "[WARN] Network Volume not mounted at $VOLUME_PATH"
    echo "[WARN] Models will be loaded from Docker image (if available)"
fi

echo ""
echo "[START] Starting handler.py..."
echo "=========================================="

# --- handler.py 실행 ---
exec python -u handler.py
