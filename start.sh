#!/bin/bash
# RunPod Serverless entrypoint – optimised for cold-start speed
# 1. Fast model-presence check on Network Volume
# 2. Download only if missing
# 3. Launch handler.py

set -e

VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
MODELS_DIR="$VOLUME_PATH/models"

echo "=========================================="
echo "RunPod TTS Service - Startup"
echo "=========================================="

# --- Network Volume check ---
if [ -d "$VOLUME_PATH" ]; then
    echo "[OK] Network Volume: $VOLUME_PATH"
    mkdir -p "$MODELS_DIR"

    # --- Spark-TTS-0.5B: fast marker check ---
    SPARK_DIR="$MODELS_DIR/Spark-TTS-0.5B"
    if [ -f "$SPARK_DIR/config.yaml" ] && [ -d "$SPARK_DIR/BiCodec" ]; then
        echo "[OK] Spark-TTS-0.5B present"
    else
        echo "[DOWNLOAD] Downloading Spark-TTS-0.5B..."
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

    # --- VibeVoice-7B: fast marker check ---
    VIBE_DIR="$MODELS_DIR/VibeVoice-7B"
    if [ -f "$VIBE_DIR/config.json" ] && ls "$VIBE_DIR"/*.safetensors 1>/dev/null 2>&1; then
        echo "[OK] VibeVoice-7B present"
    else
        echo "[DOWNLOAD] Downloading VibeVoice-7B..."
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
    echo "[WARN] Network Volume not mounted at $VOLUME_PATH"
    echo "[WARN] Models will be loaded from Docker image (if available)"
fi

echo "[START] Starting handler.py..."
exec python -u handler.py
