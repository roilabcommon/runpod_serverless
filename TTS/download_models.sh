#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "Downloading TTS Models"
echo "================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Check if huggingface-hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install --no-cache-dir huggingface-hub
fi

# Download Spark-TTS-0.5B model
echo ""
echo "================================="
echo "Downloading Spark-TTS-0.5B model..."
echo "================================="

mkdir -p pretrained_models/Spark-TTS-0.5B

if python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SparkAudio/Spark-TTS-0.5B', local_dir='$SCRIPT_DIR/pretrained_models/Spark-TTS-0.5B', local_dir_use_symlinks=False)"; then
    echo "✓ Spark-TTS-0.5B model downloaded successfully"
    echo "Location: $SCRIPT_DIR/pretrained_models/Spark-TTS-0.5B"
else
    echo "✗ Failed to download Spark-TTS-0.5B model"
fi

# Download VibeVoice-7B model
echo ""
echo "================================="
echo "Downloading VibeVoice-7B model..."
echo "================================="

mkdir -p vibevoice/VibeVoice-7B

if python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vibevoice/VibeVoice-7B', local_dir='$SCRIPT_DIR/vibevoice/VibeVoice-7B', local_dir_use_symlinks=False)"; then
    echo "✓ VibeVoice-7B model downloaded successfully"
    echo "Location: $SCRIPT_DIR/vibevoice/VibeVoice-7B"
else
    echo "✗ Failed to download VibeVoice-7B model"
fi

echo ""
echo "================================="
echo "Model download completed!"
echo "================================="
echo ""
echo "Downloaded models:"
if [ -d "pretrained_models/Spark-TTS-0.5B" ]; then
    echo "  ✓ Spark-TTS-0.5B: $SCRIPT_DIR/pretrained_models/Spark-TTS-0.5B"
fi
if [ -d "vibevoice/VibeVoice-7B" ]; then
    echo "  ✓ VibeVoice-7B: $SCRIPT_DIR/vibevoice/VibeVoice-7B"
fi
echo ""
