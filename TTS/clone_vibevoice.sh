#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "Downloading VibeVoice-7B Model"
echo "================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TTS_DIR="$SCRIPT_DIR"

echo "TTS Directory: $TTS_DIR"

# Check if huggingface-hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install --no-cache-dir huggingface-hub
fi

# Create vibevoice directory
mkdir -p "$TTS_DIR/vibevoice/VibeVoice-7B"

echo ""
echo "Downloading VibeVoice-7B model from Hugging Face..."
if python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vibevoice/VibeVoice-7B', local_dir='$TTS_DIR/vibevoice/VibeVoice-7B', local_dir_use_symlinks=False)"; then
    echo ""
    echo "================================="
    echo "✓ VibeVoice-7B model downloaded!"
    echo "================================="
    echo "Location: $TTS_DIR/vibevoice/VibeVoice-7B"
    echo ""
    echo "Model files:"
    ls -la "$TTS_DIR/vibevoice/VibeVoice-7B" | head -10
else
    echo "✗ Failed to download VibeVoice-7B model"
    exit 1
fi
