#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "Setting up RVC dependencies..."
echo "================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Install RVC-related dependencies
echo ""
echo "--------------------------------"
echo "Installing RVC Python packages..."
echo "--------------------------------"

pip install --no-cache-dir \
    infer-rvc-python \
    pydub \
    edge-tts \
    audio-separator \
    fairseq \
    librosa \
    praat-parselmouth \
    pyworld \
    scipy \
    numba \
    resampy

# Install faiss for vector search (used by RVC index files)
echo ""
echo "--------------------------------"
echo "Installing FAISS..."
echo "--------------------------------"

pip install --no-cache-dir faiss-gpu

# Download required model files
echo ""
echo "--------------------------------"
echo "Downloading RVC model files..."
echo "--------------------------------"

# Create weights directory
mkdir -p weights

# Download RMVPE model (pitch extraction)
if [ ! -f "rmvpe.pt" ]; then
    echo "Downloading RMVPE model..."
    wget -q --show-progress https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/download/20230428/rmvpe.pt \
        -O rmvpe.pt || echo "Warning: Failed to download rmvpe.pt"
fi

# Download Hubert Base model
if [ ! -f "hubert_base.pt" ]; then
    echo "Downloading Hubert Base model..."
    wget -q --show-progress https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
        -O hubert_base.pt || echo "Warning: Failed to download hubert_base.pt"
fi

# Install additional audio processing tools
echo ""
echo "--------------------------------"
echo "Installing additional dependencies..."
echo "--------------------------------"

pip install --no-cache-dir \
    psutil \
    py-cpuinfo

# Check if tts_voice module exists, if not create a placeholder
if [ ! -f "tts_voice.py" ]; then
    echo "Creating tts_voice.py placeholder..."
    cat > tts_voice.py << 'EOF'
# TTS voice order mapping
tts_order_voice = {
    'en': 'en-US',
    'ko': 'ko-KR',
    'ja': 'ja-JP',
    'zh': 'zh-CN',
    # Add more language mappings as needed
}
EOF
fi

# Check if model_handler module exists, if not create a placeholder
if [ ! -f "model_handler.py" ]; then
    echo "Creating model_handler.py placeholder..."
    cat > model_handler.py << 'EOF'
# Model handler placeholder
class ModelHandler:
    def __init__(self):
        pass

    def load_model(self, model_path):
        pass
EOF
fi

echo ""
echo "================================="
echo "RVC setup completed successfully!"
echo "================================="
echo ""
echo "Model files location:"
echo "  - RMVPE: $SCRIPT_DIR/rmvpe.pt"
echo "  - Hubert: $SCRIPT_DIR/hubert_base.pt"
echo "  - Weights directory: $SCRIPT_DIR/weights/"
echo ""
echo "NOTE: Please place your RVC model files (.pth) and index files (.index) in:"
echo "  $SCRIPT_DIR/weights/<language>/<character_name>.pth"
echo "  $SCRIPT_DIR/weights/<language>/<character_name>.index"
echo ""
