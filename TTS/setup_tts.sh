#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "Setting up TTS dependencies..."
echo "================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Install additional TTS-specific dependencies first
echo ""
echo "--------------------------------"
echo "Installing base TTS dependencies..."
echo "--------------------------------"

pip install --no-cache-dir \
    omegaconf \
    hydra-core \
    pyworld \
    praat-parselmouth \
    phonemizer \
    g2p-en \
    unidecode \
    inflect \
    torch \
    torchaudio \
    transformers \
    accelerate

# Install SparkTTS
echo ""
echo "--------------------------------"
echo "Installing SparkTTS..."
echo "--------------------------------"

if [ ! -d "sparktts" ] && [ ! -d "cli" ]; then
    if [ -f "clone_sparktts.sh" ]; then
        bash clone_sparktts.sh
    else
        echo "Cloning SparkTTS repository..."
        TEMP_DIR=$(mktemp -d)
        git clone --depth 1 https://github.com/SparkAudio/Spark-TTS.git "$TEMP_DIR/SparkTTS"

        if [ -d "$TEMP_DIR/SparkTTS" ]; then
            cd "$TEMP_DIR/SparkTTS"
            pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo "Warning: requirements.txt failed"
            pip install --no-cache-dir -e . 2>/dev/null || echo "Warning: setup.py failed"
            cd "$SCRIPT_DIR"

            # Copy only essential folders
            echo "Extracting SparkTTS essential folders..."
            [ -d "$TEMP_DIR/SparkTTS/sparktts" ] && cp -r "$TEMP_DIR/SparkTTS/sparktts" "$SCRIPT_DIR/" || echo "Warning: sparktts not found"
            [ -d "$TEMP_DIR/SparkTTS/src" ] && cp -r "$TEMP_DIR/SparkTTS/src" "$SCRIPT_DIR/" || echo "Warning: src not found"
            [ -d "$TEMP_DIR/SparkTTS/cli" ] && cp -r "$TEMP_DIR/SparkTTS/cli" "$SCRIPT_DIR/" || echo "Warning: cli not found"
            [ -d "$TEMP_DIR/SparkTTS/runtime" ] && cp -r "$TEMP_DIR/SparkTTS/runtime" "$SCRIPT_DIR/" || echo "Warning: runtime not found"

            rm -rf "$TEMP_DIR"
            echo "✓ SparkTTS essential folders extracted successfully"
        fi
    fi
else
    echo "SparkTTS folders already exist, skipping..."
fi

# Download models
echo ""
echo "--------------------------------"
echo "Downloading TTS models..."
echo "--------------------------------"

# Check if huggingface-hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install --no-cache-dir huggingface-hub
fi

# Download Spark-TTS model
echo ""
echo "Downloading Spark-TTS-0.5B model..."
mkdir -p pretrained_models/Spark-TTS-0.5B
if [ ! -d "pretrained_models/Spark-TTS-0.5B/config.json" ] && [ ! -f "pretrained_models/Spark-TTS-0.5B/.complete" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SparkAudio/Spark-TTS-0.5B', local_dir='$SCRIPT_DIR/pretrained_models/Spark-TTS-0.5B', local_dir_use_symlinks=False)" && \
    touch pretrained_models/Spark-TTS-0.5B/.complete && \
    echo "✓ Spark-TTS-0.5B model downloaded successfully" || \
    echo "✗ Failed to download Spark-TTS-0.5B model"
else
    echo "Spark-TTS-0.5B model already exists, skipping..."
fi

# Download VibeVoice model
echo ""
echo "Downloading VibeVoice-7B model..."
mkdir -p vibevoice/VibeVoice-7B
if [ ! -f "vibevoice/VibeVoice-7B/config.json" ] && [ ! -f "vibevoice/VibeVoice-7B/.complete" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vibevoice/VibeVoice-7B', local_dir='$SCRIPT_DIR/vibevoice/VibeVoice-7B', local_dir_use_symlinks=False)" && \
    touch vibevoice/VibeVoice-7B/.complete && \
    echo "✓ VibeVoice-7B model downloaded successfully" || \
    echo "✗ Failed to download VibeVoice-7B model"
else
    echo "VibeVoice-7B model already exists, skipping..."
fi

echo ""
echo "================================="
echo "TTS setup completed successfully!"
echo "================================="
echo ""
echo "Installed components:"
if [ -d "sparktts" ] || [ -d "cli" ]; then
    echo "  ✓ SparkTTS (essential folders):"
    [ -d "sparktts" ] && echo "    - sparktts: $SCRIPT_DIR/sparktts"
    [ -d "src" ] && echo "    - src: $SCRIPT_DIR/src"
    [ -d "cli" ] && echo "    - cli: $SCRIPT_DIR/cli"
    [ -d "runtime" ] && echo "    - runtime: $SCRIPT_DIR/runtime"
fi

echo ""
echo "Downloaded models:"
if [ -d "pretrained_models/Spark-TTS-0.5B" ]; then
    echo "  ✓ Spark-TTS-0.5B: $SCRIPT_DIR/pretrained_models/Spark-TTS-0.5B"
fi
if [ -d "vibevoice/VibeVoice-7B" ]; then
    echo "  ✓ VibeVoice-7B: $SCRIPT_DIR/vibevoice/VibeVoice-7B"
fi
echo ""
