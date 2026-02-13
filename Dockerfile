# Use NVIDIA CUDA base image with Python
# Updated to CUDA 12.4.1 with cuDNN to match PyTorch version
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Build argument: models are loaded from Network Volume at runtime
# Set to "false" to embed models in Docker image (hybrid/docker-embedded strategy)
ARG SKIP_MODEL_DOWNLOAD=true

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
ENV RUNPOD_VOLUME_PATH=/runpod-volume

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Configure git for non-interactive operation
RUN git config --global credential.helper store && \
    git config --global http.postBuffer 524288000

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app/

# Install base Python dependencies (torch 2.6 required by transformers 4.56+ for CVE-2025-32434)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install common dependencies
RUN pip install --no-cache-dir \
    numpy \
    soundfile \
    librosa \
    pydub \
    transformers==4.56.2 \
    diffusers \
    accelerate \
    bitsandbytes \
    scipy \
    tqdm \
    einops \
    huggingface-hub \
    runpod==1.7.5 \
    requests

# Install TTS dependencies
WORKDIR /app/TTS

# Clone and install SparkTTS (extract only necessary folders)
RUN echo "=================================" && \
    echo "Installing SparkTTS..." && \
    echo "=================================" && \
    git clone --depth 1 https://github.com/SparkAudio/Spark-TTS.git /tmp/SparkTTS && \
    if [ -d "/tmp/SparkTTS" ]; then \
        echo "SparkTTS cloned successfully" && \
        cd /tmp/SparkTTS && \
        pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo "Warning: SparkTTS requirements.txt not found or failed" && \
        pip install --no-cache-dir -e . 2>/dev/null || echo "Warning: SparkTTS setup.py installation failed" && \
        echo "Extracting SparkTTS essential folders to /app/TTS..." && \
        mkdir -p /app/TTS/sparktts /app/TTS/src /app/TTS/cli /app/TTS/runtime && \
        [ -d "/tmp/SparkTTS/sparktts" ] && cp -r /tmp/SparkTTS/sparktts /app/TTS/ || echo "Warning: sparktts folder not found" && \
        [ -d "/tmp/SparkTTS/src" ] && cp -r /tmp/SparkTTS/src /app/TTS/ || echo "Warning: src folder not found" && \
        [ -d "/tmp/SparkTTS/cli" ] && cp -r /tmp/SparkTTS/cli /app/TTS/ || echo "Warning: cli folder not found" && \
        [ -d "/tmp/SparkTTS/runtime" ] && cp -r /tmp/SparkTTS/runtime /app/TTS/ || echo "Warning: runtime folder not found" && \
        cd /app/TTS && \
        rm -rf /tmp/SparkTTS && \
        echo "SparkTTS essential folders extracted successfully"; \
    else \
        echo "Warning: SparkTTS clone failed, skipping..."; \
    fi

# Reinstall torch/torchaudio/transformers to fix version conflicts from SparkTTS requirements
# SparkTTS requirements.txt may change torchaudio version, causing libtorchaudio.so symbol mismatch
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir --upgrade transformers==4.56.2

# Create vibevoice directory for model downloads
RUN mkdir -p /app/TTS/vibevoice

# Install additional TTS dependencies
RUN pip install --no-cache-dir \
    omegaconf \
    hydra-core \
    pyworld \
    praat-parselmouth \
    phonemizer \
    g2p-en \
    unidecode \
    inflect

# Model directories on Network Volume (downloaded at runtime by start.sh)
# Docker build does NOT download models - they are loaded from /runpod-volume/models/ at runtime
RUN mkdir -p pretrained_models/Spark-TTS-0.5B

# Download models only if SKIP_MODEL_DOWNLOAD=false (docker-embedded strategy)
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
        echo "=================================" && \
        echo "Downloading Spark-TTS-0.5B model..." && \
        echo "=================================" && \
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SparkAudio/Spark-TTS-0.5B', local_dir='/runpod-volume/models/Spark-TTS-0.5B', local_dir_use_symlinks=False)" && \
        echo "Spark-TTS model download completed"; \
    else \
        echo "Skipping Spark-TTS-0.5B download (will be loaded from Network Volume at runtime)"; \
    fi

RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
        echo "=================================" && \
        echo "Downloading VibeVoice-7B model..." && \
        echo "=================================" && \
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vibevoice/VibeVoice-7B', local_dir='/runpod-volume/models/VibeVoice-7B', local_dir_use_symlinks=False)" && \
        echo "VibeVoice model download completed"; \
    else \
        echo "Skipping VibeVoice-7B download (will be loaded from Network Volume at runtime)"; \
    fi

# Install RVC dependencies
WORKDIR /app/RVC

# Install RVC Python packages (excluding fairseq and problematic dependencies)
RUN pip install --no-cache-dir \
    pydub \
    edge-tts \
    audio-separator \
    librosa \
    praat-parselmouth \
    pyworld \
    scipy \
    numba \
    resampy \
    faiss-gpu \
    psutil \
    py-cpuinfo

# Install fairseq from source (PyPI version has build issues)
RUN echo "Installing fairseq from source..." && \
    git clone --depth 1 --branch v0.12.2 https://github.com/pytorch/fairseq.git /tmp/fairseq && \
    cd /tmp/fairseq && \
    pip install --no-cache-dir --editable ./ || \
    (echo "Warning: fairseq installation failed, continuing..." && exit 0) && \
    cd /app/RVC && \
    rm -rf /tmp/fairseq

# Install infer-rvc-python with --no-deps to avoid fairseq/omegaconf dependency conflict
# (pip 24.1+ rejects omegaconf<2.1 due to invalid metadata, but fairseq is already installed from source)
RUN pip install --no-cache-dir --no-deps infer-rvc-python && \
    pip install --no-cache-dir torchcrepe==0.0.20 ffmpeg-python typeguard==4.2.0 || \
    (echo "Warning: infer-rvc-python installation failed, continuing..." && exit 0)

# Download RVC model files to Network Volume path
RUN mkdir -p /runpod-volume/models/RVC && \
    echo "Downloading RMVPE model..." && \
    wget -q --show-progress https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/download/20230428/rmvpe.pt \
        -O /runpod-volume/models/RVC/rmvpe.pt || echo "Warning: Failed to download rmvpe.pt"

RUN echo "Downloading Hubert Base model..." && \
    wget -q --show-progress https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
        -O /runpod-volume/models/RVC/hubert_base.pt || echo "Warning: Failed to download hubert_base.pt"

# Create placeholder files if they don't exist
RUN if [ ! -f "tts_voice.py" ]; then \
        echo "# TTS voice order mapping" > tts_voice.py && \
        echo "tts_order_voice = {" >> tts_voice.py && \
        echo "    'en': 'en-US'," >> tts_voice.py && \
        echo "    'ko': 'ko-KR'," >> tts_voice.py && \
        echo "    'ja': 'ja-JP'," >> tts_voice.py && \
        echo "    'zh': 'zh-CN'," >> tts_voice.py && \
        echo "}" >> tts_voice.py; \
    fi

RUN if [ ! -f "model_handler.py" ]; then \
        echo "# Model handler placeholder" > model_handler.py && \
        echo "class ModelHandler:" >> model_handler.py && \
        echo "    def __init__(self):" >> model_handler.py && \
        echo "        pass" >> model_handler.py && \
        echo "    def load_model(self, model_path):" >> model_handler.py && \
        echo "        pass" >> model_handler.py; \
    fi

# Create RVC weights directory on Network Volume path
RUN mkdir -p /runpod-volume/models/RVC/weights

# Create output directories
RUN mkdir -p /app/example/results

# Set the working directory back to app root
WORKDIR /app

# Verify installations
RUN echo "==================================" && \
    echo "Installation Summary:" && \
    echo "==================================" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" && \
    (python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || echo "Warning: transformers check failed") && \
    echo "TTS modules:" && \
    ls -la /app/TTS/ && \
    echo "RVC modules:" && \
    ls -la /app/RVC/ && \
    echo "=================================="

# Make entrypoint script executable
RUN chmod +x /app/start.sh

# Default command - Download models to Network Volume, then start handler
CMD ["/app/start.sh"]
