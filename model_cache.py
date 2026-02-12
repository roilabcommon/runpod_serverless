"""
Model Cache Manager for RunPod Network Volume

Models are stored on Network Volume and shared across all workers.
On first worker start, models are automatically downloaded from HuggingFace Hub.

Resolves model paths with priority:
1. Network Volume (primary - fast, shared across workers)
2. Download to Network Volume from HuggingFace Hub (first use)
3. Embedded in Docker image (fallback, only if SKIP_MODEL_DOWNLOAD=false was used)

Environment Variables:
    RUNPOD_VOLUME_PATH: Mount point of RunPod Network Volume (default: /runpod-volume)
"""

import os
import time
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_VOLUME_PATH = "/runpod-volume"
DOCKER_TTS_DIR = "/app/TTS"

# Model definitions
MODELS = {
    "spark": {
        "repo_id": "SparkAudio/Spark-TTS-0.5B",
        "docker_relative_path": "pretrained_models/Spark-TTS-0.5B",
        "volume_subdir": "models/Spark-TTS-0.5B",
    },
    "vibevoice": {
        "repo_id": "vibevoice/VibeVoice-7B",
        "docker_relative_path": "vibevoice/VibeVoice-7B",
        "volume_subdir": "models/VibeVoice-7B",
    },
}


def get_volume_path():
    """Get the Network Volume mount path, or None if not available."""
    volume_path = os.environ.get("RUNPOD_VOLUME_PATH", DEFAULT_VOLUME_PATH)

    if not os.path.isdir(volume_path):
        return None

    # Verify it's writable
    try:
        test_file = os.path.join(volume_path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return volume_path
    except OSError:
        logger.warning(f"Volume path {volume_path} exists but is not writable")
        return None


def _is_model_complete(model_dir, model_key):
    """Check if a model directory contains a complete model download."""
    if not os.path.isdir(model_dir):
        return False

    if model_key == "vibevoice":
        # Check for config.json + safetensors or bin weight files
        config_file = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file):
            return False
        has_weights = any(
            f.endswith((".safetensors", ".bin"))
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f))
        )
        return has_weights

    if model_key == "spark":
        # Spark-TTS uses config.yaml (not config.json) + BiCodec directory
        config_yaml = os.path.join(model_dir, "config.yaml")
        bicodec_dir = os.path.join(model_dir, "BiCodec")
        return os.path.exists(config_yaml) and os.path.isdir(bicodec_dir)

    return True


def _download_model_to_volume(model_key, target_dir):
    """Download a model from HuggingFace Hub to the target directory."""
    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]

    logger.info(f"Downloading {repo_id} to {target_dir}...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Successfully downloaded {repo_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        return False


def _create_lock_file(lock_path):
    """Attempt to create a lock file for download coordination."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _is_lock_stale(lock_path, max_age_seconds=1800):
    """Check if a lock file is stale (older than max_age_seconds)."""
    try:
        lock_age = time.time() - os.path.getmtime(lock_path)
        return lock_age > max_age_seconds
    except OSError:
        return True


def _wait_for_download(lock_path, model_dir, model_key, timeout=1800):
    """Wait for another worker to finish downloading."""
    start = time.time()
    while time.time() - start < timeout:
        # If lock is gone and model is complete, we're done
        if not os.path.exists(lock_path) and _is_model_complete(model_dir, model_key):
            return True

        # If lock is stale, break out and take over
        if os.path.exists(lock_path) and _is_lock_stale(lock_path):
            logger.warning("Download lock is stale. Taking over download...")
            try:
                os.remove(lock_path)
            except OSError:
                pass
            return False

        elapsed = int(time.time() - start)
        logger.info(f"Waiting for another worker to finish downloading... ({elapsed}s)")
        time.sleep(10)

    logger.error(f"Timed out waiting for model download after {timeout}s")
    return False


def resolve_model_path(model_key):
    """
    Resolve the model path with priority:
    1. Network Volume (already downloaded)
    2. Docker embedded path (fallback)
    3. Download to Network Volume then use that path

    Args:
        model_key: "spark" or "vibevoice"

    Returns:
        Absolute path to the model directory.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Must be one of {list(MODELS.keys())}")

    model_info = MODELS[model_key]
    docker_path = os.path.join(DOCKER_TTS_DIR, model_info["docker_relative_path"])

    volume_path = get_volume_path()

    # --- No Network Volume: use Docker embedded models as fallback ---
    if volume_path is None:
        logger.warning(f"[{model_key}] No network volume detected! "
                       f"Checked: {os.environ.get('RUNPOD_VOLUME_PATH', DEFAULT_VOLUME_PATH)}")
        logger.info(f"[{model_key}] Falling back to Docker embedded path: {docker_path}")
        if os.path.isdir(docker_path) and _is_model_complete(docker_path, model_key):
            return docker_path
        else:
            raise FileNotFoundError(
                f"No network volume mounted and Docker model not found at {docker_path}. "
                f"Please mount Network Volume 'roi_ai_studio' to /runpod-volume, "
                f"or rebuild Docker image with SKIP_MODEL_DOWNLOAD=false."
            )

    # --- Network Volume available ---
    volume_model_dir = os.path.join(volume_path, model_info["volume_subdir"])

    # Check if model already exists on volume
    if _is_model_complete(volume_model_dir, model_key):
        logger.info(f"[{model_key}] Model found on network volume: {volume_model_dir}")
        return volume_model_dir

    # Model not on volume -- try to download with file locking
    lock_path = volume_model_dir + ".download.lock"

    if _create_lock_file(lock_path):
        # We acquired the lock -- do the download
        try:
            os.makedirs(volume_model_dir, exist_ok=True)

            # Prefer copying from Docker image if available
            if os.path.isdir(docker_path) and _is_model_complete(docker_path, model_key):
                logger.info(f"[{model_key}] Copying from Docker image to volume...")
                shutil.copytree(docker_path, volume_model_dir, dirs_exist_ok=True)
                logger.info(f"[{model_key}] Copy complete")
            else:
                # Download from HuggingFace Hub
                logger.info(f"[{model_key}] Downloading from HuggingFace Hub...")
                if not _download_model_to_volume(model_key, volume_model_dir):
                    raise RuntimeError(f"Failed to download {model_key} model")

            if _is_model_complete(volume_model_dir, model_key):
                logger.info(f"[{model_key}] Model ready on volume: {volume_model_dir}")
                return volume_model_dir
            else:
                raise RuntimeError(f"Model download succeeded but validation failed for {model_key}")
        finally:
            # Release lock
            try:
                os.remove(lock_path)
            except OSError:
                pass
    else:
        # Another worker is downloading -- wait
        logger.info(f"[{model_key}] Another worker is downloading. Waiting...")
        if _wait_for_download(lock_path, volume_model_dir, model_key):
            return volume_model_dir

        # Stale lock was removed -- try again (recursive, max 1 retry)
        if not os.path.exists(lock_path):
            logger.info(f"[{model_key}] Retrying after stale lock removal...")
            return resolve_model_path(model_key)

        # Fall back to Docker path
        if os.path.isdir(docker_path) and _is_model_complete(docker_path, model_key):
            logger.warning(f"[{model_key}] Falling back to Docker path: {docker_path}")
            return docker_path

        raise RuntimeError(f"Could not resolve model path for {model_key}")
