"""
Optimized RunPod Serverless Handler for TTS Generation

This version minimizes cold starts by:
1. Pre-loading all imports at module level
2. Initializing models once at startup
3. Minimizing directory changes
4. Optimizing RunPod serverless configuration
5. Reducing unnecessary logging
"""

import runpod
import torch
import os
import sys
import base64
import tempfile
import soundfile as sf
import numpy as np
import logging
import requests
from typing import Optional, Tuple, Dict, Any

# ===== STARTUP OPTIMIZATION =====
# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get application root directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = os.path.join(APP_ROOT, 'TTS')

# Add TTS directory to Python path ONCE at module load
if TTS_DIR not in sys.path:
    sys.path.insert(0, TTS_DIR)

# Pre-import model classes to avoid lazy loading
# These will be imported when the module loads, not when initialize_models() is called
logger.info("Pre-loading model classes...")
VibeVoiceModel = None
SparkModel = None

# Change to TTS directory ONCE at module level for all model operations
original_cwd = os.getcwd()
os.chdir(TTS_DIR)

try:
    from VibeVoiceModel import VibeVoiceModel
    logger.info("✓ VibeVoiceModel class loaded")
except Exception as e:
    logger.warning(f"⚠ VibeVoiceModel class not available: {e}")

try:
    from SparkModel import SparkModel
    logger.info("✓ SparkModel class loaded")
except Exception as e:
    logger.warning(f"⚠ SparkModel class not available: {e}")

# Restore directory after imports
os.chdir(original_cwd)

# ===== GLOBAL MODEL INSTANCES =====
# Models are loaded once at container startup and reused for all requests
vibevoice_model = None
spark_model = None

# ===== HELPER FUNCTIONS =====

def download_audio_from_url(url: str, output_path: str) -> bool:
    """Download audio file from URL with timeout."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return False


def decode_base64_audio(base64_str: str, output_path: str) -> bool:
    """Decode base64 audio string and save to file."""
    try:
        audio_data = base64.b64decode(base64_str)
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        return True
    except Exception as e:
        logger.error(f"Failed to decode base64 audio: {e}")
        return False


def encode_audio_to_base64(audio_path: str) -> Optional[str]:
    """Encode audio file to base64 string."""
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode audio: {e}")
        return None


def encode_numpy_audio_to_base64(audio_array: Any, sample_rate: int) -> Optional[str]:
    """Encode numpy/torch audio array to base64 string."""
    try:
        # Convert torch tensor to numpy if needed
        if hasattr(audio_array, 'cpu'):
            audio_array = audio_array.cpu().numpy()

        # Ensure numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)

        # Ensure float32 dtype for soundfile
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Ensure 1D array
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        sf.write(tmp_path, audio_array, sample_rate, subtype='PCM_16')

        # Read and encode
        with open(tmp_path, 'rb') as f:
            audio_data = f.read()

        # Clean up
        os.unlink(tmp_path)

        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode audio: {e}")
        return None


# ===== MODEL INITIALIZATION =====

def initialize_models() -> None:
    """
    Initialize TTS models at startup.
    This function is called ONCE when the container starts.
    """
    global vibevoice_model, spark_model

    logger.info("=" * 50)
    logger.info("Initializing TTS models...")
    logger.info("=" * 50)

    # Save current directory
    current_dir = os.getcwd()

    try:
        # Change to TTS directory for model initialization
        os.chdir(TTS_DIR)

        # Initialize VibeVoice model
        if VibeVoiceModel is not None:
            try:
                logger.info("Loading VibeVoice model...")
                vibevoice_model = VibeVoiceModel()
                logger.info("✅ VibeVoice model loaded")
            except Exception as e:
                logger.error(f"❌ VibeVoice model failed: {e}")
                vibevoice_model = None
        else:
            logger.warning("⚠ VibeVoiceModel class not available")

        # Initialize Spark model
        if SparkModel is not None:
            try:
                model_dir = "pretrained_models/Spark-TTS-0.5B"
                if os.path.exists(model_dir):
                    logger.info("Loading Spark model...")
                    spark_model = SparkModel(model_dir=model_dir)
                    logger.info("✅ Spark model loaded")
                else:
                    logger.warning(f"⚠ Spark model directory not found: {model_dir}")
                    spark_model = None
            except Exception as e:
                logger.error(f"❌ Spark model failed: {e}")
                spark_model = None
        else:
            logger.warning("⚠ SparkModel class not available")

        # Verify at least one model loaded
        if vibevoice_model is None and spark_model is None:
            raise RuntimeError("❌ No TTS models could be loaded!")

        logger.info("=" * 50)
        logger.info("🚀 Model initialization complete")
        logger.info(f"   VibeVoice: {'✅' if vibevoice_model else '❌'}")
        logger.info(f"   Spark: {'✅' if spark_model else '❌'}")
        logger.info("=" * 50)

    finally:
        # Restore directory
        os.chdir(current_dir)


# ===== REQUEST HANDLER =====

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for TTS generation.

    This function is called for EACH request.
    Models are already loaded, so this should be fast.

    Input:
    {
        "input": {
            "text": str,
            "prompt_speech": str (base64 or URL),
            "model_type": str ("vibevoice" or "spark"),
            "cfg_scale": float (optional, for VibeVoice)
        }
    }

    Output:
    {
        "audio": str (base64),
        "sample_rate": int,
        "model_used": str,
        "text_length": int
    }
    """
    try:
        job_input = event.get("input", {})

        # Validate inputs
        text = job_input.get("text")
        prompt_speech = job_input.get("prompt_speech")

        if not text:
            return {"error": "Missing required parameter: text"}
        if not prompt_speech:
            return {"error": "Missing required parameter: prompt_speech"}

        # Get parameters
        model_type = job_input.get("model_type", "vibevoice").lower()
        cfg_scale = job_input.get("cfg_scale", 2.0)

        logger.info(f"Request: model={model_type}, text_len={len(text)}")

        # Select model
        if model_type == "vibevoice":
            if vibevoice_model is None:
                return {"error": "VibeVoice model is not available"}
            selected_model = vibevoice_model
        elif model_type == "spark":
            if spark_model is None:
                return {"error": "Spark model is not available"}
            selected_model = spark_model
        else:
            return {"error": f"Invalid model_type: {model_type}"}

        # Process with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle prompt audio
            prompt_audio_path = os.path.join(temp_dir, "prompt.wav")

            if prompt_speech.startswith(("http://", "https://")):
                if not download_audio_from_url(prompt_speech, prompt_audio_path):
                    return {"error": "Failed to download prompt audio"}
            else:
                if not decode_base64_audio(prompt_speech, prompt_audio_path):
                    return {"error": "Failed to decode prompt audio"}

            # Run TTS (directory change needed for model inference)
            current_dir = os.getcwd()
            try:
                os.chdir(TTS_DIR)

                if model_type == "vibevoice":
                    result = selected_model.run_tts(
                        text=text,
                        prompt_speech=prompt_audio_path,
                        cfg_scale=cfg_scale
                    )
                    if result is None:
                        return {"error": "TTS generation failed"}

                    audio_output, sample_rate = result
                    audio_base64 = encode_numpy_audio_to_base64(audio_output, sample_rate)

                elif model_type == "spark":
                    output_audio_path = os.path.join(temp_dir, "output.wav")
                    result = selected_model.run_tts(
                        text=text,
                        prompt_speech=prompt_audio_path,
                        output_path=output_audio_path
                    )
                    if result is None:
                        return {"error": "TTS generation failed"}

                    _, sample_rate = result
                    audio_base64 = encode_audio_to_base64(output_audio_path)

                if audio_base64 is None:
                    return {"error": "Failed to encode audio output"}

            finally:
                os.chdir(current_dir)

            logger.info("✅ TTS completed")

            return {
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "model_used": model_type,
                "text_length": len(text)
            }

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}


# ===== STARTUP =====

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RunPod Serverless TTS Handler (Optimized)")
    logger.info("=" * 60)

    # Initialize models once at container startup
    initialize_models()

    # Start RunPod serverless worker with optimized configuration
    logger.info("🚀 Starting RunPod serverless worker...")

    runpod.serverless.start(
        {
            "handler": handler,
            # Increase concurrency for better performance
            "concurrency_modifier": lambda current_concurrency: current_concurrency + 1,
            # Return immediately after processing
            "return_aggregate_stream": False,
        }
    )
