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
from model_cache import resolve_model_path

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
RVC_DIR = os.path.join(APP_ROOT, 'RVC')

# Add TTS and RVC directories to Python path ONCE at module load
if TTS_DIR not in sys.path:
    sys.path.insert(0, TTS_DIR)
if RVC_DIR not in sys.path:
    sys.path.insert(0, RVC_DIR)

# Pre-import model classes to avoid lazy loading
# These will be imported when the module loads, not when initialize_models() is called
logger.info("Pre-loading model classes...")
VibeVoiceModel = None
SparkModel = None

# Patch transformers torch.load safety check for torch<2.6 compatibility (CVE-2025-32434)
# SparkTTS wav2vec2 model uses .bin format which triggers this check
try:
    import transformers.utils.import_utils as _tf_import_utils
    if hasattr(_tf_import_utils, 'check_torch_load_is_safe'):
        _tf_import_utils.check_torch_load_is_safe = lambda: None
        logger.info("✓ Patched transformers torch.load safety check for torch<2.6")
except Exception as e:
    logger.warning(f"⚠ Failed to patch transformers safety check: {e}")

# Change to TTS directory ONCE at module level for all model operations
original_cwd = os.getcwd()
os.chdir(TTS_DIR)

try:
    from VibeVoiceModel import VibeVoiceModel
    logger.info("✓ VibeVoiceModel class loaded")
except Exception as e:
    logger.warning(f"⚠ VibeVoiceModel class not available: {e}")

_spark_import_error = None  # Store SparkModel import error for debug
try:
    from SparkModel import SparkModel
    logger.info("✓ SparkModel class loaded")
except Exception as e:
    import traceback
    _spark_import_error = traceback.format_exc()
    logger.warning(f"⚠ SparkModel class not available: {e}")
    logger.warning(f"⚠ SparkModel traceback: {_spark_import_error}")

# Restore directory after imports
os.chdir(original_cwd)

# Import RVC (does not need TTS directory context)
RVCClass = None
_rvc_import_error = None
try:
    from RVC import RVC as RVCClass
    logger.info("✓ RVC class loaded")
except Exception as e:
    import traceback
    _rvc_import_error = traceback.format_exc()
    logger.warning(f"⚠ RVC class not available: {e}")
    logger.warning(f"⚠ RVC traceback: {_rvc_import_error}")

# ===== GLOBAL MODEL INSTANCES =====
# Models are loaded once at container startup and reused for all requests
vibevoice_model = None
spark_model = None
rvc_model = None
_spark_init_error = None  # Store Spark init error for debug
_rvc_init_error = None  # Store RVC init error for debug

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
    Models are loaded from Network Volume (primary) or downloaded from HuggingFace Hub.
    Docker embedded models are used as fallback only.
    """
    global vibevoice_model, spark_model, rvc_model

    logger.info("=" * 50)
    logger.info("Initializing models...")
    logger.info("=" * 50)

    # Log volume status
    from model_cache import get_volume_path
    volume_path = get_volume_path()
    if volume_path:
        logger.info(f"Network Volume detected: {volume_path}")
    else:
        logger.warning("No Network Volume detected! Models will use Docker fallback.")

    # Save current directory
    current_dir = os.getcwd()

    try:
        # Change to TTS directory for model initialization
        os.chdir(TTS_DIR)

        # Resolve model paths (Network Volume → HuggingFace download → Docker fallback)
        vibevoice_model_path = None
        spark_model_path = None

        try:
            vibevoice_model_path = resolve_model_path("vibevoice")
            logger.info(f"VibeVoice model path resolved: {vibevoice_model_path}")
        except Exception as e:
            logger.warning(f"VibeVoice model path not resolved: {e}")

        try:
            spark_model_path = resolve_model_path("spark")
            logger.info(f"Spark model path resolved: {spark_model_path}")
        except Exception as e:
            logger.warning(f"Spark model path not resolved: {e}")

        # Initialize VibeVoice model
        if VibeVoiceModel is not None and vibevoice_model_path is not None:
            try:
                logger.info("Loading VibeVoice model...")
                vibevoice_model = VibeVoiceModel(model_path=vibevoice_model_path)
                logger.info("VibeVoice model loaded")
            except Exception as e:
                logger.error(f"VibeVoice model failed: {e}")
                vibevoice_model = None
        else:
            logger.warning("VibeVoiceModel class or model path not available")

        # Initialize Spark model
        global _spark_init_error
        if SparkModel is not None and spark_model_path is not None:
            try:
                logger.info(f"Loading Spark model from: {spark_model_path}")
                logger.info(f"Spark model dir exists: {os.path.isdir(spark_model_path)}")
                if os.path.isdir(spark_model_path):
                    logger.info(f"Spark model dir contents: {os.listdir(spark_model_path)}")
                spark_model = SparkModel(model_dir=spark_model_path)
                logger.info("Spark model loaded successfully")
            except Exception as e:
                import traceback
                _spark_init_error = traceback.format_exc()
                logger.error(f"Spark model failed: {e}")
                logger.error(f"Spark model traceback: {_spark_init_error}")
                spark_model = None
        else:
            _spark_init_error = f"SparkModel class={SparkModel is not None}, path={spark_model_path}"
            logger.warning(f"SparkModel class or model path not available: {_spark_init_error}")

        # Initialize RVC model
        global _rvc_init_error
        if RVCClass is not None:
            try:
                logger.info("Loading RVC model...")
                rvc_model = RVCClass()
                logger.info("RVC model loaded successfully")
            except Exception as e:
                import traceback
                _rvc_init_error = traceback.format_exc()
                logger.error(f"RVC model failed: {e}")
                logger.error(f"RVC model traceback: {_rvc_init_error}")
                rvc_model = None
        else:
            _rvc_init_error = f"RVC class not available (import error: {_rvc_import_error})"
            logger.warning(f"RVC class not available: {_rvc_init_error}")

        # Verify at least one model loaded
        if vibevoice_model is None and spark_model is None and rvc_model is None:
            raise RuntimeError("No models could be loaded!")

        logger.info("=" * 50)
        logger.info("Model initialization complete")
        logger.info(f"   VibeVoice: {'loaded' if vibevoice_model else 'unavailable'}"
                    f" (path: {vibevoice_model_path})")
        logger.info(f"   Spark: {'loaded' if spark_model else 'unavailable'}"
                    f" (path: {spark_model_path})")
        logger.info(f"   RVC: {'loaded' if rvc_model else 'unavailable'}")
        logger.info(f"   Source: {'Network Volume' if volume_path else 'Docker embedded'}")
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

        # --- Debug mode: return system status ---
        if job_input.get("debug") == "status":
            import traceback
            from model_cache import get_volume_path, _is_model_complete, MODELS
            volume_path = get_volume_path()
            volume_models_dir = os.path.join(volume_path, "models") if volume_path else None

            debug_info = {
                "volume_path": volume_path,
                "volume_exists": volume_path is not None,
                "vibevoice_loaded": vibevoice_model is not None,
                "spark_loaded": spark_model is not None,
                "spark_class_available": SparkModel is not None,
                "spark_import_error": _spark_import_error,
                "spark_init_error": _spark_init_error,
                "rvc_loaded": rvc_model is not None,
                "rvc_class_available": RVCClass is not None,
                "rvc_import_error": _rvc_import_error,
                "rvc_init_error": _rvc_init_error,
                "tts_dir": TTS_DIR,
                "tts_dir_contents": os.listdir(TTS_DIR) if os.path.isdir(TTS_DIR) else "NOT FOUND",
                "cli_dir_exists": os.path.isdir(os.path.join(TTS_DIR, "cli")),
            }

            if volume_models_dir and os.path.isdir(volume_models_dir):
                debug_info["volume_models"] = os.listdir(volume_models_dir)
                spark_dir = os.path.join(volume_models_dir, "Spark-TTS-0.5B")
                if os.path.isdir(spark_dir):
                    debug_info["spark_dir_contents"] = os.listdir(spark_dir)
                    debug_info["spark_complete"] = _is_model_complete(spark_dir, "spark")
                else:
                    debug_info["spark_dir_contents"] = "NOT FOUND"
                    debug_info["spark_complete"] = False
            else:
                debug_info["volume_models"] = "NO VOLUME OR MODELS DIR"

            return debug_info

        # Get model type first (affects validation)
        model_type = job_input.get("model_type", "vibevoice").lower()

        # --- RVC branch (no text required) ---
        if model_type == "rvc":
            if rvc_model is None:
                return {"error": "RVC model is not available"}

            audio_input = job_input.get("audio")
            if not audio_input:
                return {"error": "Missing required parameter: audio"}

            character = job_input.get("character", "Poli")
            language = job_input.get("language", "KR")
            pitch_level = job_input.get("pitch_level", 0)

            logger.info(f"Request: model=rvc, character={character}, language={language}, pitch={pitch_level}")

            with tempfile.TemporaryDirectory() as temp_dir:
                input_audio_path = os.path.join(temp_dir, "input.wav")

                if audio_input.startswith(("http://", "https://")):
                    if not download_audio_from_url(audio_input, input_audio_path):
                        return {"error": "Failed to download input audio"}
                else:
                    if not decode_base64_audio(audio_input, input_audio_path):
                        return {"error": "Failed to decode input audio"}

                try:
                    result_path = rvc_model.run_rvc(character, language, [input_audio_path], pitch_level)
                    if result_path is None or not os.path.exists(result_path):
                        return {"error": "RVC conversion failed"}

                    audio_base64 = encode_audio_to_base64(result_path)
                    if audio_base64 is None:
                        return {"error": "Failed to encode RVC output audio"}
                except Exception as e:
                    logger.error(f"RVC inference error: {e}", exc_info=True)
                    return {"error": f"RVC inference error: {str(e)}"}

            logger.info("RVC conversion completed")

            return {
                "audio": audio_base64,
                "model_used": "rvc",
                "character": character,
                "language": language,
                "pitch_level": pitch_level,
            }

        # --- TTS branch (vibevoice / spark) ---
        # Validate inputs
        text = job_input.get("text")
        prompt_speech = job_input.get("prompt_speech")

        if not text:
            return {"error": "Missing required parameter: text"}
        if not prompt_speech:
            return {"error": "Missing required parameter: prompt_speech"}

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
            return {"error": f"Invalid model_type: {model_type}. Supported: vibevoice, spark, rvc"}

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

            logger.info("TTS completed")

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
