import runpod
import torch
import os
import sys
import base64
import tempfile
import soundfile as sf
import numpy as np
import logging
from io import BytesIO
import requests
from model_cache import resolve_model_path

# Get the application root directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add TTS directory to path
TTS_DIR = os.path.join(APP_ROOT, 'TTS')
sys.path.append(TTS_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances (loaded once at startup)
vibevoice_model = None
spark_model = None


def download_audio_from_url(url: str, output_path: str) -> bool:
    """Download audio file from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download audio from URL: {e}")
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


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode audio file to base64 string."""
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode audio to base64: {e}")
        return None


def encode_numpy_audio_to_base64(audio_array, sample_rate: int) -> str:
    """Encode numpy/torch audio array to base64 string."""
    try:
        # Convert torch tensor to numpy if needed
        if hasattr(audio_array, 'cpu'):
            logger.info("Converting torch tensor to numpy array...")
            audio_array = audio_array.cpu().numpy()

        # Ensure numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)

        # Ensure float32 dtype for soundfile
        if audio_array.dtype != np.float32:
            logger.info(f"Converting audio from {audio_array.dtype} to float32...")
            audio_array = audio_array.astype(np.float32)

        # Ensure 1D array
        if len(audio_array.shape) > 1:
            logger.info(f"Reshaping audio from {audio_array.shape} to 1D...")
            audio_array = audio_array.squeeze()

        logger.info(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}, sample_rate: {sample_rate}")

        # Create a temporary file to write audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, mode='wb') as tmp_file:
            tmp_path = tmp_file.name

        # Write audio to file
        sf.write(tmp_path, audio_array, sample_rate, subtype='PCM_16')
        logger.info(f"Audio written to temporary file: {tmp_path}")

        # Read and encode
        with open(tmp_path, 'rb') as f:
            audio_data = f.read()

        logger.info(f"Audio file size: {len(audio_data)} bytes")

        # Clean up
        os.unlink(tmp_path)

        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode audio to base64: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_models():
    """Initialize TTS models at startup."""
    global vibevoice_model, spark_model

    logger.info("Initializing TTS models...")

    # Save current directory
    current_dir = os.getcwd()

    try:
        # Change to TTS directory for model loading
        os.chdir(TTS_DIR)
        logger.info(f"Changed working directory to: {os.getcwd()}")

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

        # Initialize VibeVoice model if available
        if vibevoice_model_path is not None:
            try:
                logger.info("Loading VibeVoice model...")
                from VibeVoiceModel import VibeVoiceModel
                vibevoice_model = VibeVoiceModel(model_path=vibevoice_model_path)
                logger.info("✅ VibeVoice model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load VibeVoice model: {e}")
                import traceback
                traceback.print_exc()
                vibevoice_model = None
        else:
            logger.warning("⚠️ VibeVoice model path not available")
            vibevoice_model = None

        # Initialize Spark model if available
        if spark_model_path is not None:
            try:
                logger.info("Loading Spark model...")
                from SparkModel import SparkModel
                spark_model = SparkModel(model_dir=spark_model_path)
                logger.info("✅ Spark model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Spark model: {e}")
                spark_model = None
        else:
            logger.warning("⚠️ Spark model path not available")
            spark_model = None

        if vibevoice_model is None and spark_model is None:
            raise RuntimeError("❌ No TTS models could be loaded!")

        logger.info("🚀 Model initialization complete")

    finally:
        # Always return to original directory
        os.chdir(current_dir)
        logger.info(f"Restored working directory to: {os.getcwd()}")


def handler(event):
    """
    RunPod serverless handler for TTS generation.

    Expected input format:
    {
        "input": {
            "text": "Text to synthesize",
            "prompt_speech": "base64_encoded_audio" or "http://url/to/audio.wav",
            "model_type": "vibevoice" or "spark",  # default: "vibevoice"
            "cfg_scale": 2.0,  # optional, for VibeVoice only
            "return_format": "base64" or "url"  # default: "base64"
        }
    }

    Returns:
    {
        "audio": "base64_encoded_audio",
        "sample_rate": 24000,
        "model_used": "vibevoice"
    }
    """
    try:
        job_input = event.get("input", {})

        # Validate required inputs
        text = job_input.get("text")
        prompt_speech = job_input.get("prompt_speech")

        if not text:
            return {"error": "Missing required parameter: text"}

        if not prompt_speech:
            return {"error": "Missing required parameter: prompt_speech"}

        # Get optional parameters
        model_type = job_input.get("model_type", "vibevoice").lower()
        cfg_scale = job_input.get("cfg_scale", 2.0)
        return_format = job_input.get("return_format", "base64").lower()

        logger.info(f"Processing TTS request with model: {model_type}")
        logger.info(f"Text length: {len(text)} characters")

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
            return {"error": f"Invalid model_type: {model_type}. Use 'vibevoice' or 'spark'"}

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle prompt_speech input (base64 or URL)
            prompt_audio_path = os.path.join(temp_dir, "prompt_speech.wav")

            if prompt_speech.startswith("http://") or prompt_speech.startswith("https://"):
                logger.info("Downloading prompt speech from URL...")
                if not download_audio_from_url(prompt_speech, prompt_audio_path):
                    return {"error": "Failed to download prompt speech from URL"}
            else:
                logger.info("Decoding prompt speech from base64...")
                if not decode_base64_audio(prompt_speech, prompt_audio_path):
                    return {"error": "Failed to decode prompt speech from base64"}

            logger.info(f"Prompt speech saved to: {prompt_audio_path}")

            # Run TTS generation
            logger.info("Running TTS generation...")

            # Save current directory
            current_dir = os.getcwd()

            try:
                # Change to TTS directory for model inference
                os.chdir(TTS_DIR)

                if model_type == "vibevoice":
                    # VibeVoice returns (numpy_array, sample_rate)
                    result = selected_model.run_tts(
                        text=text,
                        prompt_speech=prompt_audio_path,
                        cfg_scale=cfg_scale
                    )

                    if result is None:
                        return {"error": "TTS generation failed"}

                    audio_output, sample_rate = result

                    # Encode to base64
                    logger.info("Encoding audio to base64...")
                    audio_base64 = encode_numpy_audio_to_base64(audio_output, sample_rate)

                    if audio_base64 is None:
                        return {"error": "Failed to encode audio output"}

                elif model_type == "spark":
                    # Spark saves to file and returns (file_path, sample_rate)
                    output_audio_path = os.path.join(temp_dir, "output.wav")

                    result = selected_model.run_tts(
                        text=text,
                        prompt_speech=prompt_audio_path,
                        output_path=output_audio_path
                    )

                    if result is None:
                        return {"error": "TTS generation failed"}

                    output_path, sample_rate = result

                    # Encode to base64
                    logger.info("Encoding audio to base64...")
                    audio_base64 = encode_audio_to_base64(output_path)

                    if audio_base64 is None:
                        return {"error": "Failed to encode audio output"}

            finally:
                # Restore working directory
                os.chdir(current_dir)

            logger.info("✅ TTS generation completed successfully")

            # Return result
            return {
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "model_used": model_type,
                "text_length": len(text)
            }

    except Exception as e:
        logger.error(f"Error in handler: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    # Initialize models once at startup
    initialize_models()

    # Start RunPod serverless worker
    logger.info("🚀 Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
