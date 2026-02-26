import os
import tempfile
import platform
import torch
import soundfile as sf
import logging
from datetime import datetime

# Patch BiCodecTokenizer for PyTorch 2.6+ meta tensor compatibility.
# BiCodec.load_from_checkpoint() may leave some parameters as meta tensors
# (via strict=False load_state_dict + remove_weight_norm). The original
# .to(device) call fails on meta tensors. This patch captures the real
# (non-meta) state dict and falls back to to_empty() + reload when needed.
def _patch_bicodec_tokenizer():
    try:
        from sparktts.models import audio_tokenizer
        from sparktts.models.bicodec import BiCodec
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        def _safe_initialize_model(self):
            model = BiCodec.load_from_checkpoint(f"{self.model_dir}/BiCodec")
            cpu_sd = {k: v for k, v in model.state_dict().items()
                      if v.device.type != "meta"}
            try:
                self.model = model.to(self.device)
            except NotImplementedError:
                logging.info("BiCodec meta tensor fallback: using to_empty()")
                self.model = model.to_empty(device=self.device)
                self.model.load_state_dict(
                    {k: v.to(self.device) for k, v in cpu_sd.items()},
                    strict=False,
                )

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                f"{self.model_dir}/wav2vec2-large-xlsr-53"
            )
            self.feature_extractor = Wav2Vec2Model.from_pretrained(
                f"{self.model_dir}/wav2vec2-large-xlsr-53"
            ).to(self.device)
            self.feature_extractor.config.output_hidden_states = True

        audio_tokenizer.BiCodecTokenizer._initialize_model = _safe_initialize_model
        logging.info("Patched BiCodecTokenizer for meta tensor compatibility")
    except ImportError:
        logging.warning("SparkTTS modules not found, skipping BiCodec patch")

_patch_bicodec_tokenizer()

from cli.SparkTTS import SparkTTS

class SparkModel:
    def __init__(self, model_dir=None, device=0):
        self.model = None
        self.device = None
        """Load the model once at the beginning.

        Args:
            model_dir: Path to Spark-TTS model directory.
                       Resolved by model_cache.py (Network Volume or Docker fallback).
        """
        if model_dir is None:
            # Default: Network Volume path
            volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
            model_dir = os.path.join(volume_path, "models", "Spark-TTS-0.5B")
        logging.info(f"Loading Spark model from: {model_dir}")
        # Determine appropriate device based on platform and availability
        if platform.system() == "Darwin":
            # macOS with MPS support (Apple Silicon)
            self.device = torch.device(f"mps:{device}")
            logging.info(f"Using MPS device: {device}")
        elif torch.cuda.is_available():
            # System with CUDA support
            self.device = torch.device(f"cuda:{device}")
            logging.info(f"Using CUDA device: {device}")
        else:
            # Fall back to CPU
            device = torch.device("cpu")
            logging.info("GPU acceleration not available, using CPU")   
        self.model = SparkTTS(model_dir, device)

    def run_tts(
        self,
        text,
        prompt_speech=None,
        output_path=None,
    ):
        prompt_text = None
        if prompt_speech is None:
            return None
        if prompt_text is not None:
            prompt_text = None if len(prompt_text) <= 1 else prompt_text
        #sample_data, sr = sf.read(prompt_speech)
        # Perform inference and save the output audio
        with torch.no_grad():
            wav = self.model.inference(
                text,
                prompt_speech,
                prompt_text,
                None,
                None,
                None,
            )

            sf.write(output_path, wav, samplerate=16000)
        logging.info(f"Audio saved at: {output_path}")

        return output_path, 16000