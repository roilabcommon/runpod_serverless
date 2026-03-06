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
            try:
                self.model = model.to(self.device)
            except NotImplementedError:
                # Only replace meta tensors (no real data) with empty CPU tensors
                # so that .to(device) can proceed. All real weights are preserved.
                logging.info("BiCodec has meta tensors, replacing before device transfer")
                for module in model.modules():
                    for name, param in list(module._parameters.items()):
                        if param is not None and param.device.type == "meta":
                            module._parameters[name] = torch.nn.Parameter(
                                torch.empty(param.shape, dtype=param.dtype),
                                requires_grad=param.requires_grad,
                            )
                    for name, buf in list(module._buffers.items()):
                        if buf is not None and buf.device.type == "meta":
                            module._buffers[name] = torch.empty(
                                buf.shape, dtype=buf.dtype
                            )
                self.model = model.to(self.device)
            # Ensure float32: mel spectrogram expects float32 audio input
            self.model.float()
            # Re-register spectrogram window buffers that were lost as meta tensors
            for module in self.model.modules():
                cls_name = type(module).__name__
                if hasattr(module, 'spectrogram') and hasattr(module.spectrogram, 'window'):
                    spec = module.spectrogram
                    if spec.window is None or spec.window.numel() == 0:
                        n_fft = spec.n_fft if hasattr(spec, 'n_fft') else spec.window.shape[0]
                        spec.register_buffer('window', torch.hann_window(n_fft, device=self.device))
                        logging.info(f"Re-registered spectrogram.window for {cls_name}")
                elif cls_name in ('MelSpectrogram', 'Spectrogram') and hasattr(module, 'window'):
                    if module.window is None or module.window.numel() == 0:
                        n_fft = module.n_fft if hasattr(module, 'n_fft') else module.win_length
                        module.register_buffer('window', torch.hann_window(n_fft, device=self.device))
                        logging.info(f"Re-registered window for {cls_name}")

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


# Patch detokenize to cast token indices to long (int64).
# einx's take() requires long tensors but SparkTTS may produce int32 tokens.
def _patch_detokenize_dtype():
    try:
        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        _orig_detokenize = BiCodecTokenizer.detokenize

        def _safe_detokenize(self, *args, **kwargs):
            args = tuple(
                a.long() if isinstance(a, torch.Tensor) and a.dtype == torch.int32 else a
                for a in args
            )
            return _orig_detokenize(self, *args, **kwargs)

        BiCodecTokenizer.detokenize = _safe_detokenize
        logging.info("Patched BiCodecTokenizer.detokenize for int64 token indices")
    except ImportError:
        logging.warning("SparkTTS modules not found, skipping detokenize patch")

_patch_detokenize_dtype()


from cli.SparkTTS import SparkTTS

class SparkModel:
    def __init__(self, model_dir=None, device=0):
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
            self.device = torch.device(f"mps:{device}")
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Using device: {self.device}")
        self.model = SparkTTS(model_dir, self.device)

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

            # Convert CUDA tensor to numpy for sf.write
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            sf.write(output_path, wav, samplerate=16000)
        logging.info(f"Audio saved at: {output_path}")

        return output_path, 16000