import torch
import traceback
import os
import sys
import gc
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed, BitsAndBytesConfig
    
    
def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VibeVoiceModel:
    def __init__(self, model_path=None):
        """
        Initialize VibeVoice model optimized for 11GB VRAM using 4-bit quantization.

        Args:
            model_path: Path to model directory. If None, uses default
                        relative path "vibevoice/VibeVoice-7B".

        Requirements:
            pip install bitsandbytes-windows  (for Windows)
            or
            pip install bitsandbytes  (for Linux)
        """
        if model_path is None:
            # Default: Network Volume path
            volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
            model_path = os.path.join(volume_path, "models", "VibeVoice-7B")
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_steps = 20
        self.cfg_scale = 2.0
        self.processor = None
        self.model = None
        self._setup_memory_config()
        self.load_model()
    
    def _setup_memory_config(self):
        """Configure memory settings for optimal VRAM usage."""
        if self.device == "cuda":
            clear_memory()
        
    def load_model(self):
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        if self.device == "cuda":
            print("📥 Loading model with 4-bit quantization to GPU...")
            print("💡 Using NF4 quantization (~4-5GB VRAM)")
            
            # 4-bit quantization config for 11GB VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                )
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"📊 GPU Memory after loading: {allocated:.2f}GB")
                
            except Exception as e:
                print(f"⚠️ 4-bit loading failed: {e}")
                print("📥 Falling back to CPU mode...")
                self.device = "cpu"
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
        else:
            print("📥 Loading model to CPU...")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        
        self.model.eval()
        
        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
        
        print(f"✅ Model loaded successfully! Device: {self.device}")
        clear_memory()

    def __read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav, sr
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([]), 0    

    def _estimate_max_tokens(self, text: str) -> int:
        """
        Estimate max_new_tokens based on text length.

        Each speech_diffusion_id token = 3200 audio samples at 24kHz = ~0.133 sec.
        So 1 second of audio = ~7.5 diffusion tokens.
        Korean speaking rate: ~3-4 chars/sec.
        """
        char_count = len(text.strip())

        # Estimate speech duration (conservative: slower speaking rate)
        estimated_seconds = max(char_count / 3.0, 1.5)  # at least 1.5 seconds
        # ~7.5 diffusion tokens per second + 2.5x safety margin for pauses/prosody
        estimated_tokens = int(estimated_seconds * 7.5 * 2.5)
        # Add overhead for speech_start, speech_end, eos tokens
        estimated_tokens += 15
        # Minimum 30 tokens to avoid cutting off very short utterances
        return max(estimated_tokens, 30)

    def _trim_trailing_silence(self, audio: np.ndarray, sr: int = 24000,
                               threshold_db: float = -40.0,
                               frame_ms: int = 50,
                               min_silence_ms: int = 500) -> np.ndarray:
        """
        Trim trailing silence or low-energy noise/music from generated audio.

        Finds the last frame with energy above threshold, then keeps audio
        up to that point plus a short fade-out.
        """
        if len(audio) == 0:
            return audio

        frame_size = int(sr * frame_ms / 1000)
        num_frames = len(audio) // frame_size

        if num_frames == 0:
            return audio

        # Calculate RMS energy per frame
        threshold_linear = 10 ** (threshold_db / 20)
        last_active_frame = 0

        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > threshold_linear:
                last_active_frame = i

        # Keep up to the last active frame + trailing margin
        margin_frames = int(min_silence_ms / frame_ms)
        end_frame = min(last_active_frame + margin_frames + 1, num_frames)
        trimmed = audio[: end_frame * frame_size]

        if len(trimmed) < len(audio):
            print(f"Trimmed audio: {len(audio)/sr:.2f}s -> {len(trimmed)/sr:.2f}s")

        return trimmed

    def run_tts(
        self,
        text,
        prompt_speech=None,
        cfg_scale=2,
    ):
        if prompt_speech is None:
            return None
        voice_sample, sr = self.__read_audio(prompt_speech)
        lines = text.strip().split('\n')
        formatted_script_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if line already has speaker format
            if line.startswith('Speaker ') and ':' in line:
                formatted_script_lines.append(line)
            else:
                # Auto-assign to speakers in rotation
                speaker_id = len(formatted_script_lines)
                formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

        formatted_script = '\n'.join(formatted_script_lines)

        # Estimate max tokens to prevent over-generation on short text
        max_tokens = self._estimate_max_tokens(formatted_script)
        # Use higher cfg_scale for short text to keep generation faithful
        effective_cfg = max(cfg_scale, 3.0) if len(formatted_script) < 30 else cfg_scale
        print(f"Text length: {len(formatted_script)} chars, max_new_tokens: {max_tokens}, cfg_scale: {effective_cfg}")

        inputs = self.processor(
            text=[formatted_script],
            voice_samples=[voice_sample],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                max_length_times=1.0,
                cfg_scale=effective_cfg,
                inference_steps=self.inference_steps,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
            )

        audio_output = outputs.speech_outputs[0]

        # Post-process: trim trailing silence/noise/music
        if hasattr(audio_output, 'cpu'):
            audio_np = audio_output.cpu().numpy()
        else:
            audio_np = np.array(audio_output)
        if len(audio_np.shape) > 1:
            audio_np = audio_np.squeeze()

        audio_np = self._trim_trailing_silence(audio_np, sr=24000)

        print("Generation successful!")
        return audio_np, 24000

