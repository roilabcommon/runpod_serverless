import os
import tempfile
import platform
import torch
import soundfile as sf
import logging
from datetime import datetime
from cli.SparkTTS import SparkTTS

class SparkModel:
    def __init__(self, model_dir="TTS/pretrained_models/Spark-TTS-0.5B", device=0):
        self.model = None
        self.device = None
        """Load the model once at the beginning."""
        logging.info(f"Loading model from: {model_dir}")    
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