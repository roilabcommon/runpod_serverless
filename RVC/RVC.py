import random
import os
import zipfile 
import librosa
import time
from infer_rvc_python import BaseLoader
from pydub import AudioSegment
from tts_voice import tts_order_voice
import edge_tts
import tempfile
from audio_separator.separator import Separator
import model_handler
import psutil
import cpuinfo
import fairseq
import torch

class RVC:
    def __init__(self):
        torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
        self.INDEX_INFLUENCE = 0.6
        self.RESPIRATION_MEDIAN_FILTERING = 3
        self.ENVELOPE_RATIO = 0.1
        self.CONSONANT_BREATH_PROTECTION = 0.5

        self.language_dict = tts_order_voice
        self.separator = Separator()
        
        rmvpe_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rmvpe.pt')
        hubert_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'hubert_base.pt')
        self.converter = BaseLoader(only_cpu=False, hubert_path=hubert_path, rmvpe_path=rmvpe_path)
        self.model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights')
        print(f"model_dir : {self.model_dir}")
        

    def run_rvc(self, ch, lang, audio_files, pitch_level):
        print('rvc inference...')
        if not audio_files:
            raise ValueError("The audio pls")

        if isinstance(audio_files, str):
            audio_files = [audio_files]

        try:
            duration_base = librosa.get_duration(filename=audio_files[0])
            print("Duration:", duration_base)
        except Exception as e:
            print(e)

        random_tag = "USER_"+str(random.randint(10000000, 99999999))

        model_file = os.path.join(self.model_dir, lang, f"{ch}.pth")
        index_file = os.path.join(self.model_dir, lang, f"{ch}.index")
        print("File model:", model_file)

        print("Random tag:", random_tag)
        print("File model:", model_file)
        print("Pitch algorithm:", "pm")
        print("Pitch level:", pitch_level)
        print("File index:", None)
        print("Index influence:", self.INDEX_INFLUENCE)
        print("Respiration median filtering:", self.RESPIRATION_MEDIAN_FILTERING)
        print("Envelope ratio:", self.CONSONANT_BREATH_PROTECTION)

        self.converter.apply_conf(
            tag=random_tag,
            file_model=model_file,
            pitch_algo="rmvpe+",
            pitch_lvl=pitch_level,
            file_index=index_file if os.path.exists(index_file) else None,
            index_influence=self.INDEX_INFLUENCE,
            respiration_median_filtering=self.RESPIRATION_MEDIAN_FILTERING,
            envelope_ratio=self.ENVELOPE_RATIO,
            consonant_breath_protection=self.CONSONANT_BREATH_PROTECTION,
            resample_sr=48000 if audio_files[0].endswith('.mp3') else 0, 
        )
        time.sleep(0.1)

        result = self.converter(audio_files, random_tag, self.converter)
        #print("Result:", result)

        return result[0]
    

if __name__ == "__main__":
    rvc = RVC()
    result = rvc.run_rvc("Poli", "test.wav", 0, None)
    print(result)