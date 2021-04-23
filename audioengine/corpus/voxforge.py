import os
from pathlib import Path
import pandas as pd
import numpy as np
from audioengine.corpus.audiodataset import AudioDataset


class VoxForge(AudioDataset):
    def __init__(self, path, **kwargs):
        super(VoxForge, self).__init__(audio_format="wav", sample_rate=48_000, **kwargs)
        self.path = path

    def load_dataframe(self, **kwargs):
        data = self._load_data()
        shuffle = kwargs.get("shuffle", False)

        df = pd.DataFrame(data, columns=['path', 'sentence'])
        print("KWARGS", shuffle)
        if shuffle:
            df = df.sample(frac=1)

        return df

    def _list_prompts(self):
        folders = os.listdir(self.path)
        info_paths = [Path(self.path, entry, "etc", "PROMPTS") for entry in folders]
        return info_paths

    def _load_prompt(self, prompt_path):
        data = []
        with open(prompt_path, "r") as file:
            for line in file:
                _split = line.split()
                audio_path = _split[0]
                audio_path = str(Path(self.path, audio_path + ".wav").resolve())

                #/share/datasets/voxforge_todo/anonymhatschie-20140526-wth/mfc/de3-47.wav
                #->
                #/share/datasets/voxforge_todo/anonymhatschie-20140526-wth/wav/de3-47.wav

                audio_splitted = audio_path.split("/")
                audio_splitted[-2] = "wav"
                audio_path = "/".join(audio_splitted)

                text = " ".join(_split[1:]).lower()
                data.append((audio_path, text))
        return data

    def _load_data(self):
        """ [(path, transcription), (path, transcription), ...] """
        prompts = self._list_prompts()
        data = np.array([self._load_prompt(p) for p in prompts])
        flattend = data.reshape(-1, data.shape[-1])
        return flattend






