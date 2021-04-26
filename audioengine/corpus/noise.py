import os
from pathlib import Path
import pandas as pd

from audioengine.corpus.audiodataset import AudioDataset


class Noise(AudioDataset):
    def __init__(self, path, audio_format="wav", sample_rate=16_000, **kwargs):
        super(Noise, self).__init__(audio_format=audio_format, sample_rate=sample_rate, **kwargs)
        self.path = path

    def load_dataframe(self, **kwargs):
        paths = os.listdir(self.path)  # relative Path
        paths = [str(Path(self.path, p).resolve()) for p in paths]  # absolute Path

        return pd.DataFrame({"path": paths})
