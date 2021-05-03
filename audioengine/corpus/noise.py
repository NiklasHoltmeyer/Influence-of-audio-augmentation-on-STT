import os
from pathlib import Path
import pandas as pd

from audioengine.corpus.audiodataset import AudioDataset
from audioengine.corpus.util.interceptors import time_logger


class Noise(AudioDataset):
    def __init__(self, path, audio_format="wav", sample_rate=16_000, **kwargs):
        super(Noise, self).__init__(audio_format=audio_format, sample_rate=sample_rate, **kwargs)
        self.path = path

    @time_logger(name="Noise-load DF",
                 header="Noise", padding_length=50)
    def load_dataframe(self, **kwargs):
        preprocessed = Path(self.path, "info.csv")
        if not preprocessed.exists():
            self.create_df(preprocessed)
        kwargs["min_target_length"] = kwargs["max_target_length"] = None

        return super().load_dataframe(preprocessed, sep="\t", encoding="utf-8", **kwargs)

    def create_df(self, target_path, **kwargs):
        paths = os.listdir(self.path)
        paths = [str(Path(self.path, p).resolve()) for p in paths]  # absolute Path
        df = pd.DataFrame({"path": paths})
        df = super().add_duration_column(df, desc=f"Preprocess Noise-DF")
        df = df[["path", "duration"]]
        df.to_csv(target_path, sep="\t", encoding="utf-8", index=False)




if __name__ == "__main__":
    path = "/share/datasets/FSD50K"
    noise_df = Noise(path).load_dataframe()
    print(noise_df.duration.mean())
    print(noise_df.head(5))

