import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from audioengine.corpus.audiodataset import AudioDataset
import codecs

from audioengine.corpus.util.interceptors import time_logger


class VoxForge(AudioDataset):
    def __init__(self, path, **kwargs):
        super(VoxForge, self).__init__(audio_format="wav", sample_rate=48_000, **kwargs)
        self.path = path

    @time_logger(name="VF-load DF",
                 header="VoxForge", padding_length=50)
    def load_dataframe(self, **kwargs):
        tsv_path = self.load_preprocessed_df(**kwargs)

        return super().load_dataframe(tsv_path, sep="\t", encoding="utf-8", **kwargs)


    def load_preprocessed_df(self, **kwargs):
        processed_path = Path(self.path, "info.csv")
        type = kwargs.get("type", None)

        if not processed_path.exists():
            data = self._load_data()
            df = pd.DataFrame(data, columns=['path', 'sentence'])
            df = super().add_duration_column(df, desc=f"Preprocess VF-DF (Durations)")
            df = super().add_target_Lengths(df, desc=f"Preprocess VF-DF (Target-Lengths)")
            df = df[["path", "sentence", "duration", "target_length"]]
            df.to_csv(processed_path, sep="\t", encoding="utf-8", index=False)

        if type is None:
            return processed_path

        assert type in {"train", "test"}

        processed_type_path = Path(self.path, f"info_{type}.csv")
        if not processed_type_path.exists():
            processed_train_path = Path(self.path, f"info_train.csv")
            processed_test_path = Path(self.path, f"info_test.csv")

            dataframe = super().load_dataframe(processed_path, sep="\t", encoding="utf-8", **kwargs)
            dataframe = dataframe[["path", "sentence", "duration", "target_length"]]

            train_dataframe, test_dataframe = train_test_split(dataframe, train_size=0.8)
            train_dataframe.to_csv(processed_train_path, sep="\t", encoding="utf-8", index=False)
            test_dataframe.to_csv(processed_test_path, sep="\t", encoding="utf-8", index=False)

        return processed_type_path

    def _list_prompts(self):
        folders = os.listdir(self.path)
        info_paths = [Path(self.path, entry, "etc", "PROMPTS") for entry in folders]
        return info_paths

    def _load_prompt(self, prompt_path):
        data = []
        base_folder = prompt_path.parent.parent
        wav_folder = Path(base_folder, "wav").resolve()
        with open(prompt_path, 'r', encoding='utf-8') as file:
            for line in file:
                _split = line.split()
                audio_path = _split[0]
                audio_path = str(Path(self.path, audio_path + ".wav").resolve())

                audio_splitted = audio_path.split("/")
                file_name = audio_splitted[-1]
                audio_path = Path(wav_folder, file_name)

                if not audio_path.exists():
                    raise Exception(f"Invalid Audio Path: {audio_path}")

                text = " ".join(_split[1:]).lower()
                data.append((str(audio_path.resolve()), text))
        return data

    def _load_data(self):
        """ [(path, transcription), (path, transcription), ...] """
        prompts = self._list_prompts()
        data = []
        for p in prompts:
            for item in self._load_prompt(p):
                data.append(item)
        return np.array(data)

if __name__ == "__main__":
    print("Path", "/share/datasets/vf_de")
    df = VoxForge("/share/datasets/vf_de").load_dataframe(type="train")
    print(df.head())







