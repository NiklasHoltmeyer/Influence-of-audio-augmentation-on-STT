from audioengine.corpus.audiodataset import AudioDataset
import logging

from audioengine.corpus.util.interceptors import time_logger
from pathlib import Path

from audioengine.logging.logging import defaultLogger


class CommonVoice(AudioDataset):

    def __init__(self, path, **kwargs):
        super(CommonVoice, self).__init__(audio_format="mp3", sample_rate=48_000, **kwargs)
        self.path = path
        self.wav_folder_path = str(Path(self.path, "clips"))
        self.logger = defaultLogger()

    @time_logger(name="CV-load DF",
                 header="CommonVoice", padding_length=50)
    def load_dataframe(self, **kwargs):
        tsv_path = self.load_preprocessed_df(**kwargs)

        self.logger.info(f'Loading CommonVoice-Split {kwargs.get("type", "dev")}')

        dataframe = super().load_dataframe(tsv_path, sep="\t", encoding="utf-8", **kwargs)

        return dataframe

    def load_preprocessed_df(self, **kwargs):
        # test, dev, train, validated, ...
        type = kwargs.get("type", "dev")
        input_path, processed_path = self.get_path(type)

        if not processed_path.exists():
            drop_cols = ["client_id", 'up_votes', "down_votes", "age", "gender", "accent", "locale", "segment"]
            rename_cols = None
            full_path_fn = lambda f: str(Path(self.wav_folder_path, f))
            dataframe = super().load_dataframe(str(input_path.resolve()),
                        drop_cols=drop_cols, rename_cols=rename_cols, sep="\t",
                        encoding="utf-8", full_path_fn=full_path_fn)
            dataframe = super().add_duration_column(dataframe, desc=f"Preprocess CV-DF-{type}")
            dataframe.to_csv(processed_path, sep="\t", encoding="utf-8")

        return str(processed_path.resolve())


    def get_path(self, name):
        mapping = {
            "audio_folder": "clips",
            "invalidated": "invalidated.tsv",
            "reported": "reported.tsv",
            "train": "train.tsv",
            "dev": "dev.tsv",
            "other": "other.tsv",
            "test": "test.tsv",
            "validated": "validated.tsv",
        }

        value = mapping.get(name, None)
        assert value

        input_path, processed_path = Path(self.path, value), Path(self.path, "processed_"+value)

        return input_path, processed_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)    
    df = CommonVoice(r"/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de").load_dataframe(type="other")
    type_all = ["train", "dev", "test", "invalidated", "reported", "other", "validated"]

    df = CommonVoice(r"/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de").load_dataframe(type="test")
    print(df.head(5))
    print("*")
    print(df.head(-5))

