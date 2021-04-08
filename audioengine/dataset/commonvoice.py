from audioengine.dataset.audiodataset import AudioDataset
import logging

from audioengine.dataset.util.interceptors import time_logger
from pathlib import Path


class CommonVoice(AudioDataset):

    def __init__(self, path, **kwargs):
        super(CommonVoice, self).__init__(audio_format="mp3", sample_rate=48_000, **kwargs)
        self.path = path

    @time_logger(logger=logging.getLogger("audioengine-dataset"),
                 name="CV-load DF",
                 header="CommonVoice", padding_length=50)
    def load_dataframe(self, **kwargs):
        type = kwargs.get("type", "dev")  # test, dev, train, validated, ...
        tsv_path = self.get_path(type)

        drop_cols = ["client_id", 'up_votes', "down_votes", "age", "gender", "accent", "locale", "segment"]
        rename_cols = {"path": "audio_path", "text": "transscript"}

        data_frame = super().load_dataframe(tsv_path, drop_cols=drop_cols, rename_cols=rename_cols, sep="\t", **kwargs)

        return data_frame

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

        return Path(self.path, value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    df = CommonVoice(r"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de").load_dataframe()
    print(df.head())