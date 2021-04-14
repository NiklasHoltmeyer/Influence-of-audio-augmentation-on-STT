from abc import abstractmethod, ABC, ABCMeta

from tqdm.auto import tqdm
import logging
import swifter
import os
from audioengine.corpus.util.text import Text
from audioengine.corpus.util.interceptors import time_logger


class AudioDataset(metaclass=ABCMeta):

    def __init__(self, sample_rate, **kwargs):
        self.logger = logging.getLogger("audioengine-corpus")
        self.sample_rate = sample_rate
        self.audio_format = kwargs.get("audio_format", "wav")

    @time_logger(logger=logging.getLogger("audioengine-corpus"),
                 name="  -load DF", padding_length=50)
    def load_dataframe(self, path, shuffle=True, drop_cols=None, rename_cols=None, **kwargs):
        data_frame = Text.read_csv(path, **kwargs).fillna("")

        if drop_cols:
            data_frame.drop(drop_cols, inplace=True, axis=1, errors='ignore')

        if shuffle:
            data_frame = data_frame.sample(frac=1) if shuffle else data_frame

        if rename_cols:
            data_frame = data_frame.rename(columns=rename_cols)

        return data_frame

    def sanity_check(self, paths):
        for path in paths:
            assert os.path.exists(path)


swifter.register_modin()
tqdm.pandas()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
