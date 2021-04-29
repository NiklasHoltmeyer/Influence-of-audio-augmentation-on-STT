from abc import abstractmethod, ABC, ABCMeta

from tqdm.auto import tqdm
import logging
import swifter
import os
from audioengine.corpus.util.text import Text
from audioengine.corpus.util.interceptors import time_logger
from sklearn.model_selection import train_test_split

class AudioDataset(metaclass=ABCMeta):

    def __init__(self, sample_rate, **kwargs):
        self.logger = logging.getLogger("audioengine-corpus")
        self.sample_rate = sample_rate
        self.audio_format = kwargs.get("audio_format", "wav")

    @time_logger(name="  -load DF", padding_length=50)
    def load_dataframe(self, path, drop_cols=None, rename_cols=None, **kwargs):
        data_frame = Text.read_csv(path, **kwargs).fillna("")
        shuffle = kwargs.get("shuffle", False)
        fixed_length = kwargs.get("fixed_length", None)

        if drop_cols:
            data_frame.drop(drop_cols, inplace=True, axis=1, errors='ignore')

        if shuffle:
            data_frame = data_frame.sample(frac=1)

        if rename_cols:
            data_frame = data_frame.rename(columns=rename_cols)

        if fixed_length:
            if not kwargs.get("shuffle"):
                self.logger.warning("Shuffle is disabled, while fixed_length is enabled.")
            _items = min(int(fixed_length), len(data_frame))
            data_frame = data_frame[: _items]

        return data_frame.reset_index(drop=True)

    def sanity_check(self, paths):
        for path in paths:
            assert os.path.exists(path)


swifter.register_modin()
tqdm.pandas()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
