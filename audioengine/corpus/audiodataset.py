from abc import abstractmethod, ABC, ABCMeta
from multiprocessing import Pool

from tqdm.auto import tqdm
import logging
import swifter
import os
from audioengine.corpus.util.text import Text
from audioengine.corpus.util.interceptors import time_logger
from sklearn.model_selection import train_test_split
import warnings
from audioengine.transformations.backend.librosa.io import IO
from tqdm.contrib.concurrent import process_map

class AudioDataset(metaclass=ABCMeta):

    def __init__(self, sample_rate, **kwargs):
        self.logger = logging.getLogger("audioengine-corpus")
        self.sample_rate = sample_rate
        self.audio_format = kwargs.get("audio_format", "wav")

    @time_logger(name="  -load DF", padding_length=50)
    def load_dataframe(self, path, drop_cols=None, rename_cols=None, full_path_fn=None, **kwargs):
        data_frame = Text.read_csv(path, **kwargs).fillna("")
        shuffle = kwargs.get("shuffle", False)
        fixed_length = kwargs.get("fixed_length", None)
        min_duration = kwargs.get("min_duration", None)
        max_duration = kwargs.get("max_duration", None)

        if full_path_fn:
            data_frame.path = data_frame.path.map(full_path_fn)

        if drop_cols:
            data_frame.drop(drop_cols, inplace=True, axis=1, errors='ignore')

        if shuffle:
            data_frame = data_frame.sample(frac=1)

        if rename_cols:
            data_frame = data_frame.rename(columns=rename_cols)

        if min_duration or max_duration:
            len_pre = len(data_frame)
            min_duration = float(max(0, min_duration))

            if max_duration:
                data_frame = data_frame[data_frame["duration"].between(min_duration, float(max_duration))]
            else:
                data_frame = data_frame[data_frame["duration"] >= min_duration]
            len_after = len(data_frame)
            self.logger.debug("*"*72)
            max_duration = max_duration if max_duration else "inf"
            self.logger.debug(f"Duration Range {min_duration}-{max_duration}s")
            self.logger.debug(f"Total DS-Length {len_pre}")
            self.logger.debug(f"Truncated DS-Length {len_after} (-{len_pre-len_after})")
            self.logger.debug("*" * 72)

        if fixed_length:
            if not kwargs.get("shuffle"):
                self.logger.warning("Shuffle is disabled, while fixed_length is enabled.")
            _items = min(int(fixed_length), len(data_frame))
            data_frame = data_frame[: _items]

        return data_frame.reset_index(drop=True)

    def add_duration_column(self, df, desc="", threads=(os.cpu_count()*2)):
        df["duration"] = process_map(IO.load_duration, df["path"], max_workers=threads, desc=f"{desc} [{threads} Threads]")
        return df

    def sanity_check(self, paths):
        for path in paths:
            assert os.path.exists(path)


#swifter.register_modin()
#tqdm.pandas()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
