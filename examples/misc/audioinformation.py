import os
from multiprocessing import Pool

from audioengine.logging.logging import defaultLogger
from audioengine.transformations.backend.librosa.io import IO

import pandas as pd
from tqdm.auto import tqdm
logger = defaultLogger()


class AudioInformation:
    """
    Load Audio-Informations from Path
    """

    def __init__(self, path, threads=None):
        """

        Args:
            path:
            threads: Thread_Count, default: 2x CPU-Count
        """
        self.path = path
        self.threads = threads if threads else os.cpu_count() * 2

        logger.debug(str({
            "path": path, "threads": self.threads
        }))

        self._load_info()

    def _load_info(self):
        _full_path = lambda relative_path: os.path.join(self.path, relative_path)
        audio_files = tqdm([_full_path(relative_path) for relative_path in os.listdir(self.path)])
        logger.debug(f"Files: {len(audio_files)}")
        _threads = min(len(audio_files), self.threads) #-> threads > jobs -> exception
        logger.debug(f"Using: {_threads} Threads.")
        with Pool(_threads) as p:
            data = p.map(IO.load_duration_and_sr, audio_files)  # -> [(dur, sr), (dur, sr), ...]
            df = pd.DataFrame(data, columns=['Durations', 'SampleRates'])

            duration_info = {
                "min": df.Durations.min(),
                "max": df.Durations.max(),
                "mean": df.Durations.mean(),
                "sum": df.Durations.sum(),
            }

            sample_rates = df.groupby(["SampleRates"]).count()

            info = {
                "duration_info": duration_info,
                "sample_rates": sample_rates
            }

            if len(audio_files) != len(df):
                info["error"] = "AudioFiles-Length != DF-Length"

            self.info = info

    def __str__(self):
        return str(self.info)


if __name__ == "__main__":
    ai = AudioInformation("C:\workspace\datasets\LJSpeech-1.1\wavs")
    print(ai)
