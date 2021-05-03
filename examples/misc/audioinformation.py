import os
from multiprocessing import Pool

from audioengine.logging.logging import defaultLogger
from audioengine.transformations.backend.librosa.io import IO

import pandas as pd
from tqdm.auto import tqdm
logger = defaultLogger()

import glob

class AudioInformation:
    """
    Load Audio-Informations from Path
    """

    def __init__(self, path, regexp=False, threads=None):
        """

        Args:
            path:
            threads: Thread_Count, default: CPU-Count
        """
        self.path = path
        self.regexp = regexp
        self.threads = threads if threads else os.cpu_count()

        logger.debug(str({
            "path": path, "threads": self.threads
        }))

        self._load_info()

    def _load_paths(self):
        if self.regexp:
            return self._load_paths_from_regexp()
        return self._load_paths_from_folder()

    def _load_paths_from_folder(self):
        _full_path = lambda relative_path: os.path.join(self.path, relative_path)
        audio_files = tqdm([_full_path(relative_path) for relative_path in os.listdir(self.path)])
        return audio_files

    def _load_paths_from_regexp(self):
        return tqdm(glob.glob(vf_de))

    def _load_info(self):
        audio_files = self._load_paths()
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
#    cv_de_full_, regexp = "C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de\clips", False #-> de
#    ljspeech_full, regexp = "C:\workspace\datasets\LJSpeech-1.1\wavs", False # -> en
    vf_de, regexp = "/share/datasets/vf_de/*/wav/*.wav", True
    VoxForge()
    ai = AudioInformation(vf_de, regexp=regexp)
    print(ai)

#LJSpeech-1.1
#{'duration_info': {'min': 1.1100680272108843, 'max': 10.096190476190475, 'mean': 6.573822616883905, 'sum': 86117.07628117915}, 'sample_rates':              Durations
#SampleRates
#22050            13100}

#vf_de
#{'duration_info': {'min': 0.375, 'max': 19.385, 'mean': 4.74977039953413, 'sum': 117248.0823125}, 'sample_rates':              Durations
#SampleRates
#16000            24685}