import os
from multiprocessing import Pool
from pathlib import Path
from random import random

from audioengine.corpus.audiodataset import AudioDataset
import pandas as pd
from tqdm import tqdm
from audioengine.corpus.util.interceptors import time_logger
import soundfile as sf

from audioengine.logging.logging import defaultLogger
from audioengine.model.pretrained.silero_tts import SileroTTS


class TTSSynthesized(AudioDataset):
    def __init__(self, text_files, output_dir, tts_engine, **kwargs):
        super(TTSSynthesized, self).__init__(audio_format=kwargs.get("audio_format", "wav"),
                                             sample_rate=kwargs.get("sample_rate", 16_000), **kwargs)
        self.text_files = text_files
        self.output_dir = output_dir

        self.threads = kwargs.get("threads", os.cpu_count())

        self.df_name = kwargs.get("dataframe_name", "data.tsv")
        self.df_sep = kwargs.get("dataframe_sep", "\t")
        self.force = kwargs.get("force_overwrite", False)
        self.min_target_len = kwargs.get("dataframe_min_target_len", 30)
        self.max_target_len = kwargs.get("dataframe_max_target_len", 140)

        self.batch_size = kwargs.get("batch_size", 16000)

        self.ffmpeg_filter = kwargs.get("filter", '-af "highpass=200,lowpass=3000,afftdn"')  # None -> deactivated

        self.tts_engine = tts_engine

        self.logger = defaultLogger()

    @time_logger(name="TTS-load DF",
                 header="TTS", padding_length=50)
    def load_dataframe(self, **kwargs):
        tsv_path = self.load_preprocessed_df(**kwargs)

        dataframe = super().load_dataframe(tsv_path, encoding="utf-8", sep=self.df_sep, **kwargs)

        return dataframe

    def load_preprocessed_df(self, **kwargs):
        df_path = Path(self.output_dir, self.df_name)
        if not df_path.exists():
            self._preprocess_df(df_path)
            dataframe = super().load_dataframe(df_path, encoding="utf-8", sep=self.df_sep, **kwargs)

            dataframe = self._add_does_exist_col(dataframe)
            self._apply_tts(dataframe)
            self._apply_filter(dataframe)
        return df_path

    def _apply_tts(self, dataframe):
        jobs = dataframe[~dataframe.path_exists]
        jobs_batched = chunks(jobs, self.batch_size)

        for batch in tqdm(jobs_batched, total=int(len(jobs) / self.batch_size)):
            texts = batch.text
            target_paths = batch.path
            audios = self.tts_engine.apply(texts)
            for audio, dst in zip(audios, target_paths):
                sf.write(dst, data=audio, samplerate=self.sample_rate, subtype="PCM_24")

    def _apply_filter(self, dataframe):
        jobs = dataframe[~dataframe.path_filterd_exists]
        inputs, outputs = jobs["path"], jobs["path_filterd"]
        cmds = [f'ffmpeg -i "{_io[0]}" {self.ffmpeg_filter} "{_io[1]}"'
                for _io in zip(inputs, outputs)]
        with Pool(min(self.threads, len(jobs))) as p:
            p.map(os.system, cmds)

    def _add_does_exist_col(self, dataframe):
        _exits = lambda x: Path(x).exists()
        dataframe["path_exists"] = dataframe["path"].map(_exits)
        dataframe["path_filterd_exists"] = dataframe["path_filterd"].map(_exits)
        return dataframe

    def _preprocess_df(self, path):
        data = []
        filter_fn = lambda x: self.min_target_len <= len(x) <= self.max_target_len

        for textfile in self.text_files:
            with open(textfile, "r", encoding="utf") as f:
                _data = [d for d in f if filter_fn(d)]
                data.extend(_data)
        df = pd.DataFrame({"text": data}).sample(frac=1).reset_index(drop=True)
        df["path"] = df.index.map(lambda x: str(Path(self.output_dir, "wav", str(x) + ".wav").resolve()))
        df["path_filterd"] = df.index.map(
            lambda x: str(Path(self.output_dir, "wav_filterd", str(x) + ".wav").resolve()))
        df["target_length"] = df["text"].map(len)
        df.to_csv(path, encoding="utf-8", index=False)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    #https://stackoverflow.com/a/312464/5026265
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    output_dir = "/share/datasets/8mil_tts/"
    texts = [output_dir + "/sentences.txt"]

    tts_engine = SileroTTS("de", "thorsten_16khz", "cpu")

    rename_cols = {"path_filterd": "path"}
    fixed_length = 15_000
    df = TTSSynthesized(text_files=texts, output_dir=output_dir, tts_engine=tts_engine)\
        .load_dataframe(rename_cols=rename_cols, fixed_length=fixed_length)

    print(df)
