from audioengine.corpus.util.text import save_settings
import pandas as pd
from audioengine.corpus.dataset import Dataset
from audioengine.logging.logging import defaultLogger
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import json
import os

logger = defaultLogger()


def build_job_df(df, settings,file_prefix="ffmpeg_"):
    df = df.rename(columns={"path": "path_src"})
    output_sub_dir = settings.get("output_subfolder", "")

    output_dir = Path(settings["output_dir"], output_sub_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    new_path = lambda path: str(Path(output_dir, file_prefix + Path(path).name).resolve())
    df["path"] = df["path_src"].map(new_path)
    return df


def apply_filter(job_df, settings):
    _threads = settings.get("threads", os.cpu_count() * 2)
    _threads = min(len(job_df), _threads)
    output_dir = Path(settings["output_dir"])
    setting_path = Path(output_dir, "settings.json")
    ffmpeg_filter = settings["ffmpeg_filter"]

    cmds = [f'ffmpeg -i "{src}" {ffmpeg_filter} "{dst}"'
            for src, dst in zip(job_df["path_src"], job_df["path"])]

    save_settings(setting_path, settings)

    logger.debug(f"Saving Settings: {str(setting_path.resolve())}")
    logger.debug(f"Apply FFMPEG Filter: {ffmpeg_filter}")
    logger.debug(f"Output Dir: {output_dir}")

    with Pool(min(_threads, len(cmds))) as p:
        p.map(os.system, tqdm(cmds, desc=f"Filterjobs: {_threads} Threads", total=len(cmds)))

    logger.debug("Filder applied!")


def save_df(df, settings):
    df_path = Path(settings["output_dir"], settings["file_name"])

    sep = settings.get("sep", ",")
    df.to_csv(df_path, encoding="utf-8", index=False, sep=sep)

    logger.debug(f"Saved Filterd Dataframe to: {str(df_path.resolve())}")


if __name__ == "__main__":
    df_settings = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train_small",
    }

    settings = {
        "output_dir": "/share/datasets/cv_sm_noise_random",
        "output_subfolder": "wavs",
        "sep": "\t",
        "file_name": "processed_train_small.tsv",
        "ffmpeg_filter": '-af "highpass=200,lowpass=3000,afftdn"',
    }

    df, df_info = Dataset("torch")._load_from_name(**df_settings)
    df_job = build_job_df(df, settings)

    apply_filter(df_job, settings)

    df_job = df_job.drop(columns=["path_src"])
    save_df(df_job, settings)







if __name__ == "__main__":
    df_settings = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train_small",
    }

    settings = {
        "output_dir": "/share/datasets/cv_sm_noise_random",
        "output_subfolder": "wavs",
        "sep": "\t",
        "file_name": "processed_train_small.tsv",
        "ffmpeg_filter": '-af "highpass=200,lowpass=3000,afftdn"',
    }

    df, df_info = Dataset("torch")._load_from_name(**df_settings)
    df_job = build_job_df(df, settings)

    apply_filter(df_job, settings)

    df_job = df_job.drop(columns=["path_src"])
    save_df(df_job, settings)




