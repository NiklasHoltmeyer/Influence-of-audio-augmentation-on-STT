import json
import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from random import uniform

from tqdm import tqdm

import librosa
import numpy as np
from audioengine.corpus.augmentation.settings import filter_settings

from audioengine.corpus.dataset import logger
from audioengine.corpus.util.interceptors import time_logger
from audioengine.logging import logging
from audioengine.logging.logging import defaultLogger
from audioengine.transformations.backend.librosa.effect import Effect

from pysndfx import AudioEffectsChain

from audioengine.transformations.backend.librosa.io import IO

logger = defaultLogger()

def add_filter_job_column(df, filter_settings):
    _filter_list = [""] * len(df)
    for name, settings in filter_settings.items():
        prop = settings["probability"]
        y_n = np.random.choice([name, ""], len(df), p=[prop, 1 - prop])

        for idx, choice in enumerate(y_n):
            _filter_list[idx] = _filter_list[idx] + "+" + choice

    clean_job = lambda entry: "+".join([x for x in entry.split("+") if len(x) > 0])
    _filter_list = [clean_job(x) for x in _filter_list]

    df["filter_job"] = _filter_list
    return df


def add_real_noise_column(signal_df, noise_df):
    #signal_df = signal_df.sort_values(by=['duration']).reset_index(drop=True)
    #noise_df = noise_df.sort_values(by=['duration']).reset_index(drop=True)

    len_dif = len(signal_df) - len(noise_df)

    if len_dif < 0:  # -> s_df < n_df
        logger.debug(f"More Noise then Signale Samples! Truncating Noise")
        noise_paths = noise_df[: len(signal_df)]["path"]
    elif len_dif > 0:  # -> s_df > n_df
        logger.debug(f"More Signale then Noise Samples! Padding Noise")
        pad_fn = 'symmetric'
        beg = int(len_dif / 2)
        end = len_dif - beg
        noise_paths = np.pad(noise_df["path"], (beg, end), pad_fn)
    else:
        logger.debug(f"Signales and Noises are of the same Shape")
        noise_paths = noise_df["path"]

    # assert len(signal_df) == len(noise_df), "Padding (Dataframes) failed"
    signal_df["path_noise"] = noise_paths

    return signal_df.sample(frac=1).reset_index(drop=True)


def add_output_path_column(df, output_dir, file_prefix, subfolder=None):
    output_path = Path(output_dir, subfolder) if subfolder else Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = str(output_path.resolve())

    def _calc_output_path(item):
        input_path = item["path_input"]
        filter_applied = len(item.filter_job) != 0
        if not filter_applied:
            return input_path
        file_name_w_o_extension = file_prefix + Path(input_path).name.split(".")[0] + ".wav"
        return str(Path(output_path, file_name_w_o_extension).resolve())

    df["path_output"] = df.apply(_calc_output_path, axis=1)
    return df


def zip_jobs(df):
    """

    Args:
        df: Input DF

    Returns:
        [(Path_Input, Path_Noise, Path_Output, Filter_Job)]
    """
    _df = (df[["path_input", "path", "path_noise", "filter_job"]])[df.filter_job != ""]
    _paths_in = _df["path_input"]
    _paths_out = _df["path"]
    _filter_jobs = _df["filter_job"]
    _path_noise = _df["path_noise"]

    return zip(_paths_in, _path_noise, _paths_out, _filter_jobs)


def random_rate(_range):
    _from, _to = _range
    return lambda: uniform(_from, _to)


def callback_dict(filter_settings, target_sample_rate):
    range_fn_mapping = {}
    for key, value in filter_settings.items():
        _range = value["range"]
        range_fn_mapping[key] = random_rate(_range)

    if "real_noise" in filter_settings.keys():
        snr_fn = random_rate(filter_settings["real_noise"]["range"])

    def __add_real_noise(idx, yp, rate, y_n_path):
        #        yp, _ = IO.load(y_path, sample_rate=target_sample_rate)
        yn, _ = IO.load(y_n_path, sample_rate=target_sample_rate)

        return Effect.add_noise(yp, yn, snr=snr_fn(), pad_idx=idx)

    job_fn_mapping = {
        "time_stretch": lambda __, y, rate, _: Effect.time_stretch(y, rate),
        "harmonic_remove": lambda __, y, rate, _: (y - 0.5 * librosa.effects.harmonic(y, margin=rate)),
        "percussive_remove": lambda __, y, rate, _: (y - 0.5 * librosa.effects.percussive(y, margin=rate)),
        "percussive": lambda __, y, rate, _: librosa.effects.percussive(y, margin=rate),
        "harmonic": lambda __, y, rate, _: librosa.effects.harmonic(y, margin=rate),
        "random_noise": lambda __, y, rate, _: Effect.add_noise_random(y, rate),
        "real_noise": lambda idx, y, rate, y_n: __add_real_noise(idx, y, rate, y_n),
        "reverb": lambda __, y, rate, _: AudioEffectsChain().reverb(reverberance=rate, hf_damping=rate,
                                                                    room_scale=rate * 2)(
            y),
        "bandpass": lambda __, y, rate, _: AudioEffectsChain().bandpass(rate)(y),
        "tremolo": lambda __, y, rate, _: AudioEffectsChain().tremolo(rate)(y),
    }

    assert False not in [key in job_fn_mapping.keys() for key in filter_settings.keys()], "Uknown Filter-Option"

    return job_fn_mapping, range_fn_mapping


def save_settings(df, output_dir, filter_settings, file_name="data.csv", **kwargs):
    sep = kwargs.get("sep", ";")
    df_path = Path(output_dir, file_name)
    json_path = Path(output_dir, "transform_settings.json")
    df.to_csv(df_path.resolve(), sep=sep, encoding="utf8", index=False)

    _len_unfilterd = len(df[df.filter_job != ""])
    _len_filterd = len(df) - _len_unfilterd

    transform_settings = {
        "filter_settings": filter_settings,
        "dataset_info": {
            "un_changed": _len_unfilterd, "transformed": _len_filterd,
            "sum": len(df)
        }
    }

    settings_json = json.dumps(transform_settings, indent=4)
    with open(json_path, "w") as f:
        f.write(settings_json)

    logger.debug(f"Save Settings {{{file_name}, transform_settings.json}} to  {output_dir}")


def execute_job(job):
    idx, (_paths_in, _path_noise, _paths_out, _filter_jobs) = job
    y, sr = IO.load(_paths_in, target_samplerate)
    job_names = _filter_jobs.split("+")
    for job_name in job_names:
        rate = range_fn_mapping[job_name]()
        y = job_fn_mapping[job_name](idx, y, rate, _path_noise)
    IO.save_wav(y, _paths_out, target_samplerate)


@time_logger(name="Building DF",
             header="Augmentation", padding_length=50)
def build_job_df(df, noise_df, filter_settings, output_dir, output_subfolder=None, **kwargs):
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    file_prefix = kwargs.get("file_prefix", "aug_")

    df = add_real_noise_column(df, noise_df)
    df = df.rename({'path': 'path_input'}, axis=1)
    df = add_filter_job_column(df, filter_settings)
    df = add_output_path_column(df=df, output_dir=output_dir, subfolder=output_subfolder, file_prefix=file_prefix)
    df = df.rename({'path_output': 'path'}, axis=1)

    df = df[["path", "sentence", "duration", "target_length", "path_input", "path_noise", "filter_job"]]

    save_settings(df, output_dir, filter_settings, **kwargs)

    return df

global job_fn_mapping, range_fn_mapping, target_samplerate

@time_logger(name="Augment-Dataset",
             header="Augmentation", padding_length=50)
def augment_dataset(df, noise_df, **kwargs):
    global job_fn_mapping, range_fn_mapping, target_samplerate
    _filter_settings = kwargs.pop("filter_settings", filter_settings)
    target_samplerate = kwargs.get("target_sample_rate", 16_000)
    _threads = kwargs.get("threads", os.cpu_count() * 2)
    _output_dir = kwargs.get("output_dir", None)
    _output_subfolder = kwargs.get("output_subfolder", None)

    assert _output_dir, "Please Specifiy output_dir."

    df = build_job_df(df, noise_df, filter_settings=_filter_settings, **kwargs)
    job_fn_mapping, range_fn_mapping = callback_dict(filter_settings=_filter_settings, target_sample_rate=target_samplerate)
    jobs = list(enumerate((zip_jobs(df))))
    job_len = len(jobs)

    warnings.filterwarnings('ignore')

    with Pool(processes=min(_threads, job_len)) as pool: #_threads
        pool.map(execute_job, tqdm(jobs, desc=f"Audio-Augmentation: {_threads} Threads", total=job_len))

    warnings.filterwarnings('default')
    logger.debug("Finished Augmenting Dataset")


