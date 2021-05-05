import json
from pathlib import Path
from random import uniform

import librosa
import numpy as np

from audioengine.corpus.dataset import logger
from audioengine.transformations.backend.librosa.effect import Effect

from pysndfx import AudioEffectsChain

from audioengine.transformations.backend.librosa.io import IO


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
    signal_df = signal_df.sort_values(by=['duration']).reset_index(drop=True)
    noise_df = noise_df.sort_values(by=['duration']).reset_index(drop=True)

    len_dif = len(signal_df) - len(noise_df)

    if len_dif < 0:  # -> s_df < n_df
        logger.debug("Remove Noise-Samples")
        noise_df = noise_df[: len(signal_df)]["path"]
    elif len_dif > 0:  # -> s_df > n_df
        pad_fn = 'symmetric'
        logger.debug(f"Pad Noise-Samples ({pad_fn})")
        beg = int(len_dif / 2)
        end = len_dif - beg
        noise_df = np.pad(noise_df["path"], (beg, end), pad_fn)
    else:
        logger.debug(f"Signales and Noises are of the same Shape")
        noise_df = noise_df["path"]

    # assert len(signal_df) == len(noise_df), "Padding (Dataframes) failed"
    signal_df["path_noise"] = noise_df
    return signal_df.sample(frac=1).reset_index(drop=True)


def add_output_path_column(df, output_dir, subfolder=None):
    output_path = Path(output_dir, subfolder) if subfolder else Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = str(output_path.resolve())

    def _calc_output_path(item):
        input_path = item["path_input"]
        filter_applied = len(item.filter_job) != 0
        if not filter_applied:
            return input_path
        file_name_w_o_extension = Path(input_path).name.split(".")[0] + ".wav"
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
    _df = (df[["path_input", "path_output", "path_noise", "filter_job"]])[df.filter_job != ""]
    _paths_in = _df["path_input"]
    _paths_out = _df["path_output"]
    _filter_jobs = _df["filter_job"]
    _path_noise = _df["path_noise"]

    return zip(_paths_in, _path_noise, _paths_out, _filter_jobs)


def random_rate(_range):
    _from, _to = _range
    return lambda: uniform(_from, _to)


def callback_dict(filter_settings, target_sample_rate=16_000):
    range_fn_mapping = {}
    for key, value in filter_settings.items():
        _range = value["range"]
        range_fn_mapping[key] = random_rate(_range)

    snr_fn = random_rate(filter_settings["real_noise"]["range"])

    def __add_real_noise(idx, y_path, rate, y_n_path):
        yp, _ = IO.load(y_path, sample_rate=target_sample_rate)
        yn, _ = IO.load(y_n_path, sample_rate=target_sample_rate)

        return Effect.add_noise(yp, yn, snr=snr_fn(), pad_idx=idx)

    job_fn_mapping = {
        "time_stretch": lambda __, y, rate, _: Effect.time_stretch(y, rate),
        "harmonic_remove": lambda __, y, rate, _: (y - 0.5 * librosa.effects.harmonic(y, margin=rate)),
        "percussive_remove": lambda __, y, rate, _: librosa.effects.percussive(y, margin=rate),
        "random_noise": lambda __, y, rate, _: Effect.add_noise_random(y, rate),
        "real_noise": lambda __, y, rate, y_n: __add_real_noise,
        "reverb": lambda __, y, rate, _: AudioEffectsChain().reverb(reverberance=rate, hf_damping=rate, room_scale=rate * 2)(
            y),
        "bandpass": lambda __, y, rate, _: AudioEffectsChain().bandpass(rate)(y),
        "tremolo": lambda __, y, rate, _: AudioEffectsChain().tremolo(rate)(y),
    }

    assert False not in [key in job_fn_mapping.keys() for key in filter_settings.keys()], "Uknown Filter-Option"

    return job_fn_mapping


def save_settings(df, output_dir, filter_settings, file_name="data.csv", **kwargs):
    sep = kwargs.get("sep", ";")
    df_path = Path(output_dir, file_name)
    json_path = Path(output_dir, "filter_settings.json")
    df.to_csv(df_path.resolve(), sep=sep, encoding="utf8", index=False)

    settings_json = json.dumps(filter_settings, indent=4)
    with open(json_path, "w") as f:
        f.write(settings_json)


def build_job_df(df, noise_df, filter_settings, output_dir, output_subfolder=None, **kwargs):
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    df = add_real_noise_column(df, noise_df)
    df = df.rename({'path': 'path_input'}, axis=1)
    df = add_filter_job_column(df, filter_settings)
    df = add_output_path_column(df, output_dir, output_subfolder)

    save_settings(df, output_dir, filter_settings, **kwargs)

    return df

# time_stretch+harmonic_remove+percussive_remove+randon_noise+realse_noise+reverb+bandpass+tremolo

# /share/datasets/vf_de/guenter-20140125-ftr/wav/de5-026.wav
