from pathlib import Path

import numpy as np

from audioengine.corpus.dataset import logger


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
    return signal_df


def add_output_path_column(df, output_dir, subfolder=None):
    output_path = Path(output_dir, subfolder) if subfolder else Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = str(output_path.resolve())

    def _calc_output_path(input_path):
        file_name_w_o_extension = Path(input_path).name.split(".")[0] + ".wav"
        return str(Path(output_path, file_name_w_o_extension).resolve())

    df["path_output"] = df["path_input"].map(_calc_output_path)
    return df

def build_job_df(df, noise_df, filter_settings, output_dir, output_subfolder=None):
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    df = add_real_noise_column(df, noise_df)
    df = df.rename({'path': 'path_input'}, axis=1)
    df = add_filter_job_column(df, filter_settings)
    df = add_output_path_column(df, output_dir, output_subfolder)

    return df



#time_stretch+harmonic_remove+percussive_remove+randon_noise+realse_noise+reverb+bandpass+tremolo

#/share/datasets/vf_de/guenter-20140125-ftr/wav/de5-026.wav