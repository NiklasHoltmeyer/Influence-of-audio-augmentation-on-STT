from pathlib import Path

import librosa
import pandas as pd
import swifter
from sklearn.model_selection import train_test_split
import numpy as np
from audioengine.transformations.backend.librosa.effect import Effect
from audioengine.transformations.backend.librosa.io import IO
from tqdm.auto import tqdm

## Pandas-Dataframe
tqdm.pandas()


def augment_dataframe(signal_df, noise_df, **settings):
    shuffle = settings.get("shuffle", True)
    data_augmented, data_clean = init_df(signal_df, **settings)
    noise = init_df(noise_df, skip_split=True, **settings)
    data_augmented = add_noise_column(data_augmented, noise)
    data_augmented["path_augmented"] = data_augmented["path"].apply(get_target_column(**settings))
    data_augmented = data_augmented.swifter.apply(merge_sounds(**settings), axis=1)

    rename = {"path": "path_source", "path_augmented": "path"}
    data_augmented.rename(columns=rename, inplace=True)
    data_augmented = data_augmented[["sentence", "path", "snr", "duration", 'path_source', 'path_noise']]

    data_clean = clean_df_add_cols(data_clean)

    return pd.concat([data_augmented, data_clean]).sample(frac=1).reset_index(drop=True) if shuffle \
        else pd.concat([data_augmented, data_clean]).reset_index(drop=True)


def init_df(df, **kwargs):
    split = kwargs.get("split", 0.5)
    skip_split = kwargs.get("skip_split", False)

    assert 0 <= split <= 1, "Split must be between 0 and 1"

    ##del_me
    df = pd.DataFrame({"path": df.path[:10], "sentence": df.sentence[:10]})
    ##del ende

    if skip_split:
        df["duration"] = df['path'].swifter.apply(IO.load_duration)
        return df.sort_values(by=['duration'])  # .reset_index(inplace=True)

    data_augmented, data_clean = train_test_split(df, test_size=1 - split, random_state=42)

    data_augmented["duration"] = data_augmented['path'].swifter.apply(IO.load_duration)

    # data_augmented, data_clean = data_augmented.reset_index(inplace=True), data_clean.reset_index(inplace=True)

    return data_augmented, data_clean


def add_noise_column(signal_df, noise_df):
    len_dif = len(signal_df) - len(noise_df)

    if len_dif < 0:  # -> s_df < n_df
        noise_df = noise_df[: len(signal_df)]["path"]
    elif len_dif > 0:  # -> s_df > n_df
        beg = int(len_dif / 2)
        end = len_dif - beg
        noise_df = np.pad(noise_df["path"], (beg, end), 'symmetric')
    else:
        noise_df = noise_df["path"]

    # assert len(signal_df) == len(noise_df), "Padding (Dataframes) failed"
    signal_df["path_noise"] = noise_df
    return signal_df


from random import uniform


def merge_sounds(**settings):
    _range = settings.get("snr_range", (0.15, 0.75))
    assert len(_range), "snr_range -> e.g. (0.15, 0.75)"
    target_sample_rate = settings.get("target_sample_rate", "16000")

    if "target_path" not in settings.keys():
        raise Exception("please Specify target_path in Settings-Dict")
    target_path = Path(settings["target_path"])
    target_path.mkdir(parents=True, exist_ok=True)

    def __call__(item):
        snr = round(uniform(_range[0], _range[1]), 4)
        yp, _ = IO.load(item["path"], sample_rate=target_sample_rate)
        yn, _ = IO.load(item["path_noise"], sample_rate=target_sample_rate)
        item["snr"] = snr
        target_path = item["path_augmented"]
        y_augmented = Effect.add_noise(yp, yn, snr=snr, pad_idx=-2)
        IO.save_wav(y_augmented, target_path, target_sample_rate)
        return item

    return __call__


def get_target_column(**settings):
    target_path = settings.get("target_path", None)
    assert target_path, "Please specify target_path"

    def __call__(item):
        target_name = Path(item).name.replace(".mp3", ".wav")
        return str(Path(target_path, target_name).resolve())

    return __call__


def clean_df_add_cols(df):
    len_clean = len(df)
    df["snr"] = [1.0] * len_clean
    df["duration"] = [-1] * len_clean
    df["path_source"] = [-1] * len_clean
    df["path_noise"] = [-1] * len_clean
    df = df[["sentence", "path", "snr", "duration", 'path_source', 'path_noise']]
    return df
