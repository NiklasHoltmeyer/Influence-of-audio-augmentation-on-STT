import logging
from pathlib import Path
import sys
# from audioengine.logging.logging import defaultLogger
from audioengine.corpus.augmentation.real_noise.helper import augment_dataframe, save_df
from audioengine.corpus.voxforge import VoxForge

from audioengine.corpus.commonvoice import CommonVoice
from audioengine.corpus.noise import Noise
from audioengine.logging.logging import defaultLogger


def sanity_check_settings(**settings):
    assert 0 <= settings.get("split", None) <= 1, "Split must be e [0, 1]"
    assert Path(settings.get("signal_path")).exists(), "signal_path does not exist!"
    assert Path(settings.get("noise_path")).exists(), "noise_path does not exist!"
    assert len(settings.get("sep")) > 0, "Invalid Value for sep. (CSV-Separator)"


def load_dataframes(**settings):
    signal_path = settings["signal_path"].lower()
    noise_path = settings["noise_path"]

    noise_df = Noise(noise_path).load_dataframe()

    if "cv" in signal_path or "commonvoice" in signal_path:
        ds_type = settings.get("signal_split")
        signal_df = CommonVoice(signal_path).load_dataframe(type=ds_type)

        return signal_df, noise_df

    if "vf" in signal_path or "voxforge" in signal_path:
        signal_df = VoxForge(signal_path).load_dataframe()
        return signal_df, noise_df

    raise Exception("Unknown Dataset. Supported Signals-Datasets -> CommonVoice, VoxForge")


def main():
    logger = defaultLogger()

    ds_type = "dev"  # common_voice

    settings = {
        "shuffle": True,
        "split": 0.5,
        "signal_path": "C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de",
        "noise_path": "C:\workspace\datasets\FSD50K\FSD50K.dev_audio\FSD50K.dev_audio",
        "signal_split": ds_type,
        "target_sample_rate": 16_000,
        "target_path": "C:\workspace\datasets\cv_augmented\clips",
        "target_file_name": f"..\{ds_type}.tsv",
        "sep": "\t"
    }

    sanity_check_settings(**settings)

    signal_df, noise_df = load_dataframes(settings)

    augmented_df = augment_dataframe(signal_df, noise_df, **settings)
    save_df(augmented_df)
