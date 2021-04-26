from pathlib import Path

from audioengine.logging.logging import defaultLogger

ds_type = "dev"  # common_voice

settings = {
    "shuffle": True,
    "split": 0.5,
    "load_augmended_info": True,
    "load_clean_info": False,
    "signal_path": "C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de",
    "noise_path": "C:\workspace\datasets\FSD50K\FSD50K.dev_audio\FSD50K.dev_audio",
    "target_sample_rate": 16_000,
    "target_path": "C:\workspace\datasets\cv_augmented\clips",
    "target_file_name": f"..\{ds_type}.tsv",
    "sep": "\t"
}


def sanity_check_settings(**settings):
    assert 0 <= settings.get("split", None) <= 1, "Split must be e [0, 1]"
    assert Path(settings.get("signal_path")).exists(), "signal_path does not exist!"
    assert Path(settings.get("noise_path")).exists(), "noise_path does not exist!"

    if 8_000 <= settings.get("target_sample_rate") <= 16_000:
        logger.warning(f"target_sample_rate:= {settings.get('target_sample_rate')}"
                       f" - change this Value if the Value was a mistake!")

    assert len(settings.get("sep")) > 0, "Invalid Value for sep. (CSV-Separator)"


logger = defaultLogger()
sanity_check_settings(settings)
