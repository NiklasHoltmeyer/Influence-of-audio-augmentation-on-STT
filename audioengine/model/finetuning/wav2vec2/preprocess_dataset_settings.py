import json

from audioengine.logging.logging import defaultLogger

logger = defaultLogger()

def preprocess_settings():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "test"
    }

    cv_train_fixed_length = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "fixed_length": 24683 / 2,  # -> 75% vf + 25% cv_train
        "type": "train"
    }

    vf_full = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
            vf_full
        ]
    }

    return test_settings


def save_settings(path, settings, infos=None, indent=4):
    """

    Args:
        path: Path
        settings: dict
        infos: Array of KVPs
    """

    infos = [] if not infos else infos

    for key, value in infos:
        settings[key] = value

    settings_json = json.dumps(settings, indent=indent)

    with open(path, "w") as f:
        f.write(settings_json)

    logger.debug(f"Saved Dataset-Settings to {path}")