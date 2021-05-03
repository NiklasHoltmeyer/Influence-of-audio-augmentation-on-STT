import json

from audioengine.logging.logging import defaultLogger

logger = defaultLogger()

def preprocess_settings():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,
        "fixed_length": 4500, # -> 20% f. train
        "type": "test",
        "min_duration": 1.00,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "fixed_length": 4000,  # -> 80% vf + 20% cv_train
        "type": "train",
        "min_duration": 1.00,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    vf_full = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        "min_duration": 1.00,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
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