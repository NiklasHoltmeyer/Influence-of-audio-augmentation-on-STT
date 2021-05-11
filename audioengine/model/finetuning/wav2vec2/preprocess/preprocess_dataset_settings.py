from audioengine.logging.logging import defaultLogger

logger = defaultLogger()

def preprocess_settings():
    return preprocess_settings_cv_realnoise_aug()
    #return preprocess_settings_eval()

def preprocess_settings_cv_eval():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,
#        "fixed_length": None, # -> 20% f. train
        "type": "test",
        "min_duration": 0.85,
#        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "fixed_length": 32,  # -> 80% vf + 20% cv_train
        "type": "train",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length
        ]
    }

    return test_settings

def preprocess_settings_cv_no_aug():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,
#        "fixed_length": None, # -> 20% f. train
        "type": "test_small",
        "min_duration": 1.5,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "fixed_length": 15_000,  # -> 80% vf + 20% cv_train
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
        ]
    }

    return test_settings

def preprocess_settings_cv_realnoise_aug():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,
#        "fixed_length": None, # -> 20% f. train
        "type": "test_small",
        "min_duration": 1.5,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "fixed_length": 15_000,  # -> 80% vf + 20% cv_train
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length_aug = {
        "base_path": "/share/datasets/cv/de/cv_small_real_noise/",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "fixed_length": 15_000,  # -> 80% vf + 20% cv_train
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
            cv_train_fixed_length_aug,
        ]
    }

    return test_settings


def preprocess_settings_vf_vf():
    vf_full_train = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        #"min_duration": 1.00,
        #        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train"
    }
    vf_full_test = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        #"min_duration": 1.00,
        #        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "test"
    }
    test_settings = {
        "val_settings": [vf_full_test],
        "train_settings": [
            vf_full_train
        ]
    }
    return test_settings