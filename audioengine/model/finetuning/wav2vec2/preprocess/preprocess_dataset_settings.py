from audioengine.logging.logging import defaultLogger

logger = defaultLogger()

def preprocess_settings():
    return preprocess_settings_cv_random_aug_rn()#preprocess_settings_cv_realnoise_aug()
    #return preprocess_settings_eval()
#"/share/datasets/8mil_tts/"

def preprocess_settings_cv_ttsaug():
    return {
        "val_settings": [
            {
                "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
                "shuffle": True,
                "validation_split": None,
                "type": "test_small",
                "min_duration": 1.5,
                "max_duration": 6.0,
                "min_target_length": 2,
                "max_target_length": None
            }
        ],
        "train_settings": [
            {
                "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
                "shuffle": True,
                "validation_split": None,
                "type": "train_small",
                "min_duration": 1.5,
                "max_duration": 6.0,
                "min_target_length": 2,
                "max_target_length": None
            },
            {
                "base_path": "/share/datasets/8mil_tts/",
                "shuffle": True,
                "validation_split": None,
                "type": "filter_trim",
                "fixed_length": 15_000,
                #"min_duration": 1.5,
                #"max_duration": 6.0,
                "min_target_length": 2,
                "max_target_length": None,
                "filter": ['-af "highpass=200,lowpass=3000,afftdn"',
                           '-af "silenceremove=start_periods=1:start_duration=1:start_threshold=-50dB:detection=peak,'
                           'aformat=dblp,areverse,'
                           'silenceremove=start_periods=1:start_duration=1:start_threshold=-50dB:detection=peak,'
                           'aformat=dblp,areverse"'],
                "tts_engine": "silero"
            }
        ],
    }

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

def preprocess_settings_cvmd_no_aug():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,
#        "fixed_length": None, # -> 20% f. train
        "type": "test_medium",
        "min_duration": 1.5,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_medium",
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

def preprocess_settings_cv_random_aug_ffmpeg():
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
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length_aug = {
        "base_path": "/share/datasets/cv_sm_harmonic",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train_small",
        "filter_settings": {
            "harmonic_remove": {
                "range": (1, 5),  # margin_range
                "probability": 1.0
            },
        }

    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
            cv_train_fixed_length_aug,
        ]
    }

    return test_settings


def preprocess_settings_cv_random_nnn():
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
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length_aug_1 = {
        "base_path": "/share/datasets/cv_sm_noise_mix/random",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train_small",
        "filter_settings": {
            "random_noise": {
                "range": (0.98, 0.99),  # PSNR range #<- still *quiet* loud!
                "probability": 1.00,
                "info": "50% cv_sm"
            },
        }
    }

    cv_train_fixed_length_aug_2 = {
        "base_path": "/share/datasets/cv_sm_noise_mix/real",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train_small",
        "filter_settings": {
            "real_noise": {
                "range": (0.6, 0.9),
                "probability": 1.00,
                "info": "50% cv_sm"
            },
        }

    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
            cv_train_fixed_length_aug_1,
            cv_train_fixed_length_aug_2
        ]
    }

    return test_settings

def preprocess_settings_cv_random_aug_rn():
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
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    cv_train_fixed_length_aug = {
        "base_path": "/share/datasets/cv_sm_harmonic",
        "shuffle": True,
        "validation_split": None,  # -> all entries
        "type": "train_small",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train_small",
        "filter_settings": {
            "harmonic": {
                "range": (1, 5),  # margin_range
                "probability": 1.0
            },
        }
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
            cv_train_fixed_length_aug,
        ]
    }

    return test_settings

def preprocess_settings_TEST_TEST_TEST():
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": True,
        "validation_split": None,
        "fixed_length": 32, # -> 20% f. train
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
        "type": "train_small",
        "min_duration": 1.50,
        "fixed_length": 32,
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