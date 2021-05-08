from audioengine.logging.logging import defaultLogger

logger = defaultLogger()

def preprocess_settings():
    return None
    #return preprocess_settings_eval()

def preprocess_settings_vf_vf():
    vf_full_train = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        "min_duration": 1.00,
        #        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "type": "train"
    }
    vf_full_test = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        "min_duration": 1.00,
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

def preprocess_settings_vfPcv_cv_duration_truncatation():
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
#        "fixed_length": 1,  # -> 80% vf + 20% cv_train
        "type": "train",
        "min_duration": 1.50,
        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None
    }

    vf_full = {
       "base_path": "/share/datasets/vf_de",
       "shuffle": True,
       "validation_split": None,  # -> all entries,
       "min_duration": 0.85,
#        "max_duration": 6.00,
       "min_target_length": 2,
       "max_target_length": None,
       "desc": "test"
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            cv_train_fixed_length,
            vf_full
        ]
    }

    return test_settings

def preprocess_settings_vfPcv_cv_full_full_no_aug():
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

    vf_full_test = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        "min_duration": 0.85,
#        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
    }


    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
            vf_full_test,
        ]
    }

    return test_settings


def preprocess_settings_vfPcv_cv_full_full():
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

#    cv_train_fixed_length = {
#        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
#        "shuffle": True,
#        "validation_split": None,  # -> all entries
##        "fixed_length": 1,  # -> 80% vf + 20% cv_train
#        "type": "train",
#        "min_duration": 0.85,
##        "max_duration": 6.00,
#        "min_target_length": 2,
#        "max_target_length": None
#    }

    vf_full_test = {
        "base_path": "/share/datasets/vf_de",
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        "min_duration": 0.85,
#        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "desc": "clean-full"
    }

    vf_augmented = {
        "base_path": "/share/datasets/vf_augment_train", #\vf_augment_train
        "shuffle": True,
        "validation_split": None,  # -> all entries,
        "min_duration": 0.85,
        #        "max_duration": 6.00,
        "min_target_length": 2,
        "max_target_length": None,
        "desc": "train-augmented"
    }

    test_settings = {
        "val_settings": [cv_test_full],
        "train_settings": [
#            cv_train_fixed_length,
            vf_full_test,
            vf_augmented
        ]
    }

    return test_settings


