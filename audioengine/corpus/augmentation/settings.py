filter_settings = {
    "real_noise": {
        "range": (0.6, 0.9), #0.5 geht fitt
        "probability": 1.0,
    }
}
filter_settings_all = {
    "time_stretch": {
        "range": (0.8, 1.5),
        "probability": 0.3
    },
    "harmonic_remove": {
        "range": (1, 5),  # margin_range
        "probability": 0.1
    },
    "percussive_remove": {
        "range": (1, 5),  # margin
        "probability": 0.1
    },
    "random_noise": {
        "range": (0.98, 1),  # PSNR range #<- still *quiet* loud!
        "probability": 0.01
    },
    "real_noise": {
        "range": (0.15, 0.45),
        "probability": 0.15,
    },
    "reverb": {
        "range": (5, 50),
        "probability": 0.05,
    },
    "bandpass": {
        "range": (0, 1000),
        "probability": 0.05
    },
    "tremolo": {
        "range": (100, 8000),
        "probability": 0.05
    }
}

assert not False in [0 <= filter_settings[key]["probability"] <= 1 for key in
                     filter_settings.keys()], "Probability Range = 0..1"
