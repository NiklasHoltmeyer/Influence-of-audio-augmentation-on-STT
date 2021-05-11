from random import randint

import librosa
import numpy as np


class Effect:
    @staticmethod
    def add_noise(y: np.ndarray, y_noise: np.ndarray, snr: float, pad_idx: int=1) -> np.ndarray:
        """
Apply Noise-Signal to Signal.
        Args:
            y: numpy.darray
                Base-Signal
            y_noise: numpy.darray
                Noise-Signal
            snr: int
                Signal-Noise_ration
            pad_idx: Int
                pad_predefined =["constant" , "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap"]
                Default = 1 -> Edge (see https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
            **kwargs: **kwargs
                Additional Parameter
        Returns:
            y_t = y*snr + y_noise * (1-snr)
                : numpy.darray [Time-Series]
        """
        s_len, n_len = len(y), len(y_noise)
        dif = s_len - n_len

        if dif < 0:
            beg = randint(0, -dif)# -> end <= 0
            end = beg+s_len

            #beg = randint(dif, 0) #int(-dif / 2)
            #end = -((-1 * dif) - beg)

            #beg, end = min(beg, end), max(beg, end)
            y_noise_padded = y_noise[beg: end]
            if len(y_noise_padded) == 0:
                print(end)
                print(beg)
                exit(0)

        elif dif > 0:
            pad_fn = pad_predefined[pad_idx % 10]
            beg = randint(0, dif) # int(dif / 2)
            end = dif - beg
            y_noise_padded = np.pad(y_noise, (beg, end), pad_fn)
        else:
            y_noise_padded = y_noise
        return y * snr + (1 - snr) * y_noise_padded

    @staticmethod
    def add_noise_random(y: np.ndarray, snr: int) -> np.ndarray:
        """
Add Random Noise to Signal.
        Args:
            y: np.ndarray
                Signal
            snr: int
                Signal-Noise_ration
        Returns:
            y_t = y + ratio * y_rnd_noise
        """
        noise = np.random.randn(len(y))
        return y*snr + noise*(1-snr)

    @staticmethod
    def time_stretch(y: np.ndarray, rate: float, **kwargs: object) -> np.ndarray:
        """
Time-Stretch Effect
        Args:
            y: np.ndarray
                Signal
            rate: float
            **kwargs: **kwargs

        Returns:
            y_t: np.ndarray
                (Time-)Stretched Singal
        """
        return librosa.effects.time_stretch(y, rate)

    @staticmethod
    def pitch_shift(y: np.ndarray, sample_rate: int, n_steps: int, **kwargs) -> np.ndarray:
        """
Shift Pitch of a Signal.
See :func:`my text <librosa.effects.pitch_shift>`
        Args:
            y: np.ndarray
                Signal
            sample_rate: int
            n_steps: int

            **kwargs: **kwargs
                See librosa.effects.pitch_shift

        Returns:
            y_t : np.ndarray
        """
        return librosa.effects.pitch_shift(y,
                                           sample_rate,
                                           n_steps,
#                                           bins_per_octave=kwargs.get("bins_per_octave", 12),
#                                           res_type=kwargs.get("res_type", 'kaiser_best'),
                                           **kwargs)
pad_predefined = ["constant", "edge", "linear_ramp", "maximum", "mean",
                  "median", "minimum", "reflect", "symmetric", "wrap"]
