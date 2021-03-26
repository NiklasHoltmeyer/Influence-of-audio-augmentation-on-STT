import librosa
import numpy as np


class Effect:
    @staticmethod
    def add_noise(y: np.ndarray, y_noise: np.ndarray, ratio: float, **kwargs: object) -> np.ndarray:
        """
Apply Noise-Signal to Signal.
        Args:
            y: numpy.darray
                Base-Signal
            y_noise: numpy.darray
                Noise-Signal
            ratio: int
                Ratio
            **kwargs: **kwargs
                Additional Parameter
        Returns:
            y_t = y + y_noise * ratio
                : numpy.darray [Time-Series]
        """
        signal_length_dif = y.shape[0] - y_noise.shape[0]
        if signal_length_dif > 0:
            y_noise = np.pad(y_noise, (0, signal_length_dif), 'constant', constant_values=(0, 0))
        elif signal_length_dif < 0:
            y = np.pad(y, (0, -signal_length_dif), 'constant', constant_values=(0, 0))

        return y + ratio * y_noise

    @staticmethod
    def add_noise_random(y: np.ndarray, ratio: float) -> np.ndarray:
        """
Add Random Noise to Signal.
        Args:
            y: np.ndarray
                Signal
            ratio: float
                Ratio
        Returns:
            y_t = y + ratio * y_rnd_noise
        """
        noise = np.random.randn(len(y))
        return y + ratio * noise

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
