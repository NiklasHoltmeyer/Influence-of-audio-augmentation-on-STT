import librosa
import numpy as np
from sklearn.preprocessing import scale


class Signal:
    @staticmethod
    def mfcc(signal: np.ndarray, **kwargs) -> np.ndarray:
        """
Calc MFCC
        Args:
            signal: np.ndarray

            **kwargs:

            ** librosa.feature.mfcc ** (https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)
                sr: Sample_Rate
                    8_000 [default]
                S:
                    None [default]
                n_mfcc:
                    20 [default]
                dct_type:
                    2 [default]
                norm:
                    'ortho' [default]
                lifter:
                    0 [default]

            ** librosa.feature.melspectrogram ** (https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram)
                n_fft:
                    2048 [default]
                hop_length:
                    512 [default]
                win_Length:
                    None [default]
                window:
                    'hann' [default]
                center:
                    True [default]
                pad_mode:
                    'reflect' [default]
                power:
                    2.0 [default]

            ** librosa.filters.mel ** (https://librosa.org/doc/main/generated/librosa.filters.mel.html#librosa.filters.mel)
                n_mels:
                    128 [default]
                fmin:
                    0.0 [default]
                fmax:
                    None [default]
                htk:
                    False [default]
                norm:
                    'slaney' [default]
                dtype:
                    <class 'numpy.float32'> [default]
        Returns:
            MFCC: numpy.ndarray
        """
        return librosa.feature.mfcc(y=signal,
                                    sr=kwargs.get("sr", 8_000),
                                    S=kwargs.get("S", None),
                                    n_mfcc=kwargs.get("n_mfcc", 20),
                                    dct_type=kwargs.get("dct_type", 2),
                                    norm=kwargs.get("norm", 'ortho'),
                                    lifter=kwargs.get("lifter", 0),
                                    **kwargs)

    @staticmethod
    def mfcc_inverse(mfcc: np.ndarray, **kwargs) -> np.ndarray:
        return librosa.feature.inverse.mfcc_to_audio(mfcc,
                                                     n_mels=kwargs.get("n_mels", 128),
                                                     dct_type=kwargs.get("dct_type", 2),
                                                     norm=kwargs.get("norm", 'ortho'),
                                                     ref=kwargs.get("ref", 1.0),
                                                     lifter=kwargs.get("lifter", 0),
                                                     **kwargs)

    @staticmethod
    def normalize(y: np.ndarray) -> np.ndarray:
        """
Normalize Signal.
        Args:
            y: np.ndarray
                Signal

        Returns:
            y_t = normalize(y): np.ndarray
        """
        return scale(y, axis=0, with_mean=True, with_std=True, copy=True)
