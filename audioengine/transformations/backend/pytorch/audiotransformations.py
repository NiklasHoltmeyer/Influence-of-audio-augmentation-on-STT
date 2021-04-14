import torch
import torchaudio
from torch import Tensor
from typing import Callable, Optional


class LoadAudio(object):
    def __init__(self, input_sr=None, output_sr=None):
        self.sr = output_sr
        self.sr_default = input_sr
        self.resampler = torchaudio.transforms.Resample(input_sr, output_sr) if input_sr and output_sr else None

    def __call__(self, data):
        waveform, _ = torchaudio.load(data['path'])
        waveform = waveform.squeeze(0).squeeze(0)
        if self.resampler:
            waveform = self.resampler(waveform)
            data["sampling_rate"] = self.sr
        else:
            data["sampling_rate"] = self.sr_default

        data["speech"] = waveform.numpy()
        return data


class Spectrogram:
    def __init__(self, n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True):
        self.transform = torchaudio.transforms.Spectrogram(n_fft, win_length, hop_length, pad, window_fn, power,
                                                           normalized, wkwargs, center, pad_mode, onesided)

    def __call__(self, data):
        data["spectrogram"] = self.transform.forward(data["speech"])
        return data

    def forward(self, waveform: Tensor) -> Tensor:
        return self.transform.forward(waveform)
