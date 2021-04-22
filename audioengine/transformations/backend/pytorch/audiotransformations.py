import torchaudio


class LoadAudio(object):
    def __init__(self, input_sr=None, output_sr=None, to_numpy=True):
        self.sr = output_sr
        self.sr_default = input_sr
        self.resampler = torchaudio.transforms.Resample(input_sr, output_sr) if input_sr and output_sr else None
        self.to_numpy = to_numpy

    def __call__(self, data):
        waveform, _ = torchaudio.load(data['path'])
        waveform = waveform
        if self.resampler:
            waveform = self.resampler(waveform)
            data["sampling_rate"] = self.sr
        else:
            data["sampling_rate"] = self.sr_default

        data["speech"] = waveform.squeeze(0).numpy() if self.to_numpy else waveform.squeeze(0)
        return data


class PreprocessTransformer:
    def __init__(self, processor, sampling_rate, padding=True):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.padding = padding

    def __call__(self, batch):
        batch["input_values"] = self.processor(batch["speech"], sampling_rate=self.sampling_rate).input_values
        #return_tensors="pt", sampling_rate=self.sampling_rate, padding=self.padding
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["sentence"]).input_ids
            #batch["labels"] = self.processor(batch["sentence"], return_tensors="pt", padding=self.padding).input_ids
        return batch

##class Spectrogram:
##    def __init__(self, n_fft: int = 400,
##                 win_length: Optional[int] = None,
##                 hop_length: Optional[int] = None,
##                 pad: int = 0,
##                 window_fn: Callable[..., Tensor] = torch.hann_window,
##                 power: Optional[float] = 2.,
##                 normalized: bool = False,
##                 wkwargs: Optional[dict] = None,
##                 center: bool = True,
##                 pad_mode: str = "reflect",
##                 onesided: bool = True):
##        self.transform = torchaudio.transforms.Spectrogram(n_fft, win_length, hop_length, pad, window_fn, power,
##normalized, wkwargs, center, pad_mode, onesided)

##    def __call__(self, data):
##        data["spectrogram"] = self.transform.forward(data["speech"])
##        return data

##    def forward(self, waveform: Tensor) -> Tensor:
##        return self.transform.forward(waveform)
