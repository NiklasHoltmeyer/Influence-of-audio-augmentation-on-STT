import torchaudio


class LoadandResampleAudio:
    def __init__(self, processor, **kwargs):
        self.processor = processor
        self.speech_key = kwargs.get("speech_key", "speech")
#        self.sentence_key = kwargs.get("sentence_key", "sentence")

        self.sample_rate_in = kwargs.get("sample_rate_in", None)
        self.sample_rate_out = kwargs.get("sample_rate_out", None)

        self.resampler = torchaudio.transforms.Resample(self.sample_rate_in,
                                                        self.sample_rate_out) if self.sample_rate_in and self.sample_rate_out else None

    def __call__(self, batch):
        waveform, sampling_rate = torchaudio.load(batch[self.speech_key])
        #batch["speech"] = speech_array[0].numpy()
        #batch["sampling_rate"] = sampling_rate
        #batch["target_text"] = batch["text"]

        if self.resampler:
            waveform = self.resampler(waveform)
            batch[self.speech_key] = waveform[0].numpy()
            batch["sampling_rate"] = self.sample_rate_out
        else:
            batch[self.speech_key] = waveform[0].numpy()
            batch["sampling_rate"] = sampling_rate

        return batch

