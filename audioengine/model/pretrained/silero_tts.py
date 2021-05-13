import torch

from audioengine.logging.logging import defaultLogger


class SileroTTS:
    def __init__(self, language, speaker, device):
        device = torch.device(device)
        self.model, self.symbols, self.sample_rate, example_text, self.apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                                   model='silero_tts',
                                                                                   language=language,
                                                                                   speaker=speaker)
        self.model = self.model.to(device)

    def apply(self, sentences):
        return self.apply_tts(texts=sentences,
                  model=self.model,
                  sample_rate=self.sample_rate,
                  symbols=self.symbols,
                  device=self.device)
