import torch


# from audioengine.transformations.backend.pytorch.audiotransformations import LoadAudio
# from audioengine.transformations.backend.pytorch.texttransformations import Regexp
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audioengine.transformations.backend.pytorch.audiotransformations import LoadAudio
from audioengine.transformations.backend.pytorch.texttransformations import ToLower


class Silero:
    def __init__(self, language, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.language = language
        self.device = device

        self.model, self.decoder, self.read_batch, self.split_into_batches, \
            self.read_audio, self.prepare_model_input = self._load_pretrained(device, language)

    def predict(self, speeches):
        with torch.no_grad():
            inputs = self.prepare_model_input(speeches, device=self.device)
            outputs = self.model(inputs)
            transcripts = [self.decoder(example.cpu()) for example in outputs]
            return transcripts

    def transformations(self, input_sample_rate=48_000, output_sample_rate=16_000):
        return [ToLower("sentence"), LoadAudio(input_sample_rate, output_sample_rate, False)]

    def _load_pretrained(self, device, language):
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language=language,  # 'de', 'es', 'en'
                                               device=device)
        (read_batch, split_into_batches,
         read_audio, prepare_model_input) = utils
        return model, decoder, read_batch, split_into_batches, read_audio, prepare_model_input

    @staticmethod
    def __model_list():
        languagees = ['de', 'es', 'en']
        return {
            'languagees': languagees
        }

    def __str__(self):
        infos = {
            "model_name": "Silero-"+self.language,
            "backend": "pytorch"
        }
        return str(infos)

# class wav2vec2:
#    def predict(self, speeches, sampling_rate=16_000, padding=True):
#        inputs = self.processor(speeches, sampling_rate=sampling_rate, return_tensors="pt", padding=padding)

#        with torch.no_grad():
#            logits = self.model(inputs.input_values.to(self.device),
#                                attention_mask=inputs.attention_mask.to(self.device)).logits

#        pred_ids = torch.argmax(logits, dim=-1)
#        transcriptions = self.processor.batch_decode(pred_ids)
#        return transcriptions





