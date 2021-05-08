import torch

from audioengine.logging.logging import defaultLogger
from audioengine.model.finetuning.wav2vec2.helper.wav2vec2_trainer import DataCollatorCTCWithPadding
from audioengine.transformations.backend.pytorch.audiotransformations import LoadAudio
from audioengine.transformations.backend.pytorch.texttransformations import Regexp, ToLower
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audioengine.transformations.backend.pytorch.texttransformations import ToUpper
from torchvision import transforms


class wav2vec2:
    def __init__(self, model_name, based_on=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), skip_loading=False):
        self.model_name = model_name
        self.device = device
        self.based_on = based_on

        self.logger = defaultLogger()
        self.logger.info(f"Wav2Vec Device: {device}")
        if not skip_loading:
            self.model, self.processor = self._load_pretrained()
        else:
            self.logger.warning("skip_loading is set! No Model loaded!")

    def predict(self, speeches, sampling_rate=16_000, padding=True):
        inputs = self.processor(speeches, sampling_rate=sampling_rate, return_tensors="pt", padding=padding)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(pred_ids)
        return transcriptions

    def transformations(self, input_sample_rate=48_000, output_sample_rate=16_000, **kwargs):
        transformations = [ToLower("sentence")]

        chars_to_ignore_regex = kwargs.get("chars_to_ignore_regex", self._chars_to_remove())

        if not chars_to_ignore_regex:
            raise Exception("Unknown Chars to Ignore.")

        regexp_subs = [("’", "'"), (chars_to_ignore_regex, '')] if chars_to_ignore_regex else []

        replacements = self._chars_to_replace()
        if replacements:
            for key, value in replacements.items():
                # replace (value, key)
                regexp_subs.append((value, key))

        if chars_to_ignore_regex:
            regexp_layer = Regexp(regexp_subs)
            transformations.append(regexp_layer)

        transformations.append(LoadAudio(input_sample_rate, output_sample_rate))

        if self.model_name == "flozi00/wav2vec-xlsr-german":
            transformations[0] = ToUpper("sentence")
        return transformations

    def transformation(self, input_sample_rate=48_000, output_sample_rate=16_000, **kwargs):
        transformations = self.transformations(input_sample_rate=input_sample_rate,
                                               output_sample_rate=output_sample_rate, **kwargs)
        return transforms.Compose(transformations)

    def _load_pretrained(self):
        processor_name = self.model_name if not self.based_on else self.based_on
        processor = Wav2Vec2Processor.from_pretrained(processor_name)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        model = model.to(self.device)
        return model, processor

    def data_collator(self):
        return DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def __str__(self):
        infos = {
            "model_name": self.model_name,
            "backend__": "pytorch"
        }
        return str(infos)

    def _chars_to_remove(self):
        mappings = {
            'facebook/wav2vec2-large-xlsr-53-german': '[\,\?\.\!\-\;\:\"]',
            'maxidl/wav2vec2-large-xlsr-german': '[\\,\\?\\.\\!\\-\\;\\:\\"\\“]',
            'marcel/wav2vec2-large-xlsr-53-german': '[\,\?\.\!\-\;\:\"\“\%\”\�\カ\æ\無\ན\カ\臣\ѹ\…\«\»\ð\ı\„\幺\א\ב\比\ш\ע\)\ứ\в\œ\ч\+\—\ш\‚\נ\м\ń\乡\$\=\ש\ф\支\(\°\и\к\̇]',
            'flozi00/wav2vec-xlsr-german': '[\\\\,\\\\?\\\\.\\\\!\\\\-\\\\;\\\\:\\\\"\\\\“\\\\%\\\\‘\\\\”\\\\�]',
            "marcel/wav2vec2-large-xlsr-german-demo": '[\,\?\.\!\-\;\:\"\“\%\”\�\カ\æ\無\ན\カ\臣\ѹ\…\«\»\ð\ı\„\幺\א\ב\比\ш\ע\)\ứ\в\œ\ч\+\—\ш\‚\נ\м\ń\乡\$\=\ש\ф\支\(\°\и\к\̇]',
            'MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German': '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
        }
        if self.based_on:
            return mappings.get(self.based_on, None)
        return mappings.get(self.model_name, None)

    def _chars_to_replace(self, _else=None):
        substitutions_marcel = {
            'e': '[\ə\é\ě\ę\ê\ế\ế\ë\ė\е]',
            'o': '[\ō\ô\ô\ó\ò\ø\ọ\ŏ\õ\ő\о]',
            'a': '[\á\ā\ā\ă\ã\å\â\à\ą\а]',
            'c': '[\č\ć\ç\с]',
            'l': '[\ł]',
            'u': '[\ú\ū\ứ\ů]',
            'und': '[\&]',
            'r': '[\ř]',
            'y': '[\ý]',
            's': '[\ś\š\ș\ş]',
            'i': '[\ī\ǐ\í\ï\î\ï]',
            'z': '[\ź\ž\ź\ż]',
            'n': '[\ñ\ń\ņ]',
            'g': '[\ğ]',
            'ss': '[\ß]',
            't': '[\ț\ť]',
            'd': '[\ď\đ]',
            "'": '[\ʿ\་\’\`\´\ʻ\`\‘]',
            'p': '\р'
        }

        mappings = {
            'marcel/wav2vec2-large-xlsr-53-german': substitutions_marcel,
            "marcel/wav2vec2-large-xlsr-german-demo": substitutions_marcel,
        }

        if self.based_on:
            return mappings.get(self.based_on, _else)
        return mappings.get(self.model_name, _else)

    @staticmethod
    def __model_list():
        de = ["facebook/wav2vec2-large-xlsr-53-german", \
              "maxidl/wav2vec2-large-xlsr-german", \
              "marcel/wav2vec2-large-xlsr-53-german", \
              "flozi00/wav2vec-xlsr-german", \
              "marcel/wav2vec2-large-xlsr-german-demo", \
              "MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German"]
        return {
            'de': de
        }


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/wav2vec2-large-xlsr-53-german"
    wav2vec = wav2vec2(model_name, device)
    #print(wav2vec)

