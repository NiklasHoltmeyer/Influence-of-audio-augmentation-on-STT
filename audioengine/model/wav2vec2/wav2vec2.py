import torch

from audioengine.transformations.backend.PyTorch.audiotransformations import LoadAudio
from audioengine.transformations.backend.PyTorch.texttransformations import RegExp
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class wav2vec2:
    def __init__(self, model_name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.model_name = model_name
        self.device = device

        self.model, self.processor = self._load_pretrained()

    def predict(self, batch, sampling_rate=16_000, padding=True):
        inputs = self.processor(batch["speech"], sampling_rate=sampling_rate, return_tensors="pt", padding=padding)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["transcription"] = self.processor.batch_decode(pred_ids)
        return batch

    def transformations(self):
        def to_lower(data):
            data["sentence"] = data["sentence"].lower().replace("’", "'")
            return data

        transformations = [to_lower]

        chars_to_ignore_regex = self._chars_to_remove()
        regexp_subs = [(chars_to_ignore_regex, '')] if chars_to_ignore_regex else []

        replacements = self._chars_to_replace()
        if replacements:
            for key, value in replacements.items():
                # replace (value, key)
                regexp_subs.append((value, key))

        if chars_to_ignore_regex:
            regexp_layer = RegExp(regexp_subs)
            transformations.append(regexp_layer)

        transformations.append(LoadAudio(48_000, 16_000))
        return transformations

    def _load_pretrained(self):
        processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        model.to("cuda")
        return model, processor

    def __str__(self):
        infos = {
            "model_name": self.model_name,
            "backend": "PyTorch"
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

