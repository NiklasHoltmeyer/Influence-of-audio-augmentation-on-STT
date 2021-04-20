from audioengine.model.pretrained.silero import Silero

from audioengine.model.pretrained.wav2vec2 import wav2vec2

from audioengine.service.singleton import Singleton
import torch


@Singleton
class STTService:

    def __init__(self, **kwargs):
        """ Use STTService.instance(**kwargs) instead of STTService(**kwargs)"""
        self.model, self.model_name, self.language = self.load_model(**kwargs)

    def load_model(self, **kwargs):
        model_type = kwargs.get("model_type", "wav2vec2")
        model_name = kwargs.get("model_name", "maxidl/wav2vec2-large-xlsr-german")
        language = kwargs.get("language", "de")
        device = kwargs.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        cls_mapping = {"wav2vec2": lambda: wav2vec2(model_name, device), #lambda -> only init if selected
                       "silero": lambda: Silero(language, device)}

        if model_type not in cls_mapping.keys():
            raise Exception("Unknown Model-Type")

        model = cls_mapping[model_type] #()

        return model, model_name, language

    def predict(self, speeches):
        return self.model.predict(speeches)


#if __name__ == "__main__":
#    print("test")
#    stts_1 = STTService.instance()
#    stts_2 = STTService.instance()
#    assert stts_1 is stts_2
#    assert id(stts_1) == id(stts_2)
#    print("Singleton [x]")

#