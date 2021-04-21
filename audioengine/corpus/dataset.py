from audioengine.corpus.commonvoice import CommonVoice
from audioengine.corpus.voxforge import VoxForge
from audioengine.transformations.backend.tensorflow.audiotransformations import AudioTransformations


class Dataset:
    def __init__(self, backend):
        """

        Args:
            backend: str
                tensorflow or Pytorch
        """
        self.backend = self.__load_backend(backend)

    def CommonVoice(self, base_path, **kwargs):
        return self._from_AudioDataset(CommonVoice(base_path, **kwargs))

    def VoxForge(self, base_path, **kwargs):
        return self._from_AudioDataset(VoxForge(base_path, **kwargs))

    def _from_AudioDataset(self, audio_ds, input_key, target_key, **kwargs):
        audio_format = audio_ds.audio_format
        dataframe = audio_ds.load_dataframe(**kwargs)
        features = [input_key, target_key, "speech"]
        return self.backend.from_dataframe(dataframe, input_key, target_key, audio_format=audio_format,
                                           features=features, **kwargs)

    def __load_backend(self, backend):
        if "torch" in backend:
            from audioengine.corpus.backend.pytorch.torchdataset import TorchDataset
            return TorchDataset()
        if "tensorflow" in backend or "tf" in backend:
            from audioengine.corpus.backend.tensorflow.tensorflowdataset import TensorflowDataset
            return TensorflowDataset()
        raise Exception(f"Unknown Backend {backend}. \n Supported Backends: pytorch, tensorflow")


if __name__ == "__main__":
    from torchvision import transforms
    from audioengine.transformations.backend.pytorch.audiotransformations import *
    from audioengine.transformations.backend.pytorch.texttransformations import *

    linux_path = "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
    windows_path = r"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de"
    path = windows_path

    audio_transformations = [
        AudioTransformations.load_audio(audio_format="mp3"),
        AudioTransformations.audio_to_spectrogram(),
        AudioTransformations.normalize(),
        AudioTransformations.pad(pad_len=2754)
    ]


    def map_audio(x, y):
        for trans in audio_transformations:
            x = trans(x)
        return x, y


    transform = AudioTransformations.load_audio(audio_format="mp3")

    ds = Dataset("tf").CommonVoice(windows_path, batch_size=1)
    ds = ds.map(map_audio)
    for x, y in ds.take(1):
        print(x)
        break
