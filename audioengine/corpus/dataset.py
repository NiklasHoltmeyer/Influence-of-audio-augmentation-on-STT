from sklearn.model_selection import train_test_split

from audioengine.corpus.backend.pytorch.huggingfacedataset import HuggingfaceDataset
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
        return self._from_AudioDataset(CommonVoice(base_path, **kwargs), **kwargs)

    def VoxForge(self, base_path, **kwargs):
        return self._from_AudioDataset(VoxForge(base_path, **kwargs), **kwargs)

    def _from_AudioDataset(self, audio_ds, validation_split, input_key="path", target_key="sentence", **kwargs):
        audio_format = audio_ds.audio_format
        dataframe = audio_ds.load_dataframe(**kwargs)

        train_df, val_df = train_test_split(dataframe, test_size=validation_split)

        features = [input_key, target_key, "speech"]
        train_ds = self.backend.from_dataframe(train_df, input_key, target_key, audio_format=audio_format,
                                               features=features, **kwargs)
        val_ds = self.backend.from_dataframe(val_df, input_key, target_key, audio_format=audio_format,
                                             features=features, **kwargs)
        return train_ds, val_ds

    def __load_backend(self, backend):
        if "torch" in backend:
            from audioengine.corpus.backend.pytorch.torchdataset import TorchDataset
            return TorchDataset()
        if "tensorflow" in backend or "tf" in backend:
            from audioengine.corpus.backend.tensorflow.tensorflowdataset import TensorflowDataset
            return TensorflowDataset()
        if "huggingface" in backend:
            return HuggingfaceDataset()
        raise Exception(f"Unknown Backend {backend}. \n Supported Backends: pytorch, tensorflow")


if __name__ == "__main__":
    from torchvision import transforms
    from audioengine.transformations.backend.pytorch.audiotransformations import *
    from audioengine.transformations.backend.pytorch.texttransformations import *

    linux_path = "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
    windows_path = r"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de"
    path = "/share/datasets/voxforge_todo"

    train_ds, val_ds = Dataset("torch").VoxForge(path, validation_split=0.2, batch_size=1)
    print(len(train_ds))
    print(len(val_ds))

# load_max_input_length
# max_input_length
#
