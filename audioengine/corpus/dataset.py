from audioengine.corpus.commonvoice import CommonVoice

class Dataset:
    def __init__(self, backend):
        """

        Args:
            backend: str
                tensorflow or Pytorch
        """
        self.backend = self.__load_backend(backend)

    def CommonVoice(self, base_path, **kwargs):
        cv = CommonVoice(base_path)
        audio_format = cv.audio_format
        dataframe = cv.load_dataframe(**kwargs)
        input_key = "path"
        target_key = "sentence"
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

    ds = Dataset("tf").CommonVoice(windows_path)
    for x in ds:
        print(x)
        break
    #print(ds)
