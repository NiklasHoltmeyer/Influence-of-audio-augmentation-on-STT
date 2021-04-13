from audioengine.corpus.commonvoice import CommonVoice

class Dataset:
    def __init__(self, backend):
        """

        Args:
            backend: str
                Tensorflow or Pytorch
        """
        self.backend = self.__load_backend(backend)

    def CommonVoice(self, base_path, **kwargs):
        cv = CommonVoice(base_path)
        audio_format = cv.audio_format
        dataframe = cv.load_dataframe(**kwargs)
        input_key = "audio_path"
        target_key = "transcript"

        return self.backend.from_dataframe(dataframe, input_key, target_key, audio_format=audio_format, **kwargs)

    def __load_backend(self, backend):
        if "torch" in backend:
            from audioengine.corpus.backend.PyTorch.torchdataset import TorchDataset
            return TorchDataset()
        if "tensorflow" in backend or "tf" in backend:
            from audioengine.corpus.backend.Tensorflow.tensorflowdataset import TensorflowDataset
            return TensorflowDataset()
        raise Exception(f"Unknown Backend {backend}. \n Supported Backends: pytorch, tensorflow")

