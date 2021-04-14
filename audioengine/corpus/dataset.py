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
            from audioengine.corpus.backend.PyTorch.torchdataset import TorchDataset
            return TorchDataset()
        if "tensorflow" in backend or "tf" in backend:
            from audioengine.corpus.backend.Tensorflow.tensorflowdataset import TensorflowDataset
            return TensorflowDataset()
        raise Exception(f"Unknown Backend {backend}. \n Supported Backends: pytorch, tensorflow")


if __name__ == "__main__":
    from torchvision import transforms
    from audioengine.transformations.backend.pytorch.audiotransformations import *
    from audioengine.transformations.backend.pytorch.texttransformations import *
    linux_path = "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
    windows_path = r"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de"
    path = windows_path

    def simple_text_transform(data):
        data["sentence"] = data["sentence"].lower().replace("’", "'")
        return data


    chars_to_ignore_regex = ('[\,\?\.\!\-\;\:\"]', '')
    regexp_layer = Regexp([chars_to_ignore_regex])
    transformations = [simple_text_transform, regexp_layer] #LoadAudio(48_000, 16_000)
    transform = transforms.Compose(transformations)

    ds_helper = Dataset("torch")
    cv_ds = ds_helper.CommonVoice(path, batch_size=1,shuffle=False, transform=transform)

    print(cv_ds)
    print("*"*32)
    for x in cv_ds:
        pass
    # audioengine
