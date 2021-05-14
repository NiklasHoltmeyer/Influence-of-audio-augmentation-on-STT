from sklearn.model_selection import train_test_split

from audioengine.corpus.backend.pytorch.huggingfacedataset import HuggingfaceDataset
from audioengine.corpus.commonvoice import CommonVoice
from audioengine.corpus.tts_synthesized import TTSSynthesized
from audioengine.corpus.voxforge import VoxForge
from audioengine.logging.logging import defaultLogger
from audioengine.transformations.backend.tensorflow.audiotransformations import AudioTransformations
import pandas as pd

logger = defaultLogger()


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

    def from_settings(self, settings):
        val_settings = settings["val_settings"]
        train_settings = settings["train_settings"]
        transform = settings.get("transform", None)

        val_ds, val_df_info = self._build_ds_from_settings(val_settings, transform)
        train_ds, train_df_info = self._build_ds_from_settings(train_settings, transform)

        return (train_ds, train_df_info), (val_ds, val_df_info)

    def _build_ds_from_settings(self, settings, transform):
        if not settings:
            return None, "No Settings Provided"
        dfs_with_info = [self._load_from_name(**settings) for settings in settings]
        df_info = "+".join([info for _, info in dfs_with_info])
        dfs = [df for df, _ in dfs_with_info]
        df = pd.concat(dfs).sample(frac=1)

        input_key, target_key = "path", "sentence"
        ds = self._from_Dataframe(audio_format="mix", dataframe=df,
                                  input_key=input_key, target_key=target_key, transform=transform)
        # fixed_length=None, validation_split=None
        return ds, df_info

    def _load_from_name(self, base_path, **kwargs):
        desc = kwargs.get("desc", None)
        desc = f"-{desc}" if desc else "-"

        if self.__is_cv(base_path):
            df = CommonVoice(base_path, **kwargs).load_dataframe(**kwargs)
            return df, f"commonvoice{desc}-{kwargs.get('type')}({len(df)})"
        if self.__is_vf(base_path):
            df = VoxForge(base_path, **kwargs).load_dataframe(**kwargs)
            _type = kwargs.get('type')
            if _type:
                return df, f"voxforge{desc}-{_type}({len(df)})"
            return df, f"voxforge{desc}-({len(df)})"
        if self.__is_tts(base_path):
            df = TTSSynthesized(output_dir=base_path, tts_engine=None, text_files=None).load_dataframe(**kwargs)
            _type = kwargs.get("type")
            if _type:
                return df, f"tts{desc}-{_type}({len(df)})"
            return df, f"tts{desc}-({len(df)})"

        raise Exception(f"Unknown DS {base_path}")

    def __is_cv(self, name):
        return "common" in name or "cv" in name.lower()

    def __is_vf(self, name):
        return "voxforge" in name.lower() or "vf" in name.lower()

    def __is_tts(self, name):
        return "texttospeech" in name.lower() or "tts" in name.lower()

    def _from_AudioDataset(self, audio_ds, input_key="path", target_key="sentence", **kwargs):
        audio_format = audio_ds.audio_format
        dataframe = audio_ds.load_dataframe(**kwargs)
        return self._from_Dataframe(audio_format, dataframe, input_key, target_key, **kwargs)

    def _from_Dataframe(self, audio_format, dataframe, input_key, target_key, **kwargs):
        validation_split = kwargs.get("validation_split", None)
        features = [input_key, target_key, "speech"]
        fixed_length = kwargs.get("fixed_length", None)
        # if fixed_length:
        #    if not kwargs.get("shuffle"):
        #        logger.warning("Shuffle is disabled, while fixed_length is enabled.")
        #    _items = min(fixed_length, len(dataframe))
        #    dataframe = dataframe[: _items]
        #    return self.backend.from_dataframe(dataframe, input_key, target_key, audio_format=audio_format,
        #                                       features=features, **kwargs)

        if fixed_length or not validation_split:
            return self.backend.from_dataframe(dataframe, input_key, target_key, audio_format=audio_format,
                                               features=features, **kwargs)
        train_df, val_df = train_test_split(dataframe, test_size=validation_split)
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
    path = linux_path

    #"/share/datasets/8mil_tts/"
    Dataset("/share/datasets/8mil_tts/")

    # train_ds, val_ds = Dataset("torch").CommonVoice(path, validation_split=0.2, batch_size=1)

# load_max_input_length
# max_input_length
#
