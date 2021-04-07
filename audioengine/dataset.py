import tensorflow as tf
from audioengine.transformations import *

class Dataset:
    @staticmethod
    def from_file_names(file_names, **kwargs):
        audio_format = Dataset._get_audio_format(file_names[0])

        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        transformations = kwargs.get("transformations", [Transformations.audio_to_spectrogram(**kwargs),
                                                         Transformations.normalize(),
                                                         Transformations.pad(**kwargs)])

        audio_ds = tf.data.Dataset.from_tensor_slices(file_names) \
            .map(Transformations.load_audio(audio_format=audio_format, **kwargs), num_parallel_calls=AUTOTUNE)

        for transformation in transformations:
            audio_ds = audio_ds.map(transformation, num_parallel_calls=AUTOTUNE)

        return audio_ds

    @staticmethod
    def _get_audio_format(file_path):
        return file_path[-3:]

if __name__ == "__main__":
    import os
    path = f"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de\clips"
    file_names = os.listdir(path)[:3] + os.listdir(path)[:-3]

    dataset = Dataset.from_file_names(file_names)

    for item in dataset.batch(1):
        print(item)
