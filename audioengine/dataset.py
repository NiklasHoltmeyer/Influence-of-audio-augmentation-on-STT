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

    path_mp3 = f"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de\clips"
    path_wav = f"C:\workspace\datasets\LJSpeech-1.1\wavs"

    file_names_mp3 = os.listdir(path_mp3)[:3] + os.listdir(path_mp3)[:-3]
    file_names_mp3 = [f"{path_mp3}\\{x}" for x in file_names_mp3]

    file_names_wav = os.listdir(path_wav)[:3] + os.listdir(path_wav)[:-3]
    file_names_wav = [f"{path_wav}\\{x}" for x in file_names_wav]

    dataset_mp3 = Dataset.from_file_names(file_names_mp3)
    dataset_wav = Dataset.from_file_names(file_names_wav)

    for idx, item in enumerate(dataset_mp3.batch(1).take(5)):
        print(f"File: {file_names_mp3[idx]}")
        print(item)
        print("*"*32)


    print("-"*32)

    for idx, item in enumerate(dataset_wav.batch(1).take(5)):
        print(f"File: {file_names_wav[idx]}")
        print(item)
        print("*"*32)
