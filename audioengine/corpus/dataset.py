import logging

from audioengine.corpus.commonvoice import CommonVoice
from audioengine.transformations import *


class Dataset:
    @staticmethod
    def from_slices(audio_paths, audio_transcriptions, audio_format, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        BATCH_SIZE = kwargs.get("batch_size", 32)

        audio_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = Dataset._transform_audio(audio_ds, audio_format)

        trans_ds = tf.data.Dataset.from_tensor_slices(audio_transcriptions)
        trans_ds = Dataset._transform_transcriptions(trans_ds)

        return tf.data.Dataset.zip((audio_ds, trans_ds)).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

    @staticmethod
    def from_file_names(file_names, transcriptions, **kwargs):
        audio_format = Dataset._get_audio_format(file_names[0])
        return Dataset.from_slices(file_names, transcriptions, **kwargs)

    @staticmethod
    def from_dataframe(dataframe, audio_format, **kwargs):
        """ (DF) req-Keys: audio_path, transscript """
        return Dataset.from_slices(dataframe.audio_path, dataframe.transcript, audio_format, **kwargs)

    @staticmethod
    def CommonVoice(base_path, **kwargs):
        cv = CommonVoice(base_path)
        audio_format = cv.audio_format
        dataframe = cv.load_dataframe(**kwargs)

        return Dataset.from_dataframe(dataframe, audio_format, **kwargs)

    @staticmethod
    def _transform_audio(audio_ds, audio_format, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        transformations = kwargs.get("audio_transformations", [Transformations.audio_to_spectrogram(**kwargs),
                                                               Transformations.normalize(),
                                                               Transformations.pad(**kwargs)])
        audio_ds = audio_ds \
            .map(Transformations.load_audio(audio_format=audio_format, **kwargs), num_parallel_calls=AUTOTUNE)

        for transformation in transformations:
            audio_ds = audio_ds.map(transformation, num_parallel_calls=AUTOTUNE)

        return audio_ds

    @staticmethod
    def _transform_transcriptions(trans_ds, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        transformations = kwargs.get("transcription_transformations", [])

        # map -> basic clean_text -> to lower etc
        trans_ds = trans_ds.map(TextTransformations.lower(), num_parallel_calls=AUTOTUNE)

        for transformation in transformations:
            trans_ds = trans_ds.map(transformation, num_parallel_calls=AUTOTUNE)

        return trans_ds

    @staticmethod
    def _get_audio_format(file_path):
        return file_path[-3:]


if __name__ == "__main__":
    #r"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de"
    logging.basicConfig(level=logging.DEBUG)
    cv_path = r"C:\workspace\datasets\cv\de\cv-corpus-6.1-2020-12-11\de"

    Dataset.CommonVoice(cv_path)
