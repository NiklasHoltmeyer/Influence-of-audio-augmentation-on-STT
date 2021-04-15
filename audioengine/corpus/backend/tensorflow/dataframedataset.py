import tensorflow as tf


class DataframeDataset:
    def __init__(self):
        pass

    @staticmethod
    def from_dataframe(dataframe, input_key, target_key, **kwargs):
        return DataframeDataset.from_slices(dataframe[input_key], dataframe[target_key], **kwargs)

    @staticmethod
    def from_slices(audio_paths, audio_transcriptions, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        shuffle = kwargs.get("shuffle", False)
        BATCH_SIZE = kwargs.get("batch_size", 32)

        audio_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = DataframeDataset._transform_audio(audio_ds, **kwargs)

        trans_ds = tf.data.Dataset.from_tensor_slices(audio_transcriptions)
        trans_ds = DataframeDataset._transform_transcriptions(trans_ds, **kwargs)

        ds = tf.data.Dataset.zip((audio_ds, trans_ds)).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(AUTOTUNE)

        return ds

    @staticmethod
    def from_file_names(file_names, transcriptions, **kwargs):
        #audio_format = DataframeDataset._get_audio_format(file_names[0])
        return DataframeDataset.from_slices(file_names, transcriptions, **kwargs)

    @staticmethod
    def _transform_audio(audio_ds, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        #audio_format = kwargs.get("audio_format", None)
        #                                                      [AudioTransformations.audio_to_spectrogram(**kwargs),
        # AudioTransformations.normalize(),
        # AudioTransformations.pad(**kwargs)])
        transformations = kwargs.get("audio_transformation", None)

        # audio_ds = audio_ds \
        # .map(AudioTransformations.load_audio(audio_format=audio_format, **kwargs), num_parallel_calls=AUTOTUNE)

        if transformations:
            audio_ds = audio_ds.map(transformations, num_parallel_calls=AUTOTUNE)

        return audio_ds

    @staticmethod
    def _transform_transcriptions(trans_ds, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        transformation = kwargs.get("transcription_transformation", None)

        if transformation:
            trans_ds = trans_ds.map(transformation, num_parallel_calls=AUTOTUNE)

        return trans_ds

    @staticmethod
    def _get_audio_format(file_path):
        return file_path[-3:]
