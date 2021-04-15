import tensorflow as tf


class DataframeDataset:
    def __init__(self):
        pass

    @staticmethod
    def from_dataframe(dataframe, input_key, target_key, transform, **kwargs):
        return DataframeDataset.from_slices(dataframe[input_key], dataframe[target_key], transform, **kwargs)

    @staticmethod
    def from_slices(audio_paths, audio_transcriptions, transform, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        shuffle = kwargs.get("shuffle", False)
        BATCH_SIZE = kwargs.get("batch_size", 32)
        audio_transform = kwargs.get("audio_transform", None)
        trans_transform = kwargs.get("trans_transform", None)

        audio_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = audio_ds.map(audio_transform, AUTOTUNE) if audio_transform else audio_ds
        trans_ds = tf.data.Dataset.from_tensor_slices(audio_transcriptions)
        trans_ds = trans_ds.map(trans_transform, AUTOTUNE) if trans_transform else trans_ds

        ds = tf.data.Dataset.zip((audio_ds, trans_ds))
        ds = ds.map(transform) if transform else ds
        ds = ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(AUTOTUNE)

        return ds

    @staticmethod
    def from_file_names(file_names, transcriptions, **kwargs):
        return DataframeDataset.from_slices(file_names, transcriptions, **kwargs)

    @staticmethod
    def _get_audio_format(file_path):
        return file_path[-3:]
