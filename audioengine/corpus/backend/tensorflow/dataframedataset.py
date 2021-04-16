import tensorflow as tf


class DataframeDataset:
    @staticmethod
    def from_dataframe(dataframe, input_key, target_key, transform, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        shuffle = kwargs.get("shuffle", False)
        BATCH_SIZE = kwargs.get("batch_size", 32)

        x, y = dataframe.pop(input_key).values, dataframe.pop(target_key).values
        dataset = tf.data.Dataset.from_tensor_slices((x, y))\
            .map(transform, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

        return dataset.shuffle(AUTOTUNE) if shuffle else dataset

    @staticmethod
    def from_file_names(file_names, transcriptions, transform, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)
        shuffle = kwargs.get("shuffle", False)
        BATCH_SIZE = kwargs.get("batch_size", 32)
        dataset = DataframeDataset.from_slices(file_names, transcriptions, **kwargs)\
            .map(transform).batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
        return dataset.shuffle(AUTOTUNE) if shuffle else dataset

#    @staticmethod
#    def _get_audio_format(file_path):
#        return file_path[-3:]
