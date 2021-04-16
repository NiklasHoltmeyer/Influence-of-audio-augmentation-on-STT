from audioengine.corpus.backend.tensorflow.dataframedataset import DataframeDataset
import tensorflow as tf


class TensorflowDataset:
    def from_dataframe(self, data_frame, input_key, target_key, transform=None, features=None, **kwargs):
        """
        Load Dataframe into Dataset
        Args:
            features: -
            data_frame: Data
            input_key: Dataframe Column Name for Input
            target_key: Dataframe Column Name for Targets
            transform: (Composed) Transform(s) (over all Data)
            **kwargs: batch_size, shuffle, AUTOTUNE
                audio_format: mp3, wav, ...
                audio_transformation: (Composed) Transform(s) (over Audio-Data)
                transcription_transformation: (Composed) Transform(s) (over Transscription-Data)

        Returns:

        """
        audio_format = kwargs.get("audio_format", None)

        if audio_format is None:
            raise Exception("please Pass audio_format (e.q 'wav' or 'mp3')")

        df = DataframeDataset.from_dataframe(data_frame, input_key, target_key, transform,
                                             **kwargs)  # audio_format -> passed per kwargs
        return df

    @staticmethod
    def compose_transformations(transformations, **kwargs):
        AUTOTUNE = kwargs.get("AUTOTUNE", tf.data.AUTOTUNE)

        def __call__(dataset):
            for transformation in transformations:
                dataset = dataset.map(transformation, num_parallel_calls=AUTOTUNE)
            return dataset

        return __call__
