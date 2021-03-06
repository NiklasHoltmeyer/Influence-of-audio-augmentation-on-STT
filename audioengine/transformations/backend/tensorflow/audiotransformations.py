import tensorflow as tf
import tensorflow_io as tfio
import logging


class AudioTransformations:
    @staticmethod
    def load_audio(**kwargs):
        """
        Args:
            audio_format : String = "wav" [default], "mp3"
            **kwargs: desired_samples, desired_channels Ref: https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav

        Returns:  f : (file_path: f  -> tensor)

        """

        audio_decoder = AudioTransformations.decode_audio(**kwargs)

        def __call__(audio_path, y):
            audio_binary = tf.io.read_file(audio_path)
            return audio_decoder(audio_binary), y

        return __call__

    @staticmethod
    def audio_to_spectrogram(**kwargs):
        frame_length = kwargs.get("frame_length", 200)
        frame_step = kwargs.get("frame_step", 80)
        fft_length = kwargs.get("fft_length", 256)

        def __call__(audio, y):
            stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
            return tf.math.pow(tf.abs(stfts), 0.5), y

        return __call__

    @staticmethod
    def normalize(x, y):
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        return ((x - means) / stddevs), y

    @staticmethod
    def pad(**kwargs):
        pad_len = kwargs.get("pad_len", 50)

        logging.getLogger("audioengine-corpus").debug(f"pad_len {pad_len}")

        def __call__(x, y):
            paddings = tf.constant([[0, pad_len], [0, 0]])
            return (tf.pad(x, paddings, "CONSTANT")[:pad_len, :]), y

        return __call__

    @staticmethod
    def decode_audio(**kwargs):
        """
        WAV: Decode 16bit PCM WAV to float Tensor
        MP3: ...

        Args:
            **kwargs: desired_samples, desired_channels Ref: https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav

        Returns:  binary: fn  -> tensor

        """
        audio_format = kwargs.get("audio_format", "wav")

        _formats = {
            "wav": AudioTransformations._decode_wav(**kwargs),
            "mp3": AudioTransformations._decode_mp3(**kwargs)
        }

        audio_decoder = _formats.get(audio_format, None)

        if audio_decoder is None:
            raise Exception("Invalid Audio-Format. Supported-Formats: " + (", ".join(_formats.keys())))

        def __call__(audio_binary):
            audio = audio_decoder(audio_binary)
            return tf.squeeze(audio, axis=-1)

        return __call__

    @staticmethod
    def _decode_wav(**kwargs):
        desired_samples = kwargs.get("desired_samples", -1)
        desired_channels = kwargs.get("desired_channels", -1)

        return lambda audio_binary: tf.audio.decode_wav(audio_binary, desired_channels=desired_channels,
                                                        desired_samples=desired_samples)[
            0]  # 0 -> audio, 1 -> sample_rate

    @staticmethod
    def _decode_mp3(**kwargs):
        shape = kwargs.get("shape", None)
        return lambda audio_binary: tfio.audio.decode_mp3(audio_binary, shape=shape)
