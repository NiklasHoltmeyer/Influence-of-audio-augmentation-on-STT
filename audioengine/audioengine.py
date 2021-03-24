import os
import librosa
import soundfile as sf
from pydub import AudioSegment


class Audioengine:
    """
        **IO Audio**

        **Todo**:
            - Adding Noise (for Training)
            - Reducing non Vocal Noise
    """

    @staticmethod
    def load_wav(file: str, sample_rate: int = 48_000, mono_channel: bool = False):
        """
        Load and Resample (WAV) Audiofile

        :param file: string, int, pathlib.Path or file-like object
            Path to (input) WAV-File
        :param sample_rate: int
            The sample rate of the audio data.
            Default: 48_000 [Hz]
        :param mono_channel: bool
            Force Convert Audio from n to 1 Channel.
            Default: False
        :return: (signal, signal_sample_rate): (np.ndarry, int)
        """
        return librosa.load(file, sr=sample_rate, mono=mono_channel)  # signal, sample_rate

    @staticmethod
    def save_wav(signal: any, destination: str, sample_rate: int = 48_000) -> None:
        """
        Write Wav-Signal to File.

        Parameters
        ----------

        :param signal: array_like
            WAV Audio as Signal-Array (see :func:`soundfile`)
        :param destination: str or int or file-like object
            The file to write to.  See :class:`SoundFile` for details.
        :param sample_rate: int
            The sample rate of the audio data.
            Default 48_000 [Hz]
        :return: None
        """
        sf.write(destination, data=signal, samplerate=sample_rate, subtype='PCM_24')

    @staticmethod
    def convert_mp3_to_wav(src, dst, sample_rate, mono_channel, overwrite=False):
        """
            Convert MP3 File to WAV File

        :param src:
            Path to Source Audiofile
        :param dst:
            Path to Destination Audiofile
        :param sample_rate:
            The sample rate of the audio data.
        :param mono_channel:
            Force Convert Audio from n to 1 Channel.
        :param overwrite:
            Overwrite File or skip if Destination-File already exists
        :return: Path to Destination Audiofile: str
        """
        if not overwrite and os.path.exists(dst):
            return dst

        if not os.path.exists(src):
            raise Exception(f"Convert Audio - File not Found\n{src}")

        tmp_dst = dst + "_tmp.wav"

        audio = AudioSegment.from_mp3(src)
        audio.export(tmp_dst, format="wav")  # mp3->wav

        y, sr = Audioengine.load_wav(tmp_dst, sample_rate, mono_channel)  # wav-> wav(sample_rate, channel, clean?, ...)
        Audioengine.save_wav(y, dst, sample_rate)

        os.remove(tmp_dst)

        return dst