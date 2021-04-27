import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import os


class IO:
    @staticmethod
    def load(file: str, sample_rate: int = None, mono_channel: bool = False) -> np.ndarray:
        """
Load and Resample Audiofile
        Args:
            file: string, int, pathlib.Path or file-like object
                Path to (input) WAV-File
            sample_rate: int
                The sample rate of the audio data.
                Default: None -> native SR
            mono_channel: bool
                Force Convert Audio from n to 1 Channel.
                Default: False
        Returns:
            (signal, signal_sample_rate): (np.ndarry, int)
        """
        return librosa.load(file, sr=sample_rate, mono=mono_channel)  # signal, sample_rate

    @staticmethod
    def save_wav(signal: np.ndarray, destination: str, sample_rate: int = 48_000) -> None:
        """
Write Wav-Signal to File.
        Args:
            signal: np.ndarray
                WAV Audio as Signal-Array
            destination: str or int or file-like object
                The file to write to.  See :class:`SoundFile` for details.
            sample_rate: int
                The sample rate of the audio data.
                Default 48_000 [Hz]
        """
        sf.write(destination, data=signal, samplerate=sample_rate, subtype='PCM_24')

    @staticmethod
    def convert_mp3_to_wav(src: str, dst: str, sample_rate: int, mono_channel: bool, overwrite: bool = False) -> str:
        """
Convert MP3 File to WAV File
        Args:
            src: str
                Path to Source Audiofile
            dst: str
                Path to Destination Audiofile
            sample_rate: int
                The sample rate of the audio data.
            mono_channel: bool
                Force Convert Audio from n to 1 Channel.
            overwrite: bool
                Overwrite File or skip if Destination-File already exists

        Returns:
            dest-path: str
        """

        if not overwrite and os.path.exists(dst):
            return dst

        if not os.path.exists(src):
            raise Exception(f"Convert Audio - File not Found\n{src}")

        tmp_dst = dst + "_tmp.wav"

        audio = AudioSegment.from_mp3(src)
        audio.export(tmp_dst, format="wav")  # mp3->wav

        y, sr = IO.load(tmp_dst, sample_rate, mono_channel)  # wav-> wav(sample_rate, channel, clean?, ...)
        IO.save_wav(y, dst, sample_rate)

        os.remove(tmp_dst)

        return dst

    @staticmethod
    def load_sample_rate(path):
        return librosa.get_samplerate(path)

    @staticmethod
    def load_duration(path, sample_rate=None):
        y, sr = IO.load(path, sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration

    @staticmethod
    def load_duration_and_sr(path):
        y, sr = IO.load(path, None) # sr -> None = native
        duration = librosa.get_duration(y=y, sr=sr)
        return duration, sr
