import io
from typing import Tuple
import numpy as np

from pydub import AudioSegment

def convert_to_mono(input_file: str, mono_audio_path: str, sample_rate: int = 16000):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(sample_rate)
    audio.export(mono_audio_path, format="wav")

def convert_to_mono_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

def from_file_to_np_array(input_file: str, dtype: str = "float32") -> Tuple[np.ndarray, int]:
    import soundfile as sf

    data, samplerate = sf.read(input_file, dtype=dtype)
    return data, samplerate


def from_bytes_to_np_ndarray(bytes_data: bytes, dtype: str = "float32") -> Tuple[np.ndarray, int]:
    """
    Convert byte data to np.ndarray format.
    Args:
        bytes_data: Audio bytes of data.
        dtype: Default is float32.

    Returns: Returns the converted np.ndarray format data, sample rate.

    """
    import soundfile as sf

    wave_bytes_buf = io.BytesIO(bytes_data)
    data, samplerate = sf.read(wave_bytes_buf, dtype=dtype)
    return data, samplerate