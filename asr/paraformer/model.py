"""
More details about the model:
    https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1
"""

import numpy as np
from funasr import AutoModel
from loguru import logger


from asr.paraformer.config import SpeechParaformerModelConfig
from utils import audio_util
from common.decorator import log_model_loading
from zerolan.data.data.asr import ASRModelPrediction, ASRModelQuery, ASRModelStreamQuery


class SpeechParaformerModel:

    def __init__(self, config: SpeechParaformerModelConfig):
        self.model_id = "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1"
        self._model: any = None
        # Streaming Inference Configuration
        # Note: chunk_size is configured for streaming delay.
        # [0,10,5] indicates that the granularity of real-time token output is 10 * 60 = 600 ms, and the future information is 5 * 60 = 300 ms.
        # The input of each inference is 600ms (the number of sampling points is 16000*0.6=960), and the output is the corresponding text.
        # The last voice clip input needs to be set to is_final=True to force the last word to be output.
        self._sample_rate = 16000  # Using a different sample rate will result in an error
        self.dtype = np.float32

        self._chunk_size = config.chunk_size
        self._encoder_chunk_look_back = config.encoder_chunk_look_back
        self._decoder_chunk_look_back = config.decoder_chunk_look_back
        self._model_path = config.model_path
        self._version = config.version
        self._chunk_stride = config.chunk_stride
        self._cache = {}
        self._is_final = False

    @log_model_loading("iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1")
    def load_model(self):
        """
        Load the model.
        """
        self._model = AutoModel(model=self._model_path, model_revision=self._version)
        assert self._model

    def predict(self, query: ASRModelQuery) -> ASRModelPrediction | None:

        """
        Predict words from the audio wave.
        Args:
            query: Contains audio data with a sample rate of 16000 and is mono

        Returns:

        """
        wave_nparray, sample_rate = audio_util.from_file_to_np_array(query.audio_path, self.dtype.__name__)
        assert sample_rate and sample_rate == self._sample_rate, "The sampling rate must be 16000, otherwise the recognition results will be severely skewed."
        is_final = query.audio_path is None

        return self._wrapper(wave_nparray, is_final)

    def stream_predict(self, query: ASRModelStreamQuery) -> ASRModelPrediction | None:
        """
        Stream predict words from the audio wave.
        Args:
            query: Contains audio data with a sample rate of 16000 and is mono

        Returns:

        """
        wave_nparray, sample_rate = audio_util.from_bytes_to_np_ndarray(query.audio_data, self.dtype.__name__)
        assert sample_rate and sample_rate == self._sample_rate, "The sampling rate must be 16000, otherwise the recognition results will be severely skewed."
        return self._wrapper(wave_nparray, query.is_final)

    def _wrapper(self, wave_nparray: np.ndarray, is_final: bool) -> ASRModelPrediction | None:
        assert wave_nparray is not None and isinstance(wave_nparray, np.ndarray), "Wrong format."
        assert len(wave_nparray) > 0, "The audio tensor size must be greater than 0."
        assert len(wave_nparray.shape) == 1, "The audio must be mono."
        assert wave_nparray.dtype == np.float32, "The dtype type that is not supported, must be numpy.float32."
        try:
            res = self._model.generate(input=wave_nparray, cache=self._cache, is_final=is_final,
                                       chunk_size=self._chunk_size,
                                       encoder_chunk_look_back=self._encoder_chunk_look_back,
                                       decoder_chunk_look_back=self._decoder_chunk_look_back)
            transcript = res[0]["text"]

            return ASRModelPrediction(transcript)
        except IndexError as e:
            if "IndexError: list index out of range" in str(e):
                logger.warning("Prediction error unexcpectedly.")
            return None
        except AssertionError as e:
            if "AssertionError: choose a window size" in str(e):
                logger.warning("Audio tensor size error.")
            logger.exception(e)
            return None
        except Exception as e:
            logger.exception(e)
            return None
