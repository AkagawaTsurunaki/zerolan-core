"""
https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0
"""

import torch
from loguru import logger
from transformers import pipeline, Pipeline
from zerolan.data.pipeline.asr import ASRPrediction, ASRQuery, ASRStreamQuery

from asr.kotoba_whisper_2.config import KotobaWhisper2Config
from common.decorator import log_model_loading


def is_cuda_available(device: str):
    if "cuda" in device:
        return torch.cuda.is_available()
    return False


class KotobaWhisper2:

    def __init__(self, config: KotobaWhisper2Config):
        # config
        self._model_path = config.model_path
        self._device = config.device
        self._torch_dtype = torch.bfloat16 if is_cuda_available(config.device) else torch.float32
        self._model_kwargs = {"attn_implementation": "sdpa"} if is_cuda_available(config.device) else {}
        self._generate_kwargs = {"language": "ja", "task": "transcribe"}
        self._model: Pipeline = None

    @log_model_loading("kotoba-tech/kotoba-whisper-v2.0")
    def load_model(self):
        # load model
        self._model = pipeline(
            "automatic-speech-recognition",
            model=self._model_path,
            torch_dtype=self._torch_dtype,
            device=self._device,
            model_kwargs=self._model_kwargs
        )

    def predict(self, query: ASRQuery) -> ASRPrediction | None:
        # run inference
        audio_path = query.audio_path
        result = self._model(audio_path, generate_kwargs=self._generate_kwargs)
        logger.info("ASR: " + result["text"])
        return ASRPrediction(transcript=result["text"])

    def stream_predict(self, query: ASRStreamQuery) -> ASRPrediction | None:
        audio_data = query.audio_data
        result = self._model(audio_data, generate_kwargs=self._generate_kwargs)
        logger.info("ASR: " + result["text"])
        return ASRPrediction(transcript=result["text"])
