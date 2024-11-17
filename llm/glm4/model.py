from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zerolan.data.data.llm import LLMQuery, LLMPrediction

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.glm4.config import GLM4ModelConfig


class ChatGLM3_6B(AbstractModel):

    def __init__(self, config: GLM4ModelConfig):
        super().__init__()
        self.model_id = "THUDM/glm-4-9b-chat-hf"
        self._model_path = config.model_path
        self._device = config.device

        self._tokenizer: any = None
        self._model: any = None

    @log_model_loading("THUDM/glm-4-9b-chat-hf")
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self._device
        ).eval()

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        pass

    def stream_predict(self, llm_query: LLMQuery):
        raise NotImplementedError()
