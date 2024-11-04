"""
More details about the model:
    https://huggingface.co/augmxnt/shisa-7b-v1
"""

import copy
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.shisa.config import ShisaModelConfig
from zerolan_live_robot_data.data.llm import LLMQuery, LLMPrediction, Conversation


class Shisa7B_V1(AbstractModel):

    def __init__(self, config: ShisaModelConfig):
        super().__init__()
        self._s_streamer = None
        self.model_id = "augmxnt/shisa-7b-v1"
        self._model_path = config.model_path
        self._device = config.device

        self._tokenizer: any = None
        self._model: any = None
        self._streamer: TextStreamer | None = None
        self._max_new_tokens = 500
        self._temperature = 0.5
        self._repetition_penalty = 1.15
        self._top_p = 0.95

    @log_model_loading("augmxnt/shisa-7b-v1")
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ).to(self._device)
        self._streamer = TextStreamer(self._tokenizer, skip_prompt=True)
        self._s_streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True)
        assert self._tokenizer and self._model and self._streamer

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        """
        Predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan_live_robot_data.data.llm.LLMQuery

        Returns: See zerolan_live_robot_data.data.llm.LLMPrediction

        """
        history = self._to_shisa_format(llm_query)

        inputs = self._tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
        first_param_device = next(self._model.parameters()).device
        inputs = inputs.to(first_param_device)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs,
                pad_token_id=self._tokenizer.eos_token_id,
                do_sample=True,
                streamer=self._streamer,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                repetition_penalty=self._repetition_penalty,
                top_p=self._top_p
            )

        new_tokens = outputs[0, inputs.size(1):]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._to_pipeline_format(response, llm_query.history)

    def stream_predict(self, llm_query: LLMQuery) -> Any:
        """
        Stream predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan_live_robot_data.data.llm.LLMQuery

        Returns: See zerolan_live_robot_data.data.llm.LLMPrediction

        """
        raise NotImplementedError('Stream prediction has not implemented!')

    @staticmethod
    def _to_shisa_format(llm_query: LLMQuery):
        assert len(llm_query.history) > 0 and llm_query.history[0].role == "system", f'The query must have system prompt.'

        llm_query_history = copy.deepcopy(llm_query.history)
        llm_query_history.append(Conversation(role="user", content=llm_query.text))
        history = [{"role": chat.role, "content": chat.content} for chat in llm_query_history]
        return history

    @staticmethod
    def _to_pipeline_format(response, history):
        history.append(Conversation(role="assistant", content=response))
        return LLMPrediction(response=response, history=history)
