"""
More details about the model:
    https://huggingface.co/Qwen/Qwen-7B-Chat
"""
from typing import Any

from loguru import logger

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.qwen.config import QwenModelConfig
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation


def catch_error(callable):
    try:
        callable()
    except AssertionError as e:
        if 'Only one of "bf16", "fp16", "fp32" can be true' in str(e):
            logger.error(
                "Tips: You may be using a quantized model; if so, set the `precise` parameter to `null`.")
        raise e


class Qwen7BChat(AbstractModel):

    def __init__(self, config: QwenModelConfig):
        super().__init__()

        self.model_id = "Qwen/Qwen-7B-Chat"
        self._model_path = config.model_path
        # Warning:
        #   You may encounter some errors when using multi-GPU.
        self._device = config.device
        self._precise = config.precise

        self._model: any = None
        self._tokenizer: any = None

    @log_model_loading("Qwen/Qwen-7B-Chat")
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True)
        if self._precise == "bf16":
            def load_bf16():
                self._model = AutoModelForCausalLM.from_pretrained(self._model_path,
                                                                   device_map=self._device,
                                                                   trust_remote_code=True,
                                                                   bf16=True)
            catch_error(load_bf16)
        elif self._precise == "fp16":
            def load_fp16():
                self._model = AutoModelForCausalLM.from_pretrained(self._model_path,
                                                                   device_map=self._device,
                                                                   trust_remote_code=True,
                                                                   fp16=True)
            catch_error(load_fp16)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self._model_path,
                                                               device_map=self._device,
                                                               trust_remote_code=True)
        assert self._tokenizer and self._model

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        """
        Predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction

        """
        text, history, sys_prompt = self._to_qwen_format(llm_query)
        response, history = self._model.chat(
            self._tokenizer, llm_query.text, history=history)
        logger.debug(response)
        return self._to_pipeline_format(response, history, sys_prompt)

    def stream_predict(self, llm_query: LLMQuery) -> Any:
        """
        Stream predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction

        """
        text, history, sys_prompt = self._to_qwen_format(llm_query)
        history.append((text, ""))
        # If the error occured
        #   typeError: isin() received an invalid combination of arguments - got (test_elements=int, elements=Tensor,), but expected one of...
        # Please see here:
        #   https://github.com/QwenLM/Qwen-VL/issues/407
        #   `pip install transformers==4.32.0`
        for response in self._model.chat_stream(self._tokenizer, llm_query.text, history=history):
            logger.debug(response)
            history[-1] = (text, response)

    @staticmethod
    def _to_qwen_format(llm_query: LLMQuery) -> (str, list[tuple[str, str]], str):
        history_content_list = [c.content for c in llm_query.history]

        sys_prompt = None
        history = []

        if llm_query.history and llm_query.history[0].role == "system":
            sys_prompt = llm_query.history[0].content
            history_content_list = history_content_list[1:]

            if history_content_list:
                history_content_list[0] = sys_prompt + history_content_list[0]

        if len(history_content_list) % 2 != 0:
            raise ValueError(
                "The number of history messages must be even (user and assistant turns must come in pairs).")

        history = [
            (history_content_list[i], history_content_list[i + 1])
            for i in range(0, len(history_content_list), 2)
        ]

        return llm_query.text, history, sys_prompt

    @staticmethod
    def _to_pipeline_format(response: str, history: list[tuple[str, str]], sys_prompt: str) -> LLMPrediction:
        # Convert the history as format of pipeline.
        ret_history: list[Conversation] = []
        for chat in history:
            q, r = chat[0], chat[1]
            assert isinstance(q, str) and isinstance(r, str)
            ret_history.append(Conversation(role="user", content=q))
            ret_history.append(Conversation(role="assistant", content=r))

        # Get system prompt.
        if sys_prompt:
            ret_history[0].content = ret_history[0].content[len(sys_prompt):]
            ret_history.insert(0, Conversation(
                role="system", content=sys_prompt))

        llm_response = LLMPrediction(response=response, history=ret_history)

        return llm_response
