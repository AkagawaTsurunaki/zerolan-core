"""
More details about the model:
    https://github.com/THUDM/ChatGLM3
"""
from loguru import logger
from transformers import AutoTokenizer, AutoModel

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.chatglm3.config import ChatGLM3ModelConfig
from zerolan_live_robot_data.data.llm import LLMQuery, LLMPrediction, Conversation


class ChatGLM3_6B(AbstractModel):

    def __init__(self, config: ChatGLM3ModelConfig):
        super().__init__()
        self.model_id = "THUDM/ChatGLM3"
        self._model_path = config.model_path
        self._quantize = config.quantize
        self._device = config.device

        self._tokenizer: any = None
        self._model: any = None

    @log_model_loading("THUDM/ChatGLM3")
    def load_model(self):

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        if self._quantize:
            self._model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True).quantize(
                self._quantize).to(
                self._device).eval()
            logger.info(f"Model is loaded as {self._quantize}")
        else:
            self._model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True).to(self._device).eval()
            logger.info(f"Model is loaded without quantization.")
        assert self._tokenizer and self._model

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        """
        Predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan_live_robot_data.data.llm.LLMQuery

        Returns: See zerolan_live_robot_data.data.llm.LLMPrediction

        """
        text, history = self._to_chatglm_format(llm_query)
        # Note: In the new version, past_key_values=None throws IndexError,
        # Because the underlying code does not determine whether past_key_values is None or not,
        # Instead, try to parse as long as there is a past_key_values parameter
        response, history = self._model.chat(self._tokenizer, text, history, top_p=1., temperature=1.)
        logger.debug(response)
        return self._to_pipeline_format(response, history)

    def stream_predict(self, llm_query: LLMQuery):
        """
        Stream predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan_live_robot_data.data.llm.LLMQuery

        Returns: See zerolan_live_robot_data.data.llm.LLMPrediction

        """
        text, history = self._to_chatglm_format(llm_query)
        for response, history, past_key_values in self._model.stream_chat(self._tokenizer, text, history=history,
                                                                          top_p=1.,
                                                                          temperature=1.,
                                                                          past_key_values=None,
                                                                          return_past_key_values=True):
            logger.debug(response)
            yield self._to_pipeline_format(response, history)

    @staticmethod
    def _to_chatglm_format(llm_query: LLMQuery) -> (str, list[dict[str:str]]):
        text = llm_query.text
        history = [{'role': chat.role, 'metadata': '', 'content': chat.content} for chat in llm_query.history]
        return text, history

    @staticmethod
    def _to_pipeline_format(response: str, history: list[dict[str:str]]) -> LLMPrediction:
        history = [Conversation(role=chat['role'], content=chat['content']) for chat in history]
        llm_response = LLMPrediction(response=response, history=history)
        return llm_response
