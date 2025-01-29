# More details about the model:
#     https://github.com/deepseek-ai/DeepSeek-R1
# export HF_ENDPOINT=https://hf-mirror.com
from typing import Any, List

from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation, RoleEnum

from common.abs_model import AbstractModel
from common.decorator import log_model_loading, issue_solver
from llm.deepseek.config import DeepSeekModelConfig


class DeepSeekLLMModel(AbstractModel):

    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self._model_path: str = config.model_path
        self._max_length: int = config.max_length if config.max_length is not None else 23000
        self._model = None

    @log_model_loading("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    @issue_solver()
    def load_model(self):
        self._model = LLM(model=self._model_path, max_model_len=self._max_length, tensor_parallel_size=2)

    def predict(self, llm_query: LLMQuery):
        text, messages = self._to_deepseek_format(llm_query)
        messages.append(ChatCompletionUserMessageParam(content=text, role="user"))
        # See https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
        # Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.
        outputs = self._model.chat(messages, SamplingParams(temperature=0.6))
        response = outputs[0].outputs[0].text
        messages.append(ChatCompletionAssistantMessageParam(content=response, role="assistant"))
        return self._to_pipeline_format(response, messages)

    def stream_predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Not implemented yet")

    @staticmethod
    def _to_pipeline_format(response: str, messages: List[ChatCompletionMessageParam]) -> LLMPrediction:
        history = [Conversation(role=chat['role'], content=chat['content']) for chat in messages]
        llm_response = LLMPrediction(response=response, history=history)
        return llm_response

    @staticmethod
    def _to_deepseek_format(llm_query: LLMQuery):
        text = llm_query.text
        history = []
        for chat in llm_query.history:
            if chat.role == RoleEnum.user:
                history.append(ChatCompletionUserMessageParam(content=chat.content, role="user"))
            elif chat.role == RoleEnum.assistant:
                history.append(ChatCompletionAssistantMessageParam(content=chat.content, role="assistant"))

        return text, history
