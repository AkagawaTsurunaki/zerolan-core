from typing import Any

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.yi.config import YiModelConfig
from zerolan.data.pipeline.llm import LLMQuery, Conversation, LLMPrediction
from transformers import AutoModelForCausalLM, AutoTokenizer


class Yi6B_Chat(AbstractModel):

    def __init__(self, config: YiModelConfig):
        super().__init__()
        self.model_id = "01-ai/Yi-6B-Chat"
        self._model = None
        self._tokenizer = None
        self._model_path = config.model_path
        self._device = config.device

    @log_model_loading("01-ai/Yi-6B-Chat")
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, use_fast=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map=self._device,
        ).eval()

    def predict(self, llm_query: LLMQuery) -> Any:
        """
        Predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction

        """
        messages = self._to_yi_format(llm_query)
        # If the erorr occured
        # AttributeError: 'LlamaTokenizer' object has no attribute 'apply_chat_template'
        # Please see here: https://github.com/01-ai/Yi/issues/241
        # > Please check the version of transformers and make sure it is 4.35.0 or above.
        input_ids = self._tokenizer.apply_chat_template(conversation=messages, tokenize=True,
                                                        add_generation_prompt=True,
                                                        return_tensors='pt').to('cuda')
        output_ids = self._model.generate(input_ids)
        response = self._tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return self._to_pipeline_format(response, messages)

    def stream_predict(self, *args, **kwargs) -> Any:
        """
        Stream predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction

        """
        raise NotImplementedError('Stream prediction has not implemented!')

    @staticmethod
    def _to_yi_format(llm_query: LLMQuery):
        messages = [{"role": c.role, "content": c.content} for c in llm_query.history]
        messages.append({"role": "user", "content": llm_query.text})
        return messages

    @staticmethod
    def _to_pipeline_format(response, messages):
        messages.append({"role": "assistant", "content": response})
        history = [Conversation(role=c['role'], content=c['content']) for c in messages]
        return LLMPrediction(response, history)
