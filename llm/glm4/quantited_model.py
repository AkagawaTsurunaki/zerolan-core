"""
More details about this model:
    https://huggingface.co/legraphista/glm-4-9b-chat-GGUF
"""
try:
    from llama_cpp import Llama
except Exception as e:
    raise Exception(e)
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from loguru import logger
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation, RoleEnum

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.glm4.config import GLM4QuantitedModelConfig


class GLM4_9b_chat_GGUF(AbstractModel):

    def __init__(self, config: GLM4QuantitedModelConfig):
        super().__init__()
        self.model_id = "legraphista/glm-4-9b-chat-GGUF"
        self._model_path = config.model_path
        self._max_length = config.max_length
        self._n_gpu_layers = config.n_gpu_layers    # -1 表示全部层卸载到 GPU，0 表示纯 CPU
        self._filename = config.filename

        self._model: any = None

    @log_model_loading("legraphista/glm-4-9b-chat-GGUF")
    def load_model(self):
        # GGUF 格式将分词器和权重打包在一起，无需单独加载 AutoTokenizer
        if os.path.exists(self._model_path) and os.path.isfile(self._model_path):
            # 本地存在模型
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._max_length,            # 上下文长度
                n_gpu_layers=self._n_gpu_layers, # 将多少层卸载到GPU，-1表示全部卸载
                chat_format="chatglm3",
                verbose=False                 # 是否打印详细日志
            )
        else:
            # huggingface 拉取模型，必要前提是已经执行 pip install huggingface-hub
            self._model = Llama.from_pretrained(
                repo_id="THUDM/glm-4-9b-chat-gguf",  # 目标仓库 ID
                filename=self._filename,  # 下载指定量化版本，具体参数参考 huggingface 链接
                n_ctx=self._max_length,
                n_gpu_layers=self._n_gpu_layers,
                chat_format="chatglm3",
                verbose=False
            )


    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        messages = self.to_glm4chat_format(llm_query)

        # 直接调用 create_chat_completion，它会自动处理模板和 Tokenize
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=self._max_length,
            temperature=0.7,
            top_k=1,
            stream=False
        )

        output = response["choices"][0]["message"]["content"]
        # logger.debug(output)

        return self.to_pipeline_format(output, llm_query.history)

    def stream_predict(self, llm_query: LLMQuery):
        raise NotImplementedError()

    @staticmethod
    def to_glm4chat_format(llm_query: LLMQuery):
        messages = [{"role": chat.role, "content": chat.content} for chat in llm_query.history]
        messages.append({"role": "user", "content": llm_query.text})
        return messages
    
    @staticmethod
    def to_pipeline_format(output: str, history: list[Conversation]):
        history.append(Conversation(role=RoleEnum.assistant, content=output))
        return LLMPrediction(response=output, history=history)