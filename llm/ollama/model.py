"""
Ollama API integration for ZerolanCore
Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

Thanks to [SLLYING](https://github.com/sllying)'s contribution
Reviewed and tested by [AkagawaTsurunaki](https://github.com/AkagawaTsurunaki)
"""
import json
import requests
from loguru import logger
from typing import Iterator

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.ollama.config import OllamaModelConfig
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation


class OllamaModel(AbstractModel):

    def __init__(self, config: OllamaModelConfig):
        super().__init__()
        self.model_id = "ollama"
        self._api_url = config.api_url.rstrip('/')
        self._model_name = config.model_name
        self._temperature = config.temperature
        self._top_p = config.top_p
        self._timeout = config.timeout

    @log_model_loading("Ollama")
    def load_model(self):
        """
        验证 Ollama 服务是否可用
        """
        try:
            # 检查 Ollama 服务是否运行
            response = requests.get(f"{self._api_url}/api/tags", timeout=self._timeout)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"Ollama 服务已连接，可用模型: {model_names}")
                
                # 检查指定的模型是否存在
                if not any(self._model_name in name for name in model_names):
                    logger.warning(f"模型 '{self._model_name}' 未找到，但服务正常运行")
                    raise RuntimeError(
                        f"Ollama model '{self._model_name}' not found. "
                        f"Run `ollama pull {self._model_name}`."
                    )
                else:
                    logger.info(f"模型 '{self._model_name}' 已就绪")
            else:
                logger.error(f"Ollama 服务响应异常: {response.status_code}")
                raise RuntimeError(f"Ollama service error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到 Ollama 服务 ({self._api_url}): {e}")
            logger.warning("请确保 Ollama 服务正在运行")
            raise RuntimeError(f"Ollama service unavailable: {self._api_url}") from e

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        """
        使用 Ollama API 进行推理
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction
        """
        messages = self._to_ollama_format(llm_query)
        
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p
            }
        }

        try:
            response = requests.post(
                f"{self._api_url}/api/chat",
                json=payload,
                timeout=self._timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if "error" in result:
                raise RuntimeError(result["error"])
            if "message" not in result or "content" not in result["message"]:
                raise RuntimeError("Ollama response missing message content.")
            assistant_message = result['message']['content']
            
            logger.debug(f"Ollama 响应: {assistant_message}")
            
            return self._to_pipeline_format(assistant_message, messages)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 请求失败: {e}")
            raise

    def stream_predict(self, llm_query: LLMQuery) -> Iterator[LLMPrediction]:
        """
        使用 Ollama API 进行流式推理
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: Iterator of zerolan.data.pipeline.llm.LLMPrediction
        """
        messages = self._to_ollama_format(llm_query)
        
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p
            }
        }

        try:
            response = requests.post(
                f"{self._api_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self._timeout
            )
            response.raise_for_status()
            
            accumulated_response = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if "error" in chunk:
                            raise RuntimeError(chunk["error"])
                        if 'message' in chunk:
                            content = chunk['message'].get('content', '')
                            accumulated_response += content
                            
                            logger.debug(f"流式响应块: {content}")
                            
                            yield self._to_pipeline_format(accumulated_response, messages)
                            
                            # 如果是最后一个块
                            if chunk.get('done', False):
                                break
                        elif chunk.get('done', False):
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析流式响应失败: {e}")
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 流式 API 请求失败: {e}")
            raise

    @staticmethod
    def _to_ollama_format(llm_query: LLMQuery) -> list[dict]:
        """
        转换为 Ollama API 格式
        Ollama 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        messages = []

        # 添加历史对话
        for chat in llm_query.history:
            messages.append({
                "role": OllamaModel._role_to_str(chat.role),
                "content": chat.content
            })
        
        # 添加当前用户输入
        messages.append({
            "role": "user",
            "content": llm_query.text
        })
        
        return messages

    @staticmethod
    def _to_pipeline_format(response: str, messages: list[dict]) -> LLMPrediction:
        """
        转换为 Pipeline 标准格式
        """
        # Include the current user turn from messages, then append assistant response.
        new_history = [
            Conversation(role=OllamaModel._role_to_str(m["role"]), content=m["content"])
            for m in messages
        ]
        new_history.append(Conversation(role="assistant", content=response))

        return LLMPrediction(response=response, history=new_history)

    @staticmethod
    def _role_to_str(role) -> str:
        if hasattr(role, "value"):
            role = role.value
        if isinstance(role, str):
            return role
        return str(role)
