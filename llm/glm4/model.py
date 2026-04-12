"""
More details about the model:
    https://github.com/THUDM/GLM-4
"""
from threading import Thread
from typing import Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation, RoleEnum

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.glm4.config import GLM4ModelConfig


class GLM4_9B_Chat_Hf(AbstractModel):

    def __init__(self, config: GLM4ModelConfig):
        super().__init__()
        self.model_id = "THUDM/glm-4-9b-chat-hf"
        self._model_path = config.model_path
        self._device = config.device
        self._max_length = config.max_length

        self._tokenizer: any = None
        self._model: any = None
        self._streamer: Union[TextIteratorStreamer, None] = None

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
        self._streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        messages = self.to_glm4chat_format(llm_query)

        inputs = self._tokenizer.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            )
        inputs = inputs.to(self._device)
        gen_kwargs = {"max_length": self._max_length, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.debug(output)

        return self.to_pipeline_format(output, llm_query.history)

    def stream_predict(self, llm_query: LLMQuery):
        messages = self.to_glm4chat_format(llm_query)

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = inputs.to(self._device)
        
        generation_kwargs = {
            **inputs,
            "streamer": self._streamer,
            "max_new_tokens": self._max_length,
            "do_sample": True,
            "top_k": 1,
        }
        
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        try:
            history = llm_query.history.copy()
            history.append(Conversation(role=RoleEnum.user, content=llm_query.text))
            history.append(Conversation(role=RoleEnum.assistant, content=""))
            generated_text = ""
            assert self._streamer is not None
            for new_text in self._streamer:
                generated_text += new_text
                history[-1].content = generated_text
                yield LLMPrediction(response=generated_text, history=history)
        except Exception as e:
            logger.exception(e)
        finally:
            thread.join(timeout=10)

    @staticmethod
    def to_glm4chat_format(llm_query: LLMQuery):
        messages = [{"role": chat.role, "content": chat.content} for chat in llm_query.history]
        messages.append({"role": "user", "content": llm_query.text})
        return messages
    
    @staticmethod
    def to_pipeline_format(output: str, history: list[Conversation]):
        history.append(Conversation(role=RoleEnum.assistant, content=output))
        return LLMPrediction(response=output, history=history)