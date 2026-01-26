"""
More details about the model:
    https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
"""
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation, RoleEnum

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from defense.deberta.config import DebertaPromptDefenseModelConfig


class DebertaPromptDefenseModel(AbstractModel):

    def __init__(self, config: DebertaPromptDefenseModelConfig):
        super().__init__()
        self.model_id = "protectai/deberta-v3-base-prompt-injection-v2"
        self._model_path = config.model_path
        self._device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"),
        self._max_length = config.max_length

        self._tokenizer: any = None
        self._model: any = None

    @log_model_loading("protectai/deberta-v3-base-prompt-injection-v2")
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path).eval()

    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        classifier = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=self._max_length,
            truncation=True,
            device=self._device
        )

        """
        模型输出格式
        [{'label': 'INJECTION', 'score': 0.99998}]
        [{'label': 'SAFE', 'score': 0.93331}]
        """
        output = classifier(llm_query.text)
        # logger.debug(f'Defense Model Output: {output}')

        return self.to_pipeline_format(output[0], llm_query.history)

    def stream_predict(self, llm_query: LLMQuery):
        raise NotImplementedError()
    
    @staticmethod
    def to_pipeline_format(output: str, history: list[Conversation]):
        history.append(Conversation(role=RoleEnum.function, content=output['label'], metadata=str(output['score'])))
        # 模型输出传输形式 response
        response = f"{output['label']}|{str(output['score'])}"
        return LLMPrediction(response=response, history=history)