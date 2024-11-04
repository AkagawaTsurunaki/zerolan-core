"""
More details about the model:
    https://huggingface.co/Salesforce/blip-image-captioning-large
"""

from typing import Any

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from ocr.paddle.config import PaddleOCRModelConfig
from zerolan.data.data.img_cap import ImgCapQuery, ImgCapPrediction


class BlipImageCaptioningLarge(AbstractModel):

    def __init__(self, config: PaddleOCRModelConfig):
        super().__init__()
        self.model_id = "Salesforce/blip-image-captioning-large"
        self._lang = 'en'
        self._model = None
        self._processor = None
        self._model_path = config.model_path
        self._device = config.device

    @log_model_loading("Salesforce/blip-image-captioning-large")
    def load_model(self):
        self._processor = BlipProcessor.from_pretrained(self._model_path)
        self._model = BlipForConditionalGeneration.from_pretrained(self._model_path, torch_dtype=torch.float16).to(
            self._device)

    def predict(self, query: ImgCapQuery) -> ImgCapPrediction:
        raw_image = Image.open(query.img_path)
        inputs = self._processor(raw_image, query.prompt, return_tensors="pt").to(self._device, torch.float16)

        out = self._model.generate(**inputs)
        output_text = self._processor.decode(out[0], skip_special_tokens=True)

        return ImgCapPrediction(caption=output_text, lang="en")

    def stream_predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Stream prediction has not implemented!")
