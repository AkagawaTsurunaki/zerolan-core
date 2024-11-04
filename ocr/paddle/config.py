from dataclasses import dataclass
from typing import Literal


@dataclass
class PaddleOCRModelConfig:
    model_path: str = "paddlepaddle/PaddleOCR"
    lang: Literal["ch", "en", "fr", "german", "korean", "japan"] = "ch"