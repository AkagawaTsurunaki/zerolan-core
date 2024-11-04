from dataclasses import dataclass

from typing_extensions import Literal


@dataclass
class BlipModelConfig:
    model_path: str = "Salesforce/blip-image-captioning-large"
    lang: Literal["ch", "en", "fr", "german", "korean", "japan"] = "ch"
    device: str = "cuda"