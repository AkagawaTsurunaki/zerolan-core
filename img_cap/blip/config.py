from dataclasses import dataclass


@dataclass
class BlipModelConfig:
    model_path: str = "Salesforce/blip-image-captioning-large"
    device: str = "cuda"