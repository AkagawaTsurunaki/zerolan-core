from dataclasses import dataclass


@dataclass
class ShisaModelConfig:
    model_path: str = "augmxnt/shisa-7b-v1"
    device: str = "cuda"