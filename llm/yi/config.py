from dataclasses import dataclass


@dataclass
class YiModelConfig:
    model_path: str = "01-ai/Yi-6B-Chat"
    device: str = "auto"