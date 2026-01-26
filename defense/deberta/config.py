from dataclasses import dataclass


@dataclass
class DebertaPromptDefenseModelConfig:
    model_path: str = "protectai/deberta-v3-base-prompt-injection-v2"
    device: str = "cuda"
    max_length: int = 256