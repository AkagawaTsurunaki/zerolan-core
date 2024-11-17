from dataclasses import dataclass


@dataclass
class GLM4ModelConfig:
    model_path: str = "THUDM/glm-4-9b-chat-hf"
    device: str = "cuda"