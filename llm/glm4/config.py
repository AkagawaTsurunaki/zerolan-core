from dataclasses import dataclass


@dataclass
class GLM4ModelConfig:
    model_path: str = "THUDM/glm-4-9b-chat-hf"
    device: str = "cuda"
    max_length: int = 5000


@dataclass
class GLM4QuantitedModelConfig:
    model_path: str = "legraphista/glm-4-9b-chat-GGUF"
    max_length: int = 10000
    n_gpu_layers: int = -1
    filename: str = "glm-4-9b-chat.Q2_K.gguf"