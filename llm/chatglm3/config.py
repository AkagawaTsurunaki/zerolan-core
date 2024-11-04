from dataclasses import dataclass


@dataclass
class ChatGLM3ModelConfig:
    model_path: str = "THUDM/chatglm3-6b"
    quantize: int = 4  # None means no-quantization.
    device: str = "cuda"