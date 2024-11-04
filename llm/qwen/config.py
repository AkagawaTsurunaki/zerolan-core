from dataclasses import dataclass
from typing import Literal


@dataclass
class QwenModelConfig:
    model_path: str = "Qwen/Qwen-7B-Chat"
    quantize: int = 4  # None means no-quantization.
    device: str = "cuda"
    precise: Literal["bf16", "fp16"] = None