from dataclasses import dataclass


@dataclass
class DeepSeekModelConfig:
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    max_length: int = 23000
    tensor_parallel_size: int = 1
