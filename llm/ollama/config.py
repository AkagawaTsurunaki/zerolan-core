from dataclasses import dataclass


@dataclass
class OllamaModelConfig:
    api_url: str = "http://localhost:11434"  # Ollama API 地址
    # api_url: str = "http://host.docker.internal:11434"  # Ollama API 地址
    model_name: str = "gemma3:1b"  # Ollama 模型名称
    temperature: float = 0.7
    top_p: float = 0.95
    timeout: int = 60  # 请求超时时间（秒）
