from dataclasses import dataclass


@dataclass
class KotobaWhisper2Config:
    model_path: str = "kotoba-tech/kotoba-whisper-v2.0"
    device: str = "cuda"