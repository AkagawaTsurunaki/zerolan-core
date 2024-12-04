from dataclasses import dataclass


@dataclass
class ShowUIModelConfig:
    min_pixels: int = 256*28*28
    max_pixels: int = 1344*28*28
    showui_model_path: str = "showlab/ShowUI-2B"
    showui_model_device: str = "auto"
    qwen_vl_2b_instruct_model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
