from dataclasses import dataclass
from typing import Literal


@dataclass
class HiteaBaseModelConfig:
    model_path: str = "damo/multi-modal_hitea_video-captioning_base_en"
    task: Literal["video-captioning"] = "video-captioning"