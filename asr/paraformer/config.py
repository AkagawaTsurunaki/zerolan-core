from typing import List
from dataclasses import dataclass, field


@dataclass
class SpeechParaformerModelConfig:

    model_path: str = "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1"
    chunk_size: List[int] = field(default_factory=lambda: [0, 10, 5])  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    encoder_chunk_look_back: int = 4  # number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back: int = 1  # number of encoder chunks to lookback for decoder cross-attention
    version: str = "v2.0.4"
    chunk_stride: int = 10 * 960  # chunk_size[1] * 960