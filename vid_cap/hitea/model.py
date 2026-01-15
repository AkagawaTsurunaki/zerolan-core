"""
More details about the model:
    https://www.modelscope.cn/models/iic/multi-modal_hitea_video-captioning_base_en
"""
import os.path
from typing import Any

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from zerolan.data.pipeline.vid_cap import VidCapQuery, VidCapPrediction

from vid_cap.hitea.config import HiteaBaseModelConfig


# Some issues:
#   1. You need pip with verison 24.0 to install fairseq
#       https://github.com/facebookresearch/fairseq/issues/5518
#   2. Running model with error raised
#       TypeError: VideoCaptioningPipeline: HiTeAForAllTasks: 'NoneType' object is not callable
#       https://github.com/modelscope/modelscope/issues/265
#       Please install fairscale

class HiteaBaseModel(AbstractModel):
    def __init__(self, config: HiteaBaseModelConfig):
        super().__init__()
        self._model = None
        self._lang = "en"
        self._model_path = config.model_path

    @log_model_loading("damo/multi-modal_hitea_video-captioning_base_en")
    def load_model(self):
        self._model = pipeline(Tasks.video_captioning, model=self._model_path)
        assert self._model is not None, "模型加载失败"

    def predict(self, query: VidCapQuery) -> VidCapPrediction:
        assert os.path.exists(query.vid_path), f"视频路径不存在：{query.vid_path}"
        caption = self._model(query.vid_path)['caption']
        return VidCapPrediction(caption=caption, lang=self._lang)

    def stream_predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
