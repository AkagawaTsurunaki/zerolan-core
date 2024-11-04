"""
More details about the model:
    https://gitee.com/paddlepaddle/PaddleOCR
"""
from typing import Any

from paddleocr import PaddleOCR
from zerolan.data.data.ocr import OCRQuery, OCRPrediction, Vector2D, Position, RegionResult

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from ocr.paddle.config import PaddleOCRModelConfig


class PaddleOCRModel(AbstractModel):

    def __init__(self, config: PaddleOCRModelConfig):
        super().__init__()
        # Supported languages: `ch`, `en`, `fr`, `german`, `korean`, `japan`
        self._model_path = config.model_path
        self.lang = config.lang
        self.model = None

    @log_model_loading("paddlepaddle/PaddleOCR")
    def load_model(self):
        self.model = PaddleOCR(use_angle_cls=True, lang=self.lang)
        assert self.model

    def predict(self, query: OCRQuery) -> OCRPrediction:
        result = self.model.ocr(query.img_path, cls=True)
        prediction = OCRPrediction(list())
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                lu, ru, rd, ld = line[0][0], line[0][1], line[0][2], line[0][3]
                lu = Vector2D(x=lu[0], y=lu[1])
                ru = Vector2D(x=ru[0], y=ru[1])
                rd = Vector2D(x=rd[0], y=rd[1])
                ld = Vector2D(x=ld[0], y=ld[1])
                position = Position(lu, ru, rd, ld)
                content, confidence = line[1][0], line[1][1]
                prediction.region_results.append(RegionResult(position, content, confidence))

        return prediction

    def stream_predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError

# ! Result 
# from PIL import Image

# result = result[0]
# image = Image.open(img_path).convert("RGB")
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path="./fonts/simfang.ttf")
# im_show = Image.fromarray(im_show)
# im_show.save("result.jpg")
