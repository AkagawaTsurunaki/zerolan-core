import os.path

from flask import Flask, request, jsonify
from loguru import logger
from zerolan.data.pipeline.ocr import OCRQuery, OCRPrediction

from common.abs_app import AbstractApplication
from common.abs_model import AbstractModel
from utils import web_util


class OCRApplication(AbstractApplication):
    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__(model, "ocr")
        self.host = host
        self.port = port
        self._app = Flask(__name__)
        self._model = model

    def run(self):
        self._model.load_model()
        self.init()
        self._app.run(self.host, self.port, False)

    def _to_pipeline_format(self) -> OCRQuery:
        with self._app.app_context():
            logger.info('Request received: processing...')

            if request.headers['Content-Type'] == 'application/json':
                # If it's in JSON format, then there must be an image location.
                json_val = request.get_json()
                query = OCRQuery.model_validate(json_val)
            elif 'multipart/form-data' in request.headers['Content-Type']:
                # If it's in multipart/form-data format, then try to get the image file.
                img_path = web_util.save_request_image(request, prefix="ocr")
                query = OCRQuery(img_path=img_path)
            else:
                raise NotImplementedError("Unsupported Content-Type.")

            logger.info(f'Location of the image: {query.img_path}')
            return query

    def init(self):
        @self._app.route("/ocr/predict", methods=["POST"])
        def _handle_predict():
            query = self._to_pipeline_format()
            assert os.path.exists(query.img_path), f"The image file does not exist: {query.img_path}"
            prediction: OCRPrediction = self._model.predict(query)
            logger.info(f"Model response: {prediction}")
            return jsonify(prediction.model_dump())

        @self._app.route("/ocr/stream-predict", methods=["POST"])
        def _handle_stream_predict():
            raise NotImplementedError()
