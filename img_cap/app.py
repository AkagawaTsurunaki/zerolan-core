from flask import Flask, request, jsonify
from loguru import logger
from zerolan.data.pipeline.img_cap import ImgCapQuery, ImgCapPrediction

from common.abs_app import AbstractApplication
from common.abs_model import AbstractModel
from utils import web_util


class ImgCapApplication(AbstractApplication):

    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__(model, "image-captioning")
        self.host = host
        self.port = port
        self._app = Flask(__name__)

    def run(self):
        self.model.load_model()
        self.init()
        self._app.run(self.host, self.port, False)

    def _to_pipeline_format(self) -> ImgCapQuery:
        with self._app.app_context():
            logger.info('Request received: processing...')

            if request.headers['Content-Type'] == 'application/json':
                # If it's in JSON format, then there must be an image location.
                json_val = request.get_json()
                query = ImgCapQuery.model_validate(json_val)
            elif 'multipart/form-data' in request.headers['Content-Type']:
                query: ImgCapQuery = web_util.get_obj_from_json(request, ImgCapQuery)
                query.img_path = web_util.save_request_image(request, prefix="imgcap")
            else:
                raise NotImplementedError("Unsupported Content-Type.")

            logger.info(f'Location of the image: {query.img_path}')
            return query

    def init(self):
        @self._app.route('/img-cap/predict', methods=['POST'])
        def _handle_predict():
            query = self._to_pipeline_format()
            prediction: ImgCapPrediction = self.model.predict(query)
            logger.info(f'Model response: {prediction.caption}')
            return jsonify(prediction.model_dump())

        @self._app.route('/img-cap/stream-predict', methods=['POST'])
        def _handle_stream_predict():
            raise NotImplementedError("Not Implemented!")
