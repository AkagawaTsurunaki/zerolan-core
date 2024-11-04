from dataclasses import asdict

from flask import Flask, request, jsonify
from loguru import logger

from common.abs_model import AbstractModel
from utils import web_util
from common.abs_app import AbstractApplication
from zerolan_live_robot_data.data.img_cap import ImgCapQuery


class ImgCapApplication(AbstractApplication):

    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__()
        self._app = Flask(__name__)
        # Warning: Here! Compatible with legacy APIs.
        self._app.add_url_rule(rule='/image-captioning/predict', view_func=self._handle_predict,
                               methods=["GET", "POST"])
        self._app.add_url_rule(rule='/img-cap/predict', view_func=self._handle_predict,
                               methods=["GET", "POST"])
        self._model = model

    def run(self):
        self._model.load_model()
        self._app.run(self.host, self.port, False)

    def _to_pipeline_format(self) -> ImgCapQuery:
        with self._app.app_context():
            logger.info('Request received: processing...')

            if request.headers['Content-Type'] == 'application/json':
                # If it's in JSON format, then there must be an image location.
                json_val = request.get_json()
                query = ImgCapQuery.from_dict(json_val)
            elif 'multipart/form-data' in request.headers['Content-Type']:
                query: ImgCapQuery = web_util.get_obj_from_json(request, ImgCapQuery)
                query.img_path = web_util.save_request_image(request, prefix="imgcap")
            else:
                raise NotImplementedError("Unsupported Content-Type.")

            logger.info(f'Location of the image: {query.img_path}')
            return query

    def _handle_predict(self):
        query = self._to_pipeline_format()
        prediction = self._model.predict(query)
        logger.info(f'Model response: {prediction.caption}')
        return jsonify(asdict(prediction))

    def _handle_stream_predict(self):
        raise NotImplementedError("Not Implemented!")
