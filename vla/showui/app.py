from common.abs_app import AbstractApplication
from common.abs_model import AbstractModel
from flask import Flask, jsonify, request, Response, stream_with_context
from zerolan.data.pipeline.vla import ShowUiQuery, ShowUiPrediction
from loguru import logger

from utils import web_util

class ShowUIApplication(AbstractApplication):
    
    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__("vla/showui")
        self.host = host
        self.port = port
        self._app = Flask(__name__)
        self._app.add_url_rule(rule='/vla/showui/predict', view_func=self._handle_predict,
                               methods=["GET", "POST"])
        self._app.add_url_rule(rule='/vla/showui/stream-predict', view_func=self._handle_stream_predict,
                               methods=["GET", "POST"])
        self._model = model
    
    def run(self):
        self._model.load_model()
        self._app.run(self.host, self.port, False)

    def _to_pipeline_format(self) -> ShowUiQuery:
        with self._app.app_context():
            logger.info('Request received: processing...')

            if request.headers['Content-Type'] == 'application/json':
                # If it's in JSON format, then there must be an image location.
                json_val = request.get_json()
                query = ShowUiQuery.model_validate(json_val)
            elif 'multipart/form-data' in request.headers['Content-Type']:
                query: ShowUiQuery = web_util.get_obj_from_json(request, ShowUiQuery)
                query.img_path = web_util.save_request_image(request, prefix="showui")
            else:
                raise NotImplementedError("Unsupported Content-Type.")

            logger.info(f'Location of the image: {query.img_path}')
            return query

    def _handle_predict(self):
        query = self._to_pipeline_format()
        prediction: ShowUiPrediction = self._model.predict(query)
        logger.info(f'Model response: {prediction.click_xy}')
        return jsonify(prediction.model_dump())

    def _handle_stream_predict(self):
        raise NotImplementedError("Not Implemented!")