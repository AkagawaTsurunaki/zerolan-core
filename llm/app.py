from dataclasses import asdict

from flask import Flask, jsonify, request, Response, stream_with_context
from loguru import logger

from common.abs_app import AbstractApplication
from common.abs_model import AbstractModel
from zerolan.data.data.llm import LLMQuery


class LLMApplication(AbstractApplication):

    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__("llm")
        self.host = host
        self.port = port
        self._app = Flask(__name__)
        self._app.add_url_rule(rule='/llm/predict', view_func=self._handle_predict,
                               methods=["GET", "POST"])
        self._app.add_url_rule(rule='/llm/stream-predict', view_func=self._handle_stream_predict,
                               methods=["GET", "POST"])
        self._llm = model

    def run(self):
        self._llm.load_model()
        self._app.run(self.host, self.port, False)

    def _to_pipeline_format(self) -> LLMQuery:
        with self._app.app_context():
            logger.info('Query received: processing...')
            json_val = request.get_json()
            llm_query = LLMQuery.from_dict(json_val)
            logger.info(f'User Input {llm_query.text}')
            return llm_query

    def _handle_predict(self):
        llm_query = self._to_pipeline_format()
        p = self._llm.predict(llm_query)
        logger.info(f'Model response: {p.response}')
        return jsonify(asdict(p))

    def _handle_stream_predict(self):
        llm_query = self._to_pipeline_format()

        def generate_output(q: LLMQuery):
            with self._app.app_context():
                for p in self._llm.stream_predict(q):
                    logger.info(f'Model response (stream): {p.response}')
                    yield jsonify(p.to_dict()).data + b'\n'

        return Response(stream_with_context(generate_output(llm_query)), content_type='application/json')
