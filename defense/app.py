from flask import Flask, request, Response, stream_with_context
from loguru import logger
from zerolan.data.pipeline.defense import DefenseQuery, DefensePrediction

from common.abs_app import AbstractApplication
from common.abs_model import AbstractModel


class DefenseLLMApplication(AbstractApplication):

    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__(model, "defense")
        self.host = host
        self.port = port
        self._app = Flask(__name__)

    def run(self):
        self.model.load_model()
        self.init()
        self._app.run(self.host, self.port, False)

    def init(self):
        @self._app.route("/defense/predict", methods=["POST"])
        def handle_predict():
            llm_query = self._to_pipeline_format()
            p: DefensePrediction = self.model.predict(llm_query)
            logger.info(f'Model response: {p.defense_result} {p.confidence}')
            return Response(
                response=p.model_dump_json(),
                status=200,
                mimetype='application/json',
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )

        @self._app.route("/defense/stream-predict", methods=["POST"])
        def handle_stream_predict():
            # TODO: Will change in the later version.
            llm_query = self._to_pipeline_format()

            def generate_output(q: DefenseQuery):
                with self._app.app_context():
                    for p in self.model.stream_predict(q):
                        p: DefensePrediction
                        logger.info(f'Model response (stream): {p.defense_result} {p.confidence}')
                        yield p.model_dump_json() + '\n'

            return Response(
                stream_with_context(generate_output(llm_query)),
                mimetype='application/json',
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )

    def _to_pipeline_format(self) -> DefenseQuery:
        with self._app.app_context():
            logger.info('Query received: processing...')
            json_val = request.get_json()
            llm_query = DefenseQuery.model_validate(json_val)
            logger.info(f'User Input: {llm_query.text}')
            return llm_query
