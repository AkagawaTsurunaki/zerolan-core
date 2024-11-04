from flask import Flask, request, jsonify
from loguru import logger

from common.abs_app import AbstractApplication
from utils import audio_util, file_util, web_util
from zerolan.data.data.asr import ASRModelQuery, ASRModelStreamQuery


class ASRApplication(AbstractApplication):
    def __init__(self, model, host: str, port: int):
        super().__init__()
        self.host = host
        self.port = port
        self._app = Flask(__name__)
        self._app.add_url_rule(rule='/asr/predict', view_func=self._handle_predict,
                               methods=["GET", "POST"])
        self._app.add_url_rule(rule='/asr/stream-predict', view_func=self._handle_stream_predict,
                               methods=["GET", "POST"])
        self._model = model

    def run(self):
        self._model.load_model()
        self._app.run(self.host, self.port, False)

    def _handle_predict(self):
        logger.info('Request received: processing...')

        query: ASRModelQuery = web_util.get_obj_from_json(request, ASRModelQuery)
        audio_path = web_util.save_request_audio(request, prefix="asr")

        # Convert to mono channel audio file.
        mono_audio_path = file_util.create_temp_file(prefix="asr", suffix=".wav", tmpdir="audio")
        audio_util.convert_to_mono(audio_path, mono_audio_path, query.sample_rate)
        query.audio_path = mono_audio_path

        prediction = self._model.predict(query)

        return jsonify(prediction.to_dict())  # type: ignore[attr-defined]

    def _handle_stream_predict(self):
        query: ASRModelStreamQuery = web_util.get_obj_from_json(request, ASRModelStreamQuery)
        audio_data = web_util.get_request_audio_file(request).stream.read()
        query.audio_data = audio_data

        prediction = self._model.stream_predict(query)
        return jsonify(prediction.to_dict())  # type: ignore[attr-defined]
