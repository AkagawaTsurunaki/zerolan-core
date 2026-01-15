import os.path

from flask import Flask, request, jsonify, Response
from loguru import logger

from common.abs_app import AbstractApplication
from utils import audio_util, file_util, web_util
from zerolan.data.pipeline.asr import ASRQuery, ASRStreamQuery, ASRPrediction


class ASRApplication(AbstractApplication):
    def __init__(self, model, host: str, port: int):
        super().__init__(model, "asr")
        self.host = host
        self.port = port
        self._app = Flask(__name__)

    def run(self):
        self.model.load_model()
        self.init()
        self._app.run(self.host, self.port, False)

    def init(self):
        @self._app.route('/asr/predict', methods=['POST'])
        def handle_predict():
            logger.info('Request received: processing...')

            query: ASRQuery = web_util.get_obj_from_json(request, ASRQuery)
            if not os.path.exists(query.audio_path):
                audio_path = web_util.save_request_audio(request, prefix="asr")
            else:
                audio_path = query.audio_path

            # Convert to mono channel audio file.
            # Warning: Using ffmpeg for conversion can create performance issues
            if query.channels != 1:
                mono_audio_path = file_util.create_temp_file(prefix="asr", suffix=".wav", tmpdir="audio")
                audio_util.convert_to_mono(audio_path, mono_audio_path, query.sample_rate)
                query.audio_path = mono_audio_path
            else:
                # Fixed: Or it will load original file path (for example local machine)
                query.audio_path = audio_path

            prediction: ASRPrediction = self.model.predict(query)
            logger.info(f"Response: {prediction.transcript}")
            return Response(
                response=prediction.model_dump_json(),
                status=200,
                mimetype='application/json',
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )

        @self._app.route('/asr/stream-predict', methods=['POST'])
        def handle_stream_predict():
            query: ASRStreamQuery = web_util.get_obj_from_json(request, ASRStreamQuery)
            audio_data = web_util.get_request_audio_file(request).stream.read()
            query.audio_data = audio_data

            prediction: ASRPrediction = self.model.stream_predict(query)
            return jsonify(prediction.model_dump())
