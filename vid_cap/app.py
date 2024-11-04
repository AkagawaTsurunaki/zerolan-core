import os.path
from dataclasses import asdict

from flask import Flask, request, jsonify
from loguru import logger

from common.abs_model import AbstractModel
from utils import file_util
from common.abs_app import AbstractApplication
from zerolan.data.data.vid_cap import VidCapQuery


class VidCapApplication(AbstractApplication):

    def __init__(self, model: AbstractModel, host: str, port: int):
        super().__init__()
        self._app = Flask(__name__)
        self._app.add_url_rule(rule='/vid-cap/predict', view_func=self._handle_predict,
                               methods=["GET", "POST"])
        self._model = model

    def run(self):
        self._model.load_model()
        self._app.run(self.host, self.port, False)

    def _to_pipeline_format(self) -> VidCapQuery:
        with self._app.app_context():
            logger.info('Request received: processing...')

            if request.headers['Content-Type'] == 'application/json':
                # If it's in JSON format, then there must be a video location
                json_val = request.get_json()
                query = VidCapQuery.from_dict(json_val)
            elif 'multipart/form-data' in request.headers['Content-Type']:
                # If it's in multipart/form-data format, then try to get the video file
                video_file = request.files.get('video', None)
                if video_file is None:
                    raise ValueError('There is no video data in the request.')
                file_type = video_file.filename.split('.')[-1]
                # file_type = video_file.mimetype.split("/")[-1]
                file_type = 'mp4'
                assert file_type in ["mp4", "avi"], "Unsupported video types"
                temp_file_path = file_util.create_temp_file(prefix="vid-cap", suffix=f".{file_type}", tmpdir="video")
                video_file.save(temp_file_path)

                logger.debug(f"Temporary files are created at: {temp_file_path}")

                query = VidCapQuery(vid_path=temp_file_path)
            else:
                raise NotImplementedError("Unsupported Content-Type.")

            logger.info(f'Location of the video: {query.vid_path}')
            return query

    def _handle_predict(self):
        query = self._to_pipeline_format()
        assert os.path.exists(query.vid_path), ""
        prediction = self._model.predict(query)
        logger.info(f'Model response: {prediction.caption}')
        return jsonify(asdict(prediction))

    def _handle_stream_predict(self):
        raise NotImplementedError()
