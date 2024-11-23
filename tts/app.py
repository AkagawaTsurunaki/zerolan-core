"""
Not implemented
"""
from loguru import logger

from common.abs_app import AbstractApplication


class TTSApplication(AbstractApplication):
    def run(self):
        logger.warning("Not implemented")

    def _handle_predict(self):
        raise NotImplementedError("Not implemented")

    def _handle_stream_predict(self):
        raise NotImplementedError("Not implemented")
