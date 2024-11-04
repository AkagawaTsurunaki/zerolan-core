"""
Not implemented
"""
from zerolan_live_robot_core.abs_app import AbstractApplication
from loguru import logger


class TTSApplication(AbstractApplication):
    def run(self):
        logger.warning("Not implemented")

    def _handle_predict(self):
        raise NotImplementedError("Not implemented")

    def _handle_stream_predict(self):
        raise NotImplementedError("Not implemented")
