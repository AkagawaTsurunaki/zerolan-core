from flask import Flask, jsonify
from zerolan.data.data.state import AppStatusEnum, AppStatus

from abc import ABC, abstractmethod

from common.abs_model import AbstractModel


class AbstractApplication(ABC):

    def __init__(self, model: AbstractModel, name: str):
        self.name = name
        self.status = AppStatusEnum.STOPPED
        self._app = Flask(__name__)
        self._app.add_url_rule(rule=f'/{self.name}/status', view_func=self._handle_status,
                               methods=["GET", "POST"])
        assert model, "Model should not be None."
        self.model = model

    def _handle_status(self):
        return jsonify(AppStatus(status=self.status))

    @abstractmethod
    def run(self):
        pass

    def validate_model_id(self, model_id: str):
        return self.model.model_id == model_id
