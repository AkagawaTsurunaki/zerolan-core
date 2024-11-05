import argparse

import yaml

from common.abs_app import AbstractApplication

parser = argparse.ArgumentParser()
parser.add_argument('service', type=str)
parser.add_argument('--config', type=str)
args = parser.parse_args()


def load_config():
    path = args.config if args.config else './config.yaml'
    with open(path, mode='r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_config = load_config()


def asr_app() -> AbstractApplication:
    from asr.app import ASRApplication

    asr_config = _config["ASR"]
    asr_id = asr_config["id"]
    model_cfg = asr_config["config"][asr_id]

    def get_model():
        if asr_id == "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1":
            from asr.paraformer.model import SpeechParaformerModel as Model
            from asr.paraformer.config import SpeechParaformerModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {asr_id}")

    asr = get_model()
    app = ASRApplication(model=asr, host=asr_config["host"], port=asr_config["port"])
    return app


def llm_app() -> AbstractApplication:
    from llm.app import LLMApplication

    llm_config = _config["LLM"]
    llm_id = llm_config["id"]
    model_cfg = llm_config["config"][llm_id]

    def get_model():
        if llm_id == "THUDM/chatglm3-6b":
            from llm.chatglm3.model import ChatGLM3_6B as Model
            from llm.chatglm3.config import ChatGLM3ModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "Qwen/Qwen-7B-Chat":
            from llm.qwen.model import Qwen7BChat as Model
            from llm.qwen.config import QwenModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "augmxnt/shisa-7b-v1":
            from llm.shisa.model import Shisa7B_V1 as Model
            from llm.shisa.config import ShisaModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "01-ai/Yi-6B-Chat":
            from llm.yi.model import Yi6B_Chat as Model
            from llm.yi.config import YiModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {llm_id}")

    llm = get_model()
    app = LLMApplication(model=llm, host=llm_config["host"], port=llm_config["port"])
    return app


def imgcap_app() -> AbstractApplication:
    from img_cap.app import ImgCapApplication

    imgcap_config = _config["ImgCap"]
    imgcap_id = imgcap_config["id"]
    model_cfg = imgcap_config["config"][imgcap_id]

    def get_model():
        if imgcap_id == "Salesforce/blip-image-captioning-large":
            from img_cap.blip.model import BlipImageCaptioningLarge as Model
            from img_cap.blip.config import BlipModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {imgcap_id}")

    imgcap = get_model()
    app = ImgCapApplication(model=imgcap, host=imgcap_config["host"], port=imgcap_config["port"])
    return app


def ocr_app() -> AbstractApplication:
    from ocr.app import OCRApplication

    ocr_config = _config["OCR"]
    ocr_id = ocr_config['id']
    model_cfg = ocr_config["config"][ocr_id]

    def get_model():
        if ocr_id == "paddlepaddle/PaddleOCR":
            from ocr.paddle.model import PaddleOCRModel as Model
            from ocr.paddle.config import PaddleOCRModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {ocr_id}")

    ocr = get_model()
    app = OCRApplication(model=ocr, host=ocr_config["host"], port=ocr_config["port"])
    return app


def tts_app() -> AbstractApplication:
    from tts.app import TTSApplication

    tts_config = _config["TTS"]
    tts_id = tts_config["id"]
    model_cfg = tts_config["config"][tts_id]

    def get_model():
        if tts_id == "AkagawaTsurunaki/GPT-SoVITS":
            from tts.gpt_sovits.model import GPT_SoVITS as Model
            return Model()

    tts = get_model()
    app = TTSApplication(model=tts, host=tts_config["host"], port=tts_config["port"])
    return app

def get_app(service):
    if "asr" == service:
        return asr_app()
    elif "llm" == service:
        return llm_app()
    elif "imgcap" == service:
        return imgcap_app()
    elif "ocr" == service:
        return ocr_app()
    elif "tts" == service:
        return tts_app()
        
def run(service=None):
    service = args.service if service is None else service
    print(service)
    app = get_app()
    app.run()


run()
