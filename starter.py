import argparse

import yaml

from common.abs_app import AbstractApplication

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--asr', type=str)
parser.add_argument('--llm', type=str)
parser.add_argument('--imgcap', type=str)
parser.add_argument('--ocr', type=str)
parser.add_argument('--tts', type=str)
args = parser.parse_args()


def load_config():
    path = args.config if args.config else './config.yaml'
    with open(path, mode='r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_config = load_config()


def asr_app(asr_id) -> AbstractApplication:
    from asr.app import ASRApplication

    asr_config = _config["ASR"]
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


def llm_app(llm_id) -> AbstractApplication:
    from llm.app import LLMApplication

    llm_config = _config["LLM"]
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


def imgcap_app(imgcap_id) -> AbstractApplication:
    from img_cap.app import ImgCapApplication

    imgcap_config = _config["ImgCap"]
    model_cfg = imgcap_config["config"][imgcap_id]

    def get_model():
        if imgcap_id == "Salesforce/blip-image-captioning-large":
            from img_cap.blip.model import BlipImageCaptioningLarge as Model
            from img_cap.blip.config import BlipModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {imgcap_id}")

    imgcap = get_model()
    app = ImgCapApplication(model=imgcap, host=model_cfg["host"], port=model_cfg["port"])
    return app


def ocr_app(ocr_id) -> AbstractApplication:
    from ocr.app import OCRApplication

    ocr_config = _config["OCR"]
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


def tts_app(tts_id) -> AbstractApplication:
    from tts.app import TTSApplication

    tts_config = _config["TTS"]
    model_cfg = tts_config["config"][tts_id]

    def get_model():
        if tts_id == "AkagawaTsurunaki/GPT-SoVITS":
            from tts.gpt_sovits.model import GPT_SoVITS as Model
            return Model()

    tts = get_model()
    app = TTSApplication(model=tts, host=tts_config["host"], port=tts_config["port"])
    return app


def run():
    def get_app():
        if args.asr is not None:
            return asr_app(args.asr)
        elif args.llm is not None:
            return llm_app(args.llm)
        elif args.imgcap is not None:
            return imgcap_app(args.imgcap)
        elif args.ocr is not None:
            return ocr_app(args.ocr)
        elif args.tts is not None:
            return tts_app(args.tts)

    app = get_app()
    app.run()


run()
