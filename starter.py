import argparse

from asr.app import ASRApplication
from llm.app import LLMApplication

parser = argparse.ArgumentParser()
parser.add_argument('--asr', type=str)
parser.add_argument('--llm', type=str)
args = parser.parse_args()


def load_asr():
    asr_id = args.asr
    def get_model():
        if asr_id == "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1":
            import asr.paraformer.model
            return asr.paraformer.model.SpeechParaformerModel()
        else:
            raise NameError(f"No such model name (id) {asr_id}")
    asr = get_model()
    app = ASRApplication(model=asr, host="127.0.0.1", port=11001)


def load_llm():
    llm_id = args.llm
    def get_model():
        if llm_id == "THUDM/chatglm3-6b":
            import llm.chatglm3.model
            return llm.chatglm3.model.ChatGLM3_6B()
        elif llm_id == "Qwen/Qwen-7B-Chat":
            import llm.qwen.model
            return llm.qwen.model.Qwen7BChat()
        elif llm_id == "augmxnt/shisa-7b-v1":
            import llm.shisa.model
            return llm.shisa.model.Shisa7B_V1()
        elif llm_id == "01-ai/Yi-6B-Chat":
            import llm.yi.model
            return llm.yi.model.Yi6B_Chat()
        else:
            raise NameError(f"No such model name (id) {llm_id}")
    llm = get_model()
    app = LLMApplication(model=llm, host="127.0.0.1", port=11002)

