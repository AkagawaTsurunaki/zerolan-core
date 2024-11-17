from llm.glm4.model import GLM4_9B_Chat_Hf
from llm.glm4.config import GLM4ModelConfig
from zerolan.data.data.llm import LLMQuery

def test_predict():
    config = GLM4ModelConfig("/home/akagawatsurunaki/models/glm-4-9b-chat-hf")
    model = GLM4_9B_Chat_Hf(config)
    model.load_model()
    p = model.predict(llm_query=LLMQuery(text="你是谁", history=[]))
    print(p.response)