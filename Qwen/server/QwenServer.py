from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from io import BytesIO
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# 下载模型(本地)
model_dir = snapshot_download('qwen/Qwen-VL-Chat', cache_dir='E:\LLM-exclusive')

device = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
qwen_vl = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True, fp16=True).eval()

# 创建FastAPI应用实例
app = FastAPI()


# 定义请求体模型，与OpenAI API兼容
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 1024
    temperature: float = 0.7


# 文本生成函数
def generate_text(model: str, messages: list, max_tokens: int, temperature: float):
    text = messages[0]["content"][0]["text"]
    image_url = messages[0]["content"][1]["image_url"]["url"]
    # print(text,image_url)
    query = tokenizer.from_list_format([
        {'image': image_url},  # Either a local path or an url
        {'text': text}
    ])
    response, history = qwen_vl.chat(tokenizer, query=query, history=None, max_new_tokens=max_tokens,
                                     temperature=temperature)
    return response


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 调用自定义的文本生成函数
    response = generate_text(request.model, request.messages, request.max_tokens, request.temperature)
    return {"choices": [{"message": {"content": response}}], "model": request.model}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
