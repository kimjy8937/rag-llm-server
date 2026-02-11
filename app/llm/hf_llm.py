from openai import OpenAI
from app.config import HF_TOKEN, MODEL_NAME

class HFLlm:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )

    def chat(self, messages):
        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        return completion.choices[0].message.content
