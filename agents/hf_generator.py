# agents/hf_generator.py
import os
import requests

class HFGenerator:
    def __init__(self, model: str = "google/flan-t5-small"):
        self.token = os.environ["HF_API_TOKEN"]
        self.url = f"https://api-inference.huggingface.co/models/{model}"

    def generate(self, prompt: str, max_length: int = 200):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_length": max_length}
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # HF returns a list of {generated_text: ...}
        return data[0]["generated_text"]
