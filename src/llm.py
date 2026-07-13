import time
import requests
from config import MODEL_NAME, QUANTIZATION, PROMPT_FILE

class InferenceEngine:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.quantization = QUANTIZATION
        self.api_url = "http://localhost:52415/v1/chat/completions"
        
        try:
            with open(PROMPT_FILE, "r") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            self.system_prompt = "You are a logistics assistant answering supply chain questions."

    def generate(self, user_query: str, retrieved_context: str) -> tuple[str, float, int]:
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": f"{self.system_prompt}\n\nContext:\n{retrieved_context}"},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            # Check for Exo server errors before trying to parse the JSON
            if not response.ok:
                print(f"\n[!] Exo API Error ({response.status_code}): {response.text}")
                response.raise_for_status()
                
            data = response.json()
            response_text = data["choices"][0]["message"]["content"]
            
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            response_tokens = data.get("usage", {}).get("completion_tokens", 0)
            total_tokens = prompt_tokens + response_tokens
            
        except requests.exceptions.RequestException as e:
            print(f"\n[!] Local Exo node connection failed: {e}")
            response_text = "Error: Could not communicate with local Exo inference node."
            total_tokens = 0
            
        latency = time.time() - start_time
        
        return response_text, latency, total_tokens