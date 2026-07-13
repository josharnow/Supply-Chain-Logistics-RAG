import time
from config import MODEL_NAME, QUANTIZATION, PROMPT_FILE

class InferenceEngine:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.quantization = QUANTIZATION
        
        # Load the base system prompt instructions
        try:
            with open(PROMPT_FILE, "r") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            self.system_prompt = "You are a logistics assistant answering supply chain questions."

    def generate(self, user_query: str, retrieved_context: str) -> tuple[str, float, int]:
        """
        Formats the final prompt and processes it through the local Exo node.
        Returns: (Response Text, Latency in Seconds, Total Tokens)
        """
        final_prompt = f"{self.system_prompt}\n\nContext:\n{retrieved_context}\n\nUser Query: {user_query}"
        
        start_time = time.time()
        
        # --- LOCAL INFERENCE EXECUTION ---
        # Replace this simulation with your actual Exo or local model generation call
        # response_text = exo.generate(final_prompt)
        time.sleep(1.5) 
        response_text = "Based on the provided supply chain context, the logistics nodes are operational."
        # ---------------------------------
        
        latency = time.time() - start_time
        
        # Naive token approximation (words * 1.3)
        prompt_tokens = int(len(final_prompt.split()) * 1.3)
        response_tokens = int(len(response_text.split()) * 1.3)
        total_tokens = prompt_tokens + response_tokens
        
        return response_text, latency, total_tokens