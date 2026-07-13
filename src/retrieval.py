import json
from config import SITREP_FILE, PROMPT_FILE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

class ContextRetriever:
    def __init__(self, data_path: Path = SITREP_FILE, prompt_path: Path = PROMPT_FILE):
        self.data_path = data_path
        self.prompt_path = prompt_path

    def load_logistics_data(self) -> dict:
        """Loads the raw supply chain sitrep data."""
        try:
            with open(self.data_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"error": "Sitrep data not found."}

    def evaluate_security(self, query: str) -> int:
        """Simple heuristic guardrail. Returns 1 if violation detected, 0 otherwise."""
        malicious_keywords = ["ignore", "bypass", "jailbreak", "system prompt"]
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in malicious_keywords):
            return 1
        return 0

    def build_prompt(self, user_query: str) -> tuple[str, int]:
        """Assembles the final prompt and returns the security status."""
        security_flag = self.evaluate_security(user_query)
        
        # Load the base system instructions
        try:
            with open(self.prompt_path, "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = "You are a logistics assistant."

        # Load context
        context_data = self.load_logistics_data()
        
        # Assemble
        final_prompt = f"{system_prompt}\n\nContext:\n{json.dumps(context_data)}\n\nUser Query: {user_query}"
        
        return final_prompt, security_flag