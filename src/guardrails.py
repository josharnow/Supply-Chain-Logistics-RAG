import re

class SecurityGuard:
    def __init__(self):
        # Compiled regex patterns for speed during inference
        self.malicious_patterns = [
            re.compile(r"(?i)ignore\s+previous"),
            re.compile(r"(?i)system\s+prompt"),
            re.compile(r"(?i)bypass"),
            re.compile(r"(?i)jailbreak"),
            re.compile(r"(?i)disregard\s+instructions"),
            re.compile(r"(?i)you\s+are\s+now")
        ]

    def inspect_for_jailbreak(self, query: str) -> int:
        """
        Inspects the user query for prompt injection or jailbreak attempts.
        Returns 1 if a violation is detected, 0 otherwise.
        """
        for pattern in self.malicious_patterns:
            if pattern.search(query):
                return 1
        return 0