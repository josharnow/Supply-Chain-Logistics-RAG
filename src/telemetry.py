import json
import time
from config import LOG_FILE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

class TelemetryLogger:
    def __init__(self, log_file: Path = LOG_FILE):
        self.log_file = log_file

    def log_inference(self, latency_sec: float, total_tokens: int, security_violation: int) -> None:
        """Appends a structured JSON log entry for Grafana Alloy to ingest."""
        payload = {
            "inference_latency_sec": round(latency_sec, 4),
            "total_tokens": total_tokens,
            "security_violation": security_violation,
            "timestamp": time.time()
        }
        
        # Append as a single JSON line
        with open(self.log_file, "a") as f:
            f.write(json.dumps(payload) + "\n")