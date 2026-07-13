import os
from pathlib import Path

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
LOG_FILE = DATA_DIR / "telemetry_logs.json"
SITREP_FILE = DATA_DIR / "sitrep_data.json"

# Prompt Paths
PROMPT_FILE = BASE_DIR / "src" / "prompts" / "logistics_assistant.md"

# Model & System Configuration
# This model is smaller for laptop use. A bigger model should be used in production to prevent hallucinations and improve accuracy.
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-8bit"
QUANTIZATION = "8-bit"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 15