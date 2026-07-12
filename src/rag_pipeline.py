"""
Core orchestration layer for the Supply-Chain-Logistics-RAG system.
Handles local embedding generation, vector indexing, input guardrails, 
and hardware-optimized inference calls to the local Exo cluster.
"""

import json
import time
import re
import requests
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

type TelemetryMetrics = dict[str, Any]

# --- 1. SECURITY & GUARDRAILS LAYER ---
def inspect_for_jailbreak(user_query: str) -> bool:
    """
    Scans incoming queries for common LLM jailbreak patterns and systemic prompt injections.
    Essential for secure deployment in sensitive defense tech environments.
    """
    jailbreak_patterns = [
        r"(ignore previous instructions|disregard instructions)",
        r"(you are now an unfiltered|act as an unrestricted)",
        r"(system prompt override|sudo mode|developer mode)",
        r"(\[system\]|system:)"
    ]
    
    for pattern in jailbreak_patterns:
        if re.search(pattern, user_query, re.IGNORECASE):
            return True
    return False

# --- 2. LOGGING & OBSERVABILITY ENGINE ---
def log_telemetry(metrics: TelemetryMetrics, log_path="data/telemetry_logs.json"):
    """
    Appends execution metrics to a structured JSON log.
    This log acts as the data source for a local Grafana dashboard.
    """
    metrics["timestamp"] = time.time()
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

# --- 3. CORE PIPELINE ---
def load_sitreps(file_path="data/sitrep_data.json"):
    """Loads synthesized multimodal SITREPs from the ingestion layer."""
    documents = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                documents.append(Document(page_content=data["sitrep_text"]))
    return documents

def create_vector_store(documents):
    """
    Generates local embeddings and constructs an in-memory Chroma vector database index.
    Utilizes standalone packages to comply with the langchain-community sunset.
    """
    # This model runs entirely locally on your M2 GPU/CPU via Apple MLX/Metal
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embedding_model)
    return vector_store

def load_system_prompt(context_str: str, prompt_path="src/prompts/logistics_assistant.md") -> str:
    """Loads the system prompt from a markdown file and injects the runtime context."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"System prompt template missing at: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Dynamically inject our retrieved RAG context into the {context} placeholder
    return template.format(context=context_str)

def query_exo_llm(prompt, context):
    """Routes deterministic context-injected prompts to the local Exo server API."""
    url = "http://localhost:52415/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Dynamically resolve the prompt from the external markdown architecture
    try:
        system_prompt = load_system_prompt(context)
    except Exception as e:
        metrics: TelemetryMetrics = {"inference_latency_sec": 0.0, "status_code": 500, "error_flag": 1}
        return f"Prompt Template Error: {e}", metrics
    
    payload = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-8bit",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0  # Zero out temperature to force strict fact-checking determinism
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        latency = time.time() - start_time
        res_json = response.json()
        
        output_text = res_json["choices"][0]["message"]["content"]
        
        metrics: TelemetryMetrics = {
            "inference_latency_sec": latency,
            "status_code": response.status_code,
            "error_flag": 0
        }
        return output_text, metrics
    except Exception as e:
        metrics: TelemetryMetrics = {"inference_latency_sec": 0.0, "status_code": 500, "error_flag": 1}
        return f"Error connecting to Exo server: {e}", metrics

# --- 4. EXECUTION LOOP ---
if __name__ == "__main__":
    print("Initializing Secure Supply Chain RAG system...")
    docs = load_sitreps()
    
    print("Generating local embeddings and building index...")
    db = create_vector_store(docs)
    
    # -------------------------------------------------------------
    # Test Scenario: Querying for an active mechanical bottleneck
    # -------------------------------------------------------------
    query = "Are there any logistics assets experiencing a Delayed status due to Mechanical Failure?"
    print(f"\n[USER QUERY]: {query}")
    
    # Step 1: Security Scan
    if inspect_for_jailbreak(query):
        print("\n[SECURITY ALERT]: Potential jailbreak attempt blocked.")
        log_telemetry({"query": query, "security_violation": 1, "error_flag": 1})
    else:
        # Step 2: Retrieval Monitoring
        retrieval_start = time.time()
        # Chroma matches the standard LangChain VectorStore interface for similarity_search
        retrieved_docs = db.similarity_search(query, k=2)
        retrieval_time = time.time() - retrieval_start
        
        context_str = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 3: Local Generation via Exo
        print("Sending contextualized payload to local Exo node...")
        answer, telemetry_metrics = query_exo_llm(query, context_str)
        
        # Step 4: Final Telemetry Consolidation
        telemetry_metrics["retrieval_latency_sec"] = retrieval_time
        telemetry_metrics["security_violation"] = 0
        telemetry_metrics["context_chunks_fed"] = len(retrieved_docs)
        
        log_telemetry(telemetry_metrics)
        
        print(f"\n[RESPONSE]:\n{answer}")
        print("\n[MLOps Log Generated]: Metrics written to data/telemetry_logs.json")