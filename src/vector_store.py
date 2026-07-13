import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import SITREP_FILE, EMBEDDING_MODEL, TOP_K_RESULTS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

class DocumentIngestor:
    def __init__(self, data_path: Path = SITREP_FILE):
        self.data_path = data_path
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.documents = []
        self.embeddings = None

    def load_and_embed(self) -> None:
        """Loads logistics JSON/JSONL, chunks it, and generates embeddings in memory."""
        try:
            self.documents = []
            with open(self.data_path, "r") as f:
                # Read line by line to gracefully handle JSONL format
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.documents.append(json.dumps(item))
                
            if self.documents:
                self.embeddings = self.model.encode(self.documents)
                
        except FileNotFoundError:
            print(f"[!] Warning: Data file not found at {self.data_path}")
            self.documents = []
        except json.JSONDecodeError as e:
            print(f"[!] JSON parsing error: {e}")
            self.documents = []

    def query_local_db(self, query: str, top_k: int = TOP_K_RESULTS) -> str:
        """Embeds the query, finds the closest documents, and returns the context string."""
        if self.embeddings is None or len(self.documents) == 0:
            return "No local supply chain data available."

        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / norms
        
        # Extract top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format the retrieved context
        retrieved_chunks = [self.documents[i] for i in top_indices]
        return "\n---\n".join(retrieved_chunks)