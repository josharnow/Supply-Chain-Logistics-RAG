import os
import json
import pandas as pd
import kagglehub
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

    def fetch_and_process_data(self) -> None:
        """
        Ingests the Smart Logistics dataset from Kaggle and synthesizes it 
        into unstructured tactical SITREPs for RAG analysis.
        """
        if os.path.exists(self.data_path):
            return  # Data already exists, skip download

        print("Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("ziya07/smart-logistics-supply-chain-dataset")

        if os.path.isdir(dataset_path):
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found inside the directory: {dataset_path}")
            csv_path = os.path.join(dataset_path, csv_files[0])
        else:
            raise NotADirectoryError(f"Dataset path {dataset_path} is invalid.")
            
        print(f"Reading structured logistics metrics from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        sitreps = []
        for _, row in df.iterrows():
            asset_id = row['Asset_ID']
            status = str(row['Shipment_Status']).upper()
            lat, lon = row['Latitude'], row['Longitude']
            traffic = row['Traffic_Status']
            delay_reason = row['Logistics_Delay_Reason']
            
            # Core operational telemetry string
            base_telemetry = (
                f"SITREP LOG [{row['Timestamp']}]: Logistics Asset {asset_id} positioned at Coordinates ({lat}, {lon}). "
                f"Current Inventory Volume: {row['Inventory_Level']} units. Asset Utilization Capacity: {row['Asset_Utilization']:.1%}. "
                f"Environmental Sensors: {row['Temperature']}°C, {row['Humidity']}% Humidity. "
            )
            
            # Dynamic narrative block based on actual target variables
            if row['Logistics_Delay'] == 1 or status == "DELAYED":
                narrative = (
                    f"{base_telemetry}OPERATIONAL ALERT: Unit status is non-nominal ({status}). "
                    f"Current transit environment profile indicates {traffic} conditions with an active waiting bottleneck of {row['Waiting_Time']} minutes. "
                    f"Primary root cause of logistics disruption verified as: {delay_reason}. "
                    f"Anticipated localized Demand Forecast for upcoming sector window: {row['Demand_Forecast']} units."
                )
            else:
                narrative = (
                    f"{base_telemetry}OPERATIONAL UPDATE: Unit tracking nominal ({status}). "
                    f"Transit environment status is {traffic}. Localized delay risk is minimal. "
                    f"Current bottleneck queue time is {row['Waiting_Time']} minutes. "
                    f"Anticipated localized Demand Forecast for upcoming sector window: {row['Demand_Forecast']} units."
                )
                
            sitreps.append({"sitrep_text": narrative})
            
        # Ensure directory exists and export line-delimited JSON
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w") as f:
            for sitrep in sitreps:
                f.write(json.dumps(sitrep) + "\n")
                
        print(f"Successfully compiled {len(sitreps)} tactical SITREPs to {self.data_path}")

    def load_and_embed(self) -> None:
        """Ensures data is present, loads JSONL, chunks it, and generates embeddings."""
        # 1. Ensure the data exists (runs Kaggle download if missing)
        self.fetch_and_process_data()

        try:
            self.documents = []
            with open(self.data_path, "r") as f:
                # Read line by line to gracefully handle JSONL format
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Target the specific 'sitrep_text' key your ingestion script built
                        if "sitrep_text" in item:
                            self.documents.append(item["sitrep_text"])
                        else:
                            self.documents.append(json.dumps(item))
                
            if self.documents:
                self.embeddings = self.model.encode(self.documents)
                
        except Exception as e:
            print(f"[!] Data loading error: {e}")
            self.documents = []

    def query_local_db(self, query: str, top_k: int = TOP_K_RESULTS) -> str:
        """Embeds the query, finds the closest documents, and returns the context string."""
        if self.embeddings is None or len(self.documents) == 0:
            return "No local supply chain data available."

        query_embedding = self.model.encode([query])[0]
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / norms
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_chunks = [self.documents[i] for i in top_indices]
        return "\n---\n".join(retrieved_chunks)