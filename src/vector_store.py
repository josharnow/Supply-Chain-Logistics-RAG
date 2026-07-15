import os
import json
import pandas as pd
import kagglehub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import SITREP_FILE, EMBEDDING_MODEL, TOP_K_RESULTS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

class DocumentIngestor:
    def __init__(self, data_path: Path = SITREP_FILE):
        self.data_path = data_path
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None

    def fetch_and_process_data(self) -> None:
        """Downloads Kaggle data and synthesizes SITREPs (Logic unchanged)."""
        if os.path.exists(self.data_path):
            return 

        print("Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("ziya07/smart-logistics-supply-chain-dataset")
        csv_path = os.path.join(dataset_path, [f for f in os.listdir(dataset_path) if f.endswith('.csv')][0])
        df = pd.read_csv(csv_path)
        
        sitreps = []
        for _, row in df.iterrows():
            asset_id = row['Asset_ID']
            status = str(row['Shipment_Status']).upper()
            
            narrative = (
                f"SITREP LOG [{row['Timestamp']}]: Logistics Asset {asset_id}. "
                f"Status: {status}. Traffic: {row['Traffic_Status']}. "
                f"Delay Reason: {row['Logistics_Delay_Reason']}. "
                f"Environmental Sensors: {row['Temperature']}°C, {row['Humidity']}% Humidity."
            )
            sitreps.append({"sitrep_text": narrative})
            
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w") as f:
            for sitrep in sitreps:
                f.write(json.dumps(sitrep) + "\n")

    def load_and_embed(self) -> None:
        """Loads JSONL and builds a LangChain FAISS vector store."""
        self.fetch_and_process_data()

        docs = []
        try:
            with open(self.data_path, "r") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        text_content = item.get("sitrep_text", json.dumps(item))
                        # Create LangChain Document objects
                        docs.append(Document(page_content=text_content))
                
            if docs:
                # Initialize FAISS vector database
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
                
        except Exception as e:
            print(f"[!] LangChain FAISS loading error: {e}")

    def query_local_db(self, query: str, top_k: int = TOP_K_RESULTS) -> str:
        """Uses LangChain's retriever to fetch documents."""
        if not self.vector_store:
            return "No local supply chain data available."

        # Perform similarity search via LangChain (using cosine similarity)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k}) # Wrap the database in a standardized LangChain "Retriever" interface so it can easily plug into other components (like LLM chains)
        # Steps for invoke():
        # 1. Embedding: Takes plain text query and converts it into a vector (an array of numbers) using predefined embedding model
        # 2. Similarity Search: Compares that query vector against all the document vectors in the database to find the closest mathematical matches
        # 3. Filtering: It applies the rule set earlier (grabbing only the top k results).
        # 4. Returning: Outputs retrieved_docs, which is a list of LangChain Document objects containing the relevant text and any associated metadata
        retrieved_docs = retriever.invoke(query)
        
        return "\n---\n".join([doc.page_content for doc in retrieved_docs])