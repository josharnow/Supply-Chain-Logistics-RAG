"""
Simulates integration of multimodal data sources (e.g., IoT telemetry, geospatial coordinates, and structured logistics metrics) into unstructured tactical SITREPs for RAG analysis.

Target Dataset: Smart Logistics Supply Chain Dataset (Kaggle)
"""
import pandas as pd
import json
import os
import kagglehub

def process_logistics_data(csv_path="data/supply_chain_data.csv", output_path="data/sitrep_data.json"):
    """
    Ingests the structured Smart Logistics Supply Chain dataset and programmatically
    synthesizes it into unstructured, operational tactical SITREPs for RAG analysis.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing base logistics data at {csv_path}")
        
    print(f"Reading structured logistics metrics from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    sitreps = []
    
    # Iterate through the actual schema features
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
        
        # Dynamic narrative block based on actual target variables and delay reasons
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
        
    # Export line-delimited JSON
    with open(output_path, "w") as f:
        for sitrep in sitreps:
            f.write(json.dumps(sitrep) + "\n")
            
    print(f"Successfully compiled {len(sitreps)} tactical SITREPs to {output_path}")

if __name__ == "__main__":
    # Download the dataset from Kaggle
    dataset_path = kagglehub.dataset_download("ziya07/smart-logistics-supply-chain-dataset")

    # Get the path to the CSV file inside the downloaded dataset directory
    if os.path.isdir(dataset_path):
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found inside the directory: {dataset_path}")
        csv_path = os.path.join(dataset_path, csv_files[0])

    # Process the logistics data and generate SITREPs
    process_logistics_data(csv_path)