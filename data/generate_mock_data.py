import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
import os
# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (your project root)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the parent directory to Python's system path
sys.path.append(parent_dir)
# Import the new 36-feature contract from your config file
from src.config import EXPECTED_METRICS, MONITORED_SERVICES

def generate_synthetic_telemetry(num_rows=5000):
    """Generates a mock dataset mapping to the 36 telemetry features."""
    print(f"Generating {num_rows} rows of mock telemetry for {len(MONITORED_SERVICES)} services...")
    
    data = {}
    
    for service in MONITORED_SERVICES:
        # Convert 'api-gateway' to 'api_gateway' to match your config variable naming
        prefix = service.replace("-", "_")
        
        # 1. CPU Usage (0% to 80% baseline)
        data[f"{prefix}_cpu_usage"] = np.random.uniform(5.0, 80.0, num_rows)
        
        # 2. Memory Usage (10% to 90% baseline)
        data[f"{prefix}_mem_usage"] = np.random.uniform(10.0, 90.0, num_rows)
        
        # 3. Network RX (Receive) in kbps (Wide range: 100 to 5000 kbps)
        data[f"{prefix}_net_rx_kbps"] = np.random.uniform(100.0, 5000.0, num_rows)
        
        # 4. Network TX (Transmit) in kbps (Wide range: 50 to 3000 kbps)
        data[f"{prefix}_net_tx_kbps"] = np.random.uniform(50.0, 3000.0, num_rows)
        
        # 5. Restart Count (Usually 0, occasionally jumps up)
        # Using a Poisson distribution so mostly 0s and 1s
        data[f"{prefix}_restart_count"] = np.random.poisson(0.1, num_rows).astype(float)
        
        # 6. Readiness State (1.0 = Ready, 0.0 = Not Ready)
        # Assuming services are ready 98% of the time
        data[f"{prefix}_is_ready"] = np.random.choice([1.0, 0.0], p=[0.98, 0.02], size=num_rows)

    # Convert dictionary to DataFrame and ENSURE column order matches config exactly
    df = pd.DataFrame(data)
    df = df[EXPECTED_METRICS] 
    return df

def preprocess_and_save(df, output_csv="training_data_v2.csv", scaler_path="scaler_v2.save"):
    print("Normalizing data ranges using MinMaxScaler...")
    
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_values, columns=EXPECTED_METRICS)
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Save SCALED data for PyTorch training
    scaled_csv_path = os.path.join(data_dir, output_csv)
    scaled_df.to_csv(scaled_csv_path, index=False)
    
    # 2. NEW: Save RAW data for the Simulator to stream
    raw_csv_path = os.path.join(data_dir, "raw_telemetry_stream.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"Saved RAW live stream data to {raw_csv_path}")
    
    # 3. Save the scaler
    scaler_full_path = os.path.join(data_dir, "saved_models", scaler_path)
    os.makedirs(os.path.dirname(scaler_full_path), exist_ok=True)
    joblib.dump(scaler, scaler_full_path)
    
    return scaled_df

if __name__ == "__main__":
    # 1. Generate the raw mock data
    raw_df = generate_synthetic_telemetry(num_rows=10000)
    
    # 2. Scale the data and save the scaler for your live inference script
    processed_df = preprocess_and_save(raw_df)
    
    print("\nShape of final training tensor:", processed_df.shape)
    print("Ready for PyTorch/Scikit-Learn training!")