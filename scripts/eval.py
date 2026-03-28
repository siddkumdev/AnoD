import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add root directory to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src import config
from src.model import TelemetryAutoencoder
from src.preprocessor import enforce_feature_contract

# Terminal Colors
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
CYAN = '\033[96m'

def evaluate():
    print(f"{CYAN}Loading Model & RAW Data for Evaluation...{RESET}")
    
    # 1. Load Model (Dynamically sizes to 30 features based on config)
    model = TelemetryAutoencoder()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    
    # 2. Load RAW Data (The stream-ready CSV)
    csv_path = os.path.join(ROOT_DIR, "data", "raw_telemetry_stream.csv")
    if not os.path.exists(csv_path):
        print(f"{RED}Error: {csv_path} not found. Run generate_mock_data.py first.{RESET}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 3. Define the Ground Truth (Mapping to your mock data scenarios)
    y_true = np.zeros(len(df))
    # Adjust these ranges if you changed your start_row in the simulator
    y_true[1000:1300] = 1  # Scenario A
    y_true[1400:1600] = 1  # Scenario B
    
    # 4. Get Model Predictions
    y_pred = []
    print(f"{CYAN}Processing {len(df)} rows through the RCA Pipeline...{RESET}")
    
    with torch.no_grad():
        for _, row in df.iterrows():
            # Convert row to dictionary to satisfy the Preprocessor's contract
            metrics_dict = row.to_dict()
            
            # 1. Scale via Preprocessor (Transforms raw 0-5000 range to 0-1)
            input_tensor = enforce_feature_contract(metrics_dict).unsqueeze(0)
            
            # 2. Run Inference (Unpacking 3 values now!)
            is_anomaly, _, _ = model.predict_anomaly(input_tensor)
            
            # 3. Store prediction
            y_pred.append(1 if is_anomaly.item() else 0)

    # 5. Print the Results
    print(f"\n{GREEN}=== ML PERFORMANCE REPORT (30-FEATURE RCA MODEL) ==={RESET}")
    
    print("\n1. Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    # Handle cases where the mock data might be smaller than the index range
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    print(f"True Negatives (Correctly ignored healthy):  {tn}")
    print(f"False Positives (False Alarms):              {fp}")
    print(f"False Negatives (Missed an anomaly):         {fn}")
    print(f"True Positives (Successfully caught bug):    {tp}")
    
    print("\n2. Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Healthy (0)", "Anomaly (1)"], digits=4))

if __name__ == "__main__":
    evaluate()