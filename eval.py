import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.model import TelemetryAutoencoder
from src.preprocessor import enforce_feature_contract

# Terminal Colors
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
CYAN = '\033[96m'

def evaluate():
    print(f"{CYAN}Loading Model & Data for Evaluation...{RESET}")
    
    # 1. Load Model
    model = TelemetryAutoencoder()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    # 2. Load Data
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mock_anomaly_telemetry.csv")
    df = pd.read_csv(csv_path)
    
    # 3. Define the Ground Truth (What ACTUALLY happened based on our mock script)
    # 0 = Healthy, 1 = Anomaly
    y_true = np.zeros(len(df))
    y_true[1000:1300] = 1  # Scenario A: Infinite Loop
    y_true[1400:1600] = 1  # Scenario B: Crash Loop
    
    # 4. Get Model Predictions
    y_pred = []
    print(f"{CYAN}Running {len(df)} rows through the Autoencoder...{RESET}\n")
    
    # Process in bulk for speed
    clean_df = enforce_feature_contract(df)
    input_tensor = torch.tensor(clean_df.values, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(len(input_tensor)):
            single_row = input_tensor[i].unsqueeze(0)
            is_anomaly, _ = model.predict_anomaly(single_row)
            # Convert boolean tensor to 1 or 0
            y_pred.append(1 if is_anomaly.item() else 0)

    # 5. Print the Results
    print(f"{GREEN}=== ML PERFORMANCE REPORT ==={RESET}")
    
    print("\n1. Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Negatives (Correctly ignored healthy):  {cm[0][0]}")
    print(f"False Positives (False Alarms):              {cm[0][1]}")
    print(f"False Negatives (Missed an anomaly):         {cm[1][0]}")
    print(f"True Positives (Successfully caught bug):    {cm[1][1]}")
    
    print("\n2. Classification Report:")
    # target_names: 0 is Healthy, 1 is Anomaly
    print(classification_report(y_true, y_pred, target_names=["Healthy (0)", "Anomaly (1)"], digits=4))

if __name__ == "__main__":
    evaluate()