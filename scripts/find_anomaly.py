import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src import config
from src.model import TelemetryAutoencoder
from src.preprocessor import enforce_feature_contract

# Terminal Colors
CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'

def find_best_threshold():
    print(f"{CYAN}Calculating optimal anomaly threshold...{RESET}")
    
    # 1. Load Model
    model = TelemetryAutoencoder()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    # 2. Load Data
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mock_anomaly_telemetry.csv")
    df = pd.read_csv(csv_path)
    
    # 3. Ground Truth
    y_true = np.zeros(len(df))
    y_true[1000:1300] = 1  
    y_true[1400:1600] = 1  
    
    # 4. Get Raw Error Scores (Not True/False, but the actual MSE floats)
    clean_df = enforce_feature_contract(df)
    input_tensor = torch.tensor(clean_df.values, dtype=torch.float32)
    
    error_scores = []
    with torch.no_grad():
        for i in range(len(input_tensor)):
            single_row = input_tensor[i].unsqueeze(0)
            _, mse_score = model.predict_anomaly(single_row)
            error_scores.append(mse_score.item())

    # 5. Use Scikit-Learn to test every possible threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, error_scores)
    
    # Calculate F1 scores for all thresholds to find the mathematical peak
    # Adding a tiny epsilon to avoid division by zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    
    print(f"\n{GREEN}=== OPTIMAL THRESHOLD FOUND ==={RESET}")
    print(f"Set ANOMALY_THRESHOLD = {best_threshold:.6f}")
    print("-" * 30)
    print(f"Expected Performance at this threshold:")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall:    {best_recall:.4f}")
    print(f"F1-Score:  {best_f1:.4f}")

if __name__ == "__main__":
    find_best_threshold()