import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from src import config
from src.model import TelemetryAutoencoder
from src.preprocessor import enforce_feature_contract

# 1. Load the Model
print("Loading model for inspection...")
model = TelemetryAutoencoder()
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
model.eval()

# 2. Load the Anomaly Data
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mock_anomaly_telemetry.csv")
df = pd.read_csv(csv_path)

# Grab row 1150 (We know this is right in the middle of the chaos spike)
print("\n🔍 EXAMINING ROW 1150 (ANOMALY DETECTED)")
print("-" * 65)

# Extract just this one row as a DataFrame
single_row_df = df.iloc[[1150]]

# 3. Preprocess it exactly like the API does
clean_df = enforce_feature_contract(single_row_df)
input_tensor = torch.tensor(clean_df.values, dtype=torch.float32)

# 4. Ask the model to reconstruct it
with torch.no_grad():
    reconstructed_tensor = model(input_tensor)

# 5. Print the Side-by-Side Breakdown
print(f"{'METRIC NAME':<25} | {'ORIGINAL':<10} | {'AI REBUILT':<10} | {'ERROR (DIFF)':<10}")
print("-" * 65)

total_mse = 0.0

for i, metric_name in enumerate(config.EXPECTED_METRICS):
    # We multiply by 100 to reverse the preprocessing scaling so it's readable
    original_val = input_tensor[0][i].item() * 100
    recon_val = reconstructed_tensor[0][i].item() * 100
    
    # Calculate the absolute difference
    diff = abs(original_val - recon_val)
    
    # Calculate the squared error for the final score
    mse_part = ((input_tensor[0][i].item() - reconstructed_tensor[0][i].item()) ** 2)
    total_mse += mse_part
    
    # Highlight the specific metric that is causing the massive error
    if diff > 20.0:
        print(f"\033[91m{metric_name:<25} | {original_val:<10.2f} | {recon_val:<10.2f} | {diff:<10.2f} <-- THE CULPRIT\033[0m")
    else:
        print(f"{metric_name:<25} | {original_val:<10.2f} | {recon_val:<10.2f} | {diff:<10.2f}")

print("-" * 65)
print(f"Final Average Error Score: {total_mse / len(config.EXPECTED_METRICS):.4f}")
print("If the score is > 0.02, the system triggers the alert.")