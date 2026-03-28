import os
import joblib
import pandas as pd
import torch
from . import config

# We load the scaler once and cache it in memory. 
# Loading it from the hard drive every single second during a live stream would cause terrible lag.
_scaler = None

def enforce_feature_contract(metrics_dict):
    global _scaler
    
    # 1. Load the scaler if it hasn't been loaded yet
    if _scaler is None:
        scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "saved_models", "scaler_v2.save"))
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Did you run generate_mock_data.py?")
        _scaler = joblib.load(scaler_path)

    # 2. Extract exactly the 36 features in the exact right order
    ordered_values = []
    for feature in config.EXPECTED_METRICS:
        # If a metric is missing from the stream, use the safe default (0.0)
        val = metrics_dict.get(feature, config.MISSING_DATA_DEFAULT)
        ordered_values.append(val)
        
    # 3. Scikit-learn expects a 2D structure (like a DataFrame with 1 row)
    df_single_row = pd.DataFrame([ordered_values], columns=config.EXPECTED_METRICS)
    
    # 4. Scale the data (transforms your 0-5000 kbps into 0.0-1.0 ranges)
    scaled_numpy_array = _scaler.transform(df_single_row)
    
    # 5. Convert to PyTorch Tensor and flatten back to 1D
    tensor_1d = torch.tensor(scaled_numpy_array[0], dtype=torch.float32)
    return tensor_1d