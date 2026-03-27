import torch
from torch.utils.data import Dataset
import pandas as pd
from src.preprocessor import enforce_feature_contract

class TelemetryDataset(Dataset):
    def __init__(self, csv_filepath: str):
        """
        Loads the CSV and prepares the fixed feature vectors.
        """
        # Load the raw mock data
        raw_df = pd.read_csv(csv_filepath)
        
        # Pass it through your shield to guarantee the shape is correct
        clean_df = enforce_feature_contract(raw_df)
        
        # Convert the clean Pandas DataFrame into a PyTorch Tensor (matrix of numbers)
        # dtype=torch.float32 is standard for PyTorch weights
        self.features = torch.tensor(clean_df.values, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of rows in your dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        PyTorch calls this function in a loop during training to grab one row at a time.
        """
        # Return a single row (one timestamp's worth of metrics)
        return self.features[idx]

# --- Quick Test Block ---
# If you run this file directly, it will test itself without breaking the rest of the app.
if __name__ == "__main__":
    import os
    
    # Create a dummy CSV path for testing
    dummy_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_healthy_telemetry.csv")
    
    # Only run the test if you've actually created the mock CSV file
    if os.path.exists(dummy_path):
        dataset = TelemetryDataset(dummy_path)
        print(f"Dataset loaded successfully! Total rows: {len(dataset)}")
        print(f"Shape of first row: {dataset[0].shape}") 
        # Expected output: torch.Size([10]) (because you have 10 metrics in config.py)
    else:
        print(f"Please create the mock CSV at {dummy_path} to test the dataset.")