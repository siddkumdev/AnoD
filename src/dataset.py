import pandas as pd
import torch
from torch.utils.data import Dataset
from . import config

class TelemetryDataset(Dataset):
    def __init__(self, csv_path):
        # 1. Read the training data (which is ALREADY scaled by generate_mock_data.py)
        df = pd.read_csv(csv_path)
        
        # 2. Extract ONLY the exact 30 features you want, in the exact order.
        # This acts as a safety net: if "frontend" is accidentally still in the CSV, 
        # it gets ignored here because it's no longer in config.EXPECTED_METRICS.
        ordered_data = df[config.EXPECTED_METRICS].values
        
        # 3. Convert directly to PyTorch Tensor (bypassing preprocessor.py)
        self.data = torch.tensor(ordered_data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]