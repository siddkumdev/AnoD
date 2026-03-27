import torch
import torch.nn as nn
from src import config

class TelemetryAutoencoder(nn.Module):
    def __init__(self):
        super(TelemetryAutoencoder, self).__init__()
        
        # Dynamically size the input based on the API Contract in config.py.
        # This ensures the model perfectly matches your dataset length.
        input_dim = len(config.EXPECTED_METRICS)
        
        # --- ENCODER ---
        # Compresses the incoming metrics into a smaller "latent" pattern.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(True),
            nn.Linear(8, 4),      # The "Bottleneck"
            nn.ReLU(True)
        )
        
        # --- DECODER ---
        # Attempts to rebuild the original metrics from the compressed pattern.
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, input_dim),
            # We use Sigmoid here because the preprocessor scaled our data between 0.0 and 1.0
            nn.Sigmoid() 
        )

    def forward(self, x):
        """Passes the data through the network."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def predict_anomaly(self, original_tensor):
        """
        Helper method for the live API. 
        Calculates how badly the model failed to reconstruct the data.
        """
        # Ensure we don't accidentally train the model during a live prediction
        self.eval() 
        with torch.no_grad():
            reconstructed_tensor = self(original_tensor)
            
            # Calculate Mean Squared Error (MSE) per row
            criterion = nn.MSELoss(reduction='none')
            errors = criterion(reconstructed_tensor, original_tensor).mean(dim=1)
            
            # Compare the error against your hardcoded threshold in config.py
            is_anomaly = errors > config.ANOMALY_THRESHOLD
            
            return is_anomaly, errors


